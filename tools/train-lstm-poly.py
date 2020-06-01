# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Polyphonic LSTM

Usage:
    train-lstm-poly.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --win-size=<int>       window size [default: 64]
    --kb-limit=<int>       kb limit [default: 150]
    --fraction=<float>     fraction of corpus to use [default: 1.0]
    --relative-pitches     use pcode with relative pitches
"""
from collections import Counter
from docopt import docopt
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from musicgen.ml import generate_sequence, train_model
from musicgen.pcode import (INSN_SILENCE, PAD_TOKEN,
                            load_data, mod_file_to_pcode,
                            pcode_to_string, pcode_to_midi_file)
from musicgen.utils import SP, file_name_for_params
from pathlib import Path
from random import randrange, shuffle
import numpy as np


########################################################################
# Model definition
########################################################################
def make_model(seq_len, vocab_size):
    # Check if an Embedding layer is worthwhile.
    model = Sequential()
    model.add(LSTM(128, return_sequences = True,
                   input_shape = (seq_len, vocab_size)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    print(model.summary())
    return model

########################################################################
# Data generator
########################################################################
class DataGen(Sequence):
    def __init__(self, seq, batch_size, win_size, vocab_size):
        self.seq = seq
        self.batch_size = batch_size
        self.win_size = win_size
        self.vocab_size = vocab_size

    def __len__(self):
        n_windows = len(self.seq) - self.win_size
        return int(np.ceil(n_windows / self.batch_size))

    def __getitem__(self, i):
        base = i * self.batch_size

        # Fix running over the edge.
        n_windows = len(self.seq) - self.win_size
        batch_size = min(n_windows - base, self.batch_size)

        X = np.zeros((batch_size, self.win_size, self.vocab_size),
                     dtype = np.bool)
        Y = np.zeros(batch_size)
        for i in range(batch_size):
            for j in range(self.win_size):
                X[i, j, self.seq[base + i + j]] = 1
            Y[i] = self.seq[base + i + self.win_size]
        return X, Y


########################################################################
# Generation
########################################################################
def generate_midi_files(model, epoch, seq,
                        vocab_size, win_size,
                        char2idx, idx2char, corpus_path,
                        relative_pitches):
    SP.header('EPOCH', '%d', epoch)
    # Pick a seed that doesn't contain the break token.
    eos = char2idx[PAD_TOKEN]
    while True:
        idx = randrange(len(seq) - win_size)
        seed = np.array(seq[idx:idx + win_size])
        if not eos in seed:
            break
    join_seq = [char2idx[(INSN_SILENCE, 8)]] * 4

    i = randrange(len(seq) - win_size)
    seed = list(seq[i:i + win_size])
    S = to_categorical(seed, vocab_size)
    temps = [0.5, 0.8, 1.0, 1.2, 1.5]
    for temp in temps:
        SP.header('TEMPERATURE %.2f' % temp)
        log_lh, seq = generate_sequence(model, S, 300, temp, eos)

        seq = seed + join_seq + seq
        seq = [idx2char[i] for i in seq]
        SP.print('logLH: %.4f', log_lh)
        SP.print(pcode_to_string(seq))
        file_name = 'pcode-%03d-%.2f.mid' % (epoch, temp)
        file_path = corpus_path / file_name
        pcode_to_midi_file(seq, file_path, relative_pitches)
        SP.leave()
    SP.leave()

def analyze_pcode(pcode):
    counts = Counter(pcode)
    for (cmd, arg), cnt in sorted(counts.items()):
        SP.print('%s %3d %10d' % (cmd, arg, cnt))

def main():
    args = docopt(__doc__, version = 'Train Poly LSTM 1.0')
    SP.enabled = args['--verbose']
    kb_limit = int(args['--kb-limit'])
    corpus_path = Path(args['<corpus-path>'])
    win_size = int(args['--win-size'])
    relative_pitches = args['--relative-pitches']

    # Load dataset
    if corpus_path.is_dir():
        dataset = load_data(corpus_path, kb_limit, relative_pitches)
        output_dir = corpus_path
    else:
        dataset = list(mod_file_to_pcode(corpus_path, relative_pitches))
        output_dir = Path('.')
    n_dataset = len(dataset)
    analyze_pcode(dataset)

    # Convert to integer sequence
    n_dataset = len(dataset)
    chars = sorted(set(dataset))
    vocab_size = len(chars)
    SP.print('%d tokens and %d token types.', (n_dataset, vocab_size))
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}
    dataset = np.array([char2idx[c] for c in dataset])

    # Split data
    n_train = int(n_dataset * 0.8)
    n_validate = int(n_dataset * 0.1)
    n_test = n_dataset - n_train - n_validate
    train = dataset[:n_train]
    validate = dataset[n_train:n_train + n_validate]
    test = dataset[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))

    # Data generators
    batch_size = 128
    train_gen = DataGen(train, batch_size, win_size, vocab_size)
    validate_gen = DataGen(validate, batch_size, win_size, vocab_size)

    # Create model and maybe load from disk.
    model = make_model(win_size, vocab_size)

    params = (win_size, n_train, n_validate, relative_pitches)
    weights_file = file_name_for_params('weights_poly', 'hdf5', params)
    weights_path = output_dir / weights_file
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(weights_path)
    else:
        SP.print(f'Weights file {weights_path} not found.')

    def on_epoch_begin(epoch, logs):
        generate_midi_files(model, epoch, test,
                            vocab_size, win_size,
                            char2idx, idx2char, output_dir,
                            relative_pitches)
    cb_cp1 = ModelCheckpoint(str(weights_path))
    best_weights_path = '%s-best-{val_loss:.4f}.hdf5' % weights_path.stem
    cb_cp2 = ModelCheckpoint(
        str(best_weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min')

    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)
    model.fit(x = train_gen,
              steps_per_epoch = len(train_gen),
              validation_data = validate_gen,
              validation_steps = len(validate_gen),
              verbose = 1,
              shuffle = True,
              epochs = 20,
              callbacks = [cb_cp1, cb_cp2, cb_generate])

if __name__ == '__main__':
    main()
