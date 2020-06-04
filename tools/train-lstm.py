# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Monophonic model

Usage:
    train-mono-model.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --win-size=<int>       window size [default: 64]
    --kb-limit=<int>       kb limit [default: 150]
    --pack-mcode           use packed mcode
    --fraction=<float>     fraction of corpus to use [default: 1.0]
"""
from docopt import docopt
from musicgen.ml import generate_sequence
from musicgen.mycode import (INSN_JUMP,
                             load_corpus,
                             mcode_to_midi_file,
                             mcode_to_string)
from musicgen.utils import (SP,
                            analyze_code,
                            encode_training_sequence,
                            file_name_for_params, flatten,
                            load_pickle_cache)
from os import environ
from pathlib import Path
from pickle import dump, load
from random import randrange, shuffle
from tensorflow.keras import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np

def flatten_corpus(corpus_path, kb_limit, pack_mcode, fraction):
    mcode_mods = load_corpus(corpus_path, kb_limit, pack_mcode)
    n_mods = len(mcode_mods)
    params = (n_mods, kb_limit, pack_mcode, fraction)
    cache_file = file_name_for_params('cached_mcode_flat',
                                      'pickle', params)
    cache_path = corpus_path / cache_file

    def rebuild_fun():
        seqs = [[c[1] for c in mcode_mod.cols]
                for mcode_mod in mcode_mods]
        seqs = flatten(seqs)
        seqs = seqs[:int(len(seqs) * fraction)]
        shuffle(seqs)
        for seq in seqs:
            seq.append((INSN_JUMP, 64))
        return encode_training_sequence(flatten(seqs))
    return load_pickle_cache(cache_path, rebuild_fun)

def generate_midi_files(model, epoch, seq, win_size,
                        ch2ix, ix2ch, corpus_path):
    SP.header('EPOCH', '%d', epoch)
    # Pick a seed that doesn't contain padding
    pad_int = ch2ix[(INSN_JUMP, 64)]
    while True:
        idx = randrange(len(seq) - win_size)
        seed = seq[idx:idx + win_size]
        if not pad_int in seed:
            break

    # One hot seed
    seed1h = to_categorical(seed, len(ch2ix))

    # So that you can hear the transition from seed to generated data.
    join_token = ch2ix[(INSN_JUMP, 8)]

    temps = [0.2, 0.5, 1.0, 1.2, 1.5]
    for temp in temps:
        log_lh, seq = list(generate_sequence(model, seed1h, 300, temp))
        seq = seed.tolist() + [join_token] + seq
        seq = [ix2ch[i] for i in seq]
        SP.header('TEMPERATURE %.2f' % temp)
        SP.print(mcode_to_string(seq))
        file_name = 'gen-%03d-%.2f.mid' % (epoch, temp)
        file_path = corpus_path / file_name
        mcode_to_midi_file(seq, file_path, 120, None)
        SP.leave()
    SP.leave()

def make_model(seq_len, n_chars):
    model = Sequential()
    model.add(LSTM(128, return_sequences = True,
                   input_shape = (seq_len, n_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    print(model.summary())
    return model

class OneHotGenerator(Sequence):
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
        Y = np.zeros((batch_size, self.vocab_size),
                     dtype = np.bool)
        for i in range(batch_size):
            for j in range(self.win_size):
                X[i, j, self.seq[base + i + j]] = 1
            Y[i, self.seq[base + i + self.win_size]] = 1
        return X, Y

def main():
    args = docopt(__doc__, version = 'Monophonic model 1.0')
    SP.enabled = args['--verbose']

    corpus_path = Path(args['<corpus-path>'])
    win_size = int(args['--win-size'])
    kb_limit = int(args['--kb-limit'])
    pack_mcode = args['--pack-mcode']
    fraction = float(args['--fraction'])

    ix2ch, ch2ix, seq = flatten_corpus(corpus_path, kb_limit,
                                       pack_mcode, fraction)
    analyze_code(ix2ch, seq)

    n_seq = len(seq)
    vocab_size = len(ix2ch)

    # Split data
    n_train = int(n_seq * 0.8)
    n_validate = int(n_seq * 0.1)
    n_test = n_seq - n_train - n_validate
    train = seq[:n_train]
    validate = seq[n_train:n_train + n_validate]
    test = seq[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))

    # Path to weights file
    params = (win_size, n_train, n_validate, pack_mcode)
    weights_file = file_name_for_params('mcode_weights', 'h5', params)
    weights_path = corpus_path / weights_file

    model = make_model(win_size, vocab_size)
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(weights_path)
    else:
        SP.print(f'Weights file {weights_path} not found.')

    batch_size = 128
    train_gen = OneHotGenerator(train, batch_size, win_size, vocab_size)
    validate_gen = OneHotGenerator(validate,
                                   batch_size, win_size, vocab_size)
    cb_checkpoint = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min')
    def on_epoch_begin(epoch, logs):
        generate_midi_files(model, epoch, test, win_size,
                            ch2ix, ix2ch, corpus_path)
    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)

    model.fit(x = train_gen,
              steps_per_epoch = len(train_gen),
              validation_data = validate_gen,
              validation_steps = len(validate_gen),
              verbose = 1,
              shuffle = True,
              epochs = 10,
              callbacks = [cb_checkpoint, cb_generate])

if __name__ == '__main__':
    main()
