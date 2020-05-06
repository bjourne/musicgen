# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from musicgen.formats.modules import *
from musicgen.formats.modules.corpus import load_index
from musicgen.formats.modules.parser import load_file
from musicgen.utils import SP
from pathlib import Path
from pickle import dump, load
import numpy as np

def cell_to_scalar(cell, col_idx, channel_samples):
    sample_idx = cell.sample_idx
    period = cell.period
    if not period:
        return 0
    if not sample_idx:
        sample_idx = channel_samples.get(col_idx, 0)
    channel_samples[col_idx] = sample_idx
    return 60 * sample_idx + period_to_idx(period)

def mod_to_sequence(fname):
    SP.print(str(fname))
    mod = load_file(fname)
    rows = linearize_rows(mod)
    channel_samples = {}
    for row in rows:
        for col_idx, cell in enumerate(row):
            yield cell_to_scalar(cell, col_idx, channel_samples)

def get_sequence_from_disk(corpus_path, mods, seq_len):
    SP.header('PARSING', '%d modules', len(mods))
    padding = [0] * seq_len
    fnames = [corpus_path / mod.genre / mod.fname for mod in mods]
    seq = sum([list(mod_to_sequence(fname)) + padding
               for fname in fnames], [])
    SP.leave()
    return seq

def get_sequence(corpus_path, model_path, seq_len):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= 150)]

    key = sum(mod.kb_size for mod in mods)
    cache_file = 'cache-%03d-%010d.pickle' % (seq_len, key)
    cache_path = model_path / cache_file
    if not cache_path.exists():
        model_path.mkdir(parents = True, exist_ok = True)
        seq = get_sequence_from_disk(corpus_path, mods, seq_len)
        with open(cache_path, 'wb') as f:
            dump(seq, f)
    else:
        SP.print('Using cache at %s.', cache_path)
    assert cache_path.exists()
    with open(cache_path, 'rb') as f:
        return load(f)

def create_samples(seq, seq_len, step):
    for i in range(0, len(seq) - seq_len, step):
        inp = seq[i:i + seq_len]
        out = seq[i + seq_len]
        yield inp, out

def make_model(seq_len, n_vocab):
    m = Sequential()
    m.add(LSTM(
        512,
        input_shape=(seq_len, 1),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    m.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    m.add(LSTM(512))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(256))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(n_vocab))
    m.add(Activation('softmax'))
    m.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop')
    return m

def train(model, X, Y, n_vocab, seq_len, int2el, weights_path):
    n_samples = len(X)
    SP.print(f'Training model with {n_samples} samples.')
    X_inp = np.reshape(X, (n_samples, seq_len, 1))
    X_inp = X_inp / n_vocab
    Y = to_categorical(Y)

    cb_checkpoint = ModelCheckpoint(
        str(weights_path),
        monitor = 'loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min'
    )

    def on_epoch_begin(n_epoch, logs):
        pass
    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)

    print(X_inp.shape, Y.shape)

    callbacks = [cb_checkpoint, cb_generate]
    model.fit(X_inp, Y,
              epochs = 200, batch_size = 128,
              callbacks = callbacks)

def main():
    parser = ArgumentParser(description = 'Music ML')
    parser.add_argument(
        '--corpus-path', required = True,
        help = 'Path to corpus')
    parser.add_argument(
        '--model-path', required = True,
        help = 'Path to store model and cache.')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'Print information')
    parser.add_argument(
        '--seq-len', required = True, type = int,
        help = 'Length of training sequence (multiple of 4)')
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    model_path = Path(args.model_path)

    # Should step be configurable too?
    seq_len = args.seq_len
    step = 3

    seq = get_sequence(corpus_path, model_path, seq_len)

    # Different tokens in sequence
    chars = sorted(set(seq))
    n_vocab = len(chars)
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}

    # Convert to integer sequence
    int_seq = [char2idx[c] for c in seq]

    SP.print('%d tokens and %d characters in training sequence.',
             (len(seq), n_vocab))

    # Now we create training samples
    data = list(create_samples(int_seq, seq_len, step))
    X, Y = zip(*data)

    model = make_model(seq_len, n_vocab)
    weights_path = model_path / 'weights.hdf5'
    train(model, X, Y, n_vocab, seq_len, idx2char, weights_path)


if __name__ == '__main__':
    main()
