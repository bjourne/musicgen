# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from collections import namedtuple
from construct import Container
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from keras.utils.np_utils import to_categorical
from musicgen.formats.modules import *
from musicgen.formats.modules.corpus import load_index
from musicgen.formats.modules.parser import (Cell, PowerPackerModule,
                                             load_file)
from musicgen.utils import SP
from pathlib import Path
from pickle import dump, load
from random import choice, randint, randrange
from sys import argv
import numpy as np

def compute_advances(delta):
    thresholds = [64, 32, 16, 8, 4, 3, 2, 1]
    for threshold in thresholds:
        while delta >= threshold:
            yield threshold
            delta -= threshold
    assert delta == 0

def channel_to_sequence(rows, col_idx):
    yield 'break', 0
    last_row = None
    last_sample = None
    for row_idx, row in enumerate(rows):
        cell = row[col_idx]
        sample = cell.sample_idx
        note = -1 if not cell.period else period_to_idx(cell.period)
        sample_diff = sample != 0 and sample != last_sample

        # This skip can cause the wrong sample to be selected for some
        # notes. But most of the time, it won't.
        if sample != 0 and note == -1:
            continue
        if not sample <= 0x1f:
            SP.print('Skipping invalid sample %d in module.', sample)
            continue

        if sample_diff or note != -1:
            if last_row is None:
                delta = row_idx
            else:
                delta = row_idx - last_row - 1
            assert delta >= 0
            for adv in compute_advances(delta):
                assert adv > 0
                yield 'advance', adv
            last_row = row_idx
        if sample_diff:
            yield 'set_sample', sample
            last_sample = sample
        if note != -1:
            yield 'play', note
    if last_row is not None:
        n_trailer = len(rows) - last_row - 1
    else:
        n_trailer = len(rows)
    for adv in compute_advances(n_trailer):
        assert adv > 0
        yield 'advance', adv

def rows_to_sequence(rows):
    for col_idx in range(4):
        for ev in channel_to_sequence(rows, col_idx):
            yield ev

def mod_to_sequence(fname):
    SP.print(str(fname))
    try:
        mod = load_file(fname)
    except PowerPackerModule:
        SP.print('Skipping PP20 module.')
        return []
    rows = linearize_rows(mod)
    return rows_to_sequence(rows)

def build_cell(sample, note, effect_cmd, effect_arg):
    if note == -1:
        period = 0
    else:
        period = PERIODS[note]
    sample_lo = sample & 0xf
    sample_hi = sample >> 4
    sample_idx = (sample_hi << 4) + sample_lo
    effect_arg1 = effect_arg >> 4
    effect_arg2 = effect_arg & 0xf
    return Container(dict(period = period,
                          sample_lo = sample_lo,
                          sample_hi = sample_hi,
                          sample_idx = sample_idx,
                          effect_cmd = effect_cmd,
                          effect_arg1 = effect_arg1,
                          effect_arg2 = effect_arg2))

ZERO_CELL = build_cell(0, -1, 0, 0)

def sequence_to_rows(seq):
    col_idx = -1
    sample = None
    cols = [[], [], [], []]
    advances = [0, 0, 0, 0]
    for cmd, arg in seq:
        if cmd == 'break':
            col_idx += 1
        if cmd == 'set_sample':
            sample = arg
        if cmd == 'play':
            assert sample != 0
            # Flush saved advances for col.
            spacing = [ZERO_CELL for _ in range(advances[col_idx])]
            cols[col_idx].extend(spacing)
            advances[col_idx] = 0
            cell = build_cell(sample, arg, 0, 0)
            cols[col_idx].append(cell)
        if cmd == 'advance':
            advances[col_idx] += arg

    for col, advance in zip(cols, advances):
        col.extend([ZERO_CELL for _ in range(advance)])
    return list(zip(*cols))

def get_sequence_from_disk(corpus_path, mods):
    SP.header('PARSING', '%d modules', len(mods))
    fnames = [corpus_path / mod.genre / mod.fname for mod in mods]
    seq = sum([list(mod_to_sequence(fname))
               for fname in fnames], [])
    SP.leave()
    return seq

def get_sequence(corpus_path, model_path):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= 150)]

    key = sum(mod.kb_size for mod in mods)
    cache_file = 'cache-064-%010d.pickle' % key
    cache_path = model_path / cache_file
    if not cache_path.exists():
        model_path.mkdir(parents = True, exist_ok = True)
        seq = get_sequence_from_disk(corpus_path, mods)
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

def make_model(seq_len, n_chars):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, n_chars)))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr = 0.01)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer)
    print(model.summary())
    return model

def weighted_sample(probs, temperature):
    probs = np.array(probs).astype('float64')
    probs = np.log(probs) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    probas = np.random.multinomial(1, probs, 1)
    return np.argmax(probas)

def generate(model, pattern, n_notes, n_chars, temp):
    for i in range(n_notes):
        inp = np.reshape(pattern, (1, len(pattern), 1))
        inp = inp / n_chars
        pred = model.predict(inp, verbose = 0)[0]

        idx = np.argmax(pred)
        if temp is None:
            rnd_idx = np.random.choice(len(pred), p = pred)
        else:
            rnd_idx = weighted_sample(pred, temp)
        if i % 100 == 0:
            fmt = '#%3d %3d %.4f %3d %.4f'
            print(fmt % (i, idx, pred[idx], rnd_idx, pred[rnd_idx]))
        yield rnd_idx
        pattern = pattern[1:] + [rnd_idx]

def generate_sequence(model, sel_X, n_chars, n_cells,
                      int2el, temp, fname):
    fmt = 'Generating %d rows with temperature %.2f to "%s".'
    SP.print(fmt, (n_cells / 4, temp or 0.0, fname))
    indices = generate(model, sel_X, n_notes, n_chars, temp)
    cells = [int2el[idx] for idx in indices]
    SP.print('Cells %s', cells)
    with open(fname, 'wb') as f:
        dump(cells, f)

def generate_sequences(model, X, n_chars, n_notes, int2el):
    SP.header('GENERATE')
    sel_X = choice(X)
    for temp in [None, 0.2, 0.5, 1.0, 1.2]:
        if temp is None:
            fname = 'test_rnd'
        else:
            fname = 'test_%.2f' % temp
        fname = fname.replace('.', '_') + '.pickle'
        generate_sequence(model, sel_X, n_chars, n_notes,
                          int2el, temp, fname)
    SP.leave()

class CustomGenerator(Sequence):
    def __init__(self, int_seq, batch_size, seq_len, n_chars):
        self.int_seq = int_seq
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_chars = n_chars

    def __len__(self):
        return int(np.ceil(len(self.int_seq) / self.batch_size))

    def __getitem__(self, i):
        base_idx = i * self.batch_size
        X = np.zeros((self.batch_size, self.seq_len, self.n_chars),
                     dtype = np.bool)
        Y = np.zeros((self.batch_size, self.n_chars),
                     dtype = np.bool)
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                X[i, j, self.int_seq[base_idx + i + j]] = 1
            Y[i, self.int_seq[base_idx + i + self.seq_len]] = 1
        return X, Y

def run_training(corpus_path, model_path, seq_len, step):
    seq = get_sequence(corpus_path, model_path)

    # Different tokens in sequence
    chars = sorted(set(seq))
    n_chars = len(chars)
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}

    # Convert to integer sequence
    int_seq = np.array([char2idx[c] for c in seq])

    SP.print(f'Training model with %d tokens and %d characters.',
             (len(seq), n_chars))

    model = make_model(seq_len, n_chars)

    weights_path = model_path / 'weights.hdf5'
    if weights_path.exists():
        SP.print(f'Loading existing weights from "{weights_path}".')
        model.load_weights(weights_path)

    batch_size = 128
    gen = CustomGenerator(int_seq, batch_size, seq_len, n_chars)

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

    callbacks = [cb_checkpoint, cb_generate]
    model.fit_generator(generator = gen,
                        steps_per_epoch = len(seq) // batch_size,
                        epochs = 10,
                        verbose = 1,
                        shuffle = False,
                        callbacks = callbacks)

def main_cmd_line():
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
        help = 'Length of training sequence')
    args = parser.parse_args()
    SP.enabled = args.info

    corpus_path = Path(args.corpus_path)
    model_path = Path(args.model_path)

    # Should step be configurable too?
    seq_len = args.seq_len
    step = 1

    run_training(corpus_path, model_path, seq_len, step)

def parse_unparse():
    fname = argv[1]
    mod = load_file(fname)
    rows = linearize_rows(mod)
    print('BEFORE')
    print(rows_to_string(rows, numbering = True))
    seq = rows_to_sequence(rows)
    print('AFTER')
    rows = sequence_to_rows(seq)
    print(rows_to_string(rows))

if __name__ == '__main__':
    #parse_unparse()
    main_cmd_line()
