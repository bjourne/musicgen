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
from musicgen.formats.modules import *
from musicgen.formats.modules.corpus import load_index
from musicgen.formats.modules.parser import (Cell, PowerPackerModule,
                                             load_file)
from musicgen.utils import OneHotGenerator, SP
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

def column_to_sequence(rows, col_idx):
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
            # Need to fix this somehow
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
    yield 'break', 0

def rows_to_sequence(rows):
    for col_idx in range(4):
        for ev in column_to_sequence(rows, col_idx):
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

def sequence_to_columns(seq):
    sample = None
    col = []
    advance = 0
    for cmd, arg in seq:
        if cmd == 'break':
            # Emit current colun
            col += [ZERO_CELL for _ in range(advance)]
            yield col
            col = []
            advance = 0
            sample = None
        if cmd == 'set_sample':
            sample = arg
        if cmd == 'play':
            # If sample is None the play command is invalid and we
            # have to ingore it.
            if sample is None:
                SP.print('Ignoring play command without sample.')
                continue

            # Flush saved advances for col.
            spacing = [ZERO_CELL for _ in range(advance)]
            col += spacing
            advance = 0
            cell = build_cell(sample, arg, 0, 0)
            col.append(cell)
        if cmd == 'advance':
            advance += arg
    col += [ZERO_CELL for _ in range(advance)]
    yield col

def columns_to_rows(cols):
    # Take the four first:
    cols = list(cols)[:4]

    # Pad with missing channels
    cols = cols + [[] for _ in range(4 - len(cols))]

    # Pad with empty cells
    max_len = max(len(col) for col in cols)
    cols = [col + [ZERO_CELL] * (max_len - len(col))
            for col in cols]
    return zip(*cols)

def sequence_to_rows(seq):
    return list(columns_to_rows(sequence_to_columns(seq)))

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

def generate_chars(model, seed, n_generate, temp, char2idx, idx2char):
    seq_len = len(seed)
    n_chars = len(char2idx)
    for _ in range(n_generate):
        X = np.zeros((1, seq_len, n_chars))
        for t, idx in enumerate(seed):
            X[0, t, idx] = 1
        P = model.predict(X, verbose = 0)[0]
        if temp is None:
            idx = np.random.choice(len(P), p = P)
        else:
            idx = weighted_sample(P, temp)
        yield idx
        seed = seed[1:] + [idx]

def generate_music(model, n_epoch, seq, seq_len,
                   char2idx, idx2char):
    SP.header('EPOCH', '%d', n_epoch)
    idx = randrange(len(seq) - seq_len)
    seed = list(seq[idx:idx + seq_len])
    for temp in [None, 0.2, 0.5, 1.0, 1.2]:
        fmt = '%s' if temp is None else '%.2f'
        SP.header('TEMPERATURE', fmt, temp)
        generated = [idx2char[i]
                     for i in generate_chars(model, seed,
                                             300, temp,
                                             char2idx, idx2char)]
        SP.print(str(generated))
        rows = sequence_to_rows(generated)
        for row in rows:
            SP.print(row_to_string(row))
        SP.leave()
    SP.leave()

def run_training(corpus_path, model_path, win_size, step):
    seq = get_sequence(corpus_path, model_path)


    # Different tokens in sequence
    chars = sorted(set(seq))
    vocab_size = len(chars)
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}

    # Cut sequence
    seq = seq[:len(seq) // 100]

    # Convert to integer sequence
    int_seq = np.array([char2idx[c] for c in seq])

    SP.print(f'Training model with %d tokens and %d characters.',
             (len(seq), vocab_size))

    model = make_model(win_size, vocab_size)

    weights_path = model_path / 'weights.hdf5'
    if weights_path.exists():
        SP.print(f'Loading existing weights from "{weights_path}".')
        model.load_weights(weights_path)

    batch_size = 64
    gen = OneHotGenerator(int_seq, batch_size, win_size, vocab_size)

    cb_checkpoint = ModelCheckpoint(
        str(weights_path),
        monitor = 'loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min'
    )

    def on_epoch_begin(n_epoch, logs):
        generate_music(model, n_epoch, int_seq, win_size,
                       char2idx, idx2char)
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
        '--win-size', required = True, type = int,
        help = 'Length of training sequence')
    args = parser.parse_args()
    SP.enabled = args.info

    corpus_path = Path(args.corpus_path)
    model_path = Path(args.model_path)

    # Should step be configurable too?
    win_size = args.win_size
    step = 1

    run_training(corpus_path, model_path, win_size, step)

def parse_unparse():
    fname = argv[1]
    mod = load_file(fname)
    rows1 = linearize_rows(mod)
    #print('BEFORE')
    #print(rows_to_string(rows1, numbering = True))
    seq = rows_to_sequence(rows1)
    #print('AFTER')
    rows2 = sequence_to_rows(seq)
    #print(rows_to_string(rows2, numbering = True))
    assert len(rows1) == len(rows2)
    for row1, row2 in zip(rows1, rows2):
        for cell1, cell2 in zip(row1, row2):
            assert cell1.sample_idx == cell2.sample_idx
            note1 = period_to_idx(cell1.period)
            note2 = period_to_idx(cell2.period)
            assert note1 == note2

if __name__ == '__main__':
    main_cmd_line()
