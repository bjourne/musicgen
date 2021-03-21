# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
"""
Model training for music generation
===================================

Usage:
    model-trainer.py [options] <root-path> <model>

Options:
    -h --help              show this screen
    -v --verbose           print more output
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from docopt import docopt
from math import ceil, floor
from musicgen.code_generators import (get_code_generator,
                                      log_file, weights_file)
from musicgen.code_utils import INSN_END
from musicgen.tensorflow import load_training_model
from musicgen.training_data import load_training_data
from musicgen.utils import SP, flatten
from pathlib import Path
from random import shuffle
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import *

import numpy as np
import tensorflow as tf

def slide_window(A, win_size, stride, padding = None):
    '''Collects windows that slides over a one-dimensional array.

    If padding is None, the last (rightmost) window is dropped if it
    is incomplete, otherwise it is padded with the padding value.
    '''
    if win_size <= 0:
        raise ValueError('Window size must be positive.')
    if not (0 < stride <= win_size):
        raise ValueError(f'Stride must satisfy 0 < stride <= {win_size}.')
    if not A.base is None:
        raise ValueError('Views cannot be slided over!')

    n_elems = len(A)
    if padding is not None:
        n_windows = ceil((n_elems - win_size) / stride) + 1
        A = np.pad(A, (0, n_windows * win_size - n_elems),
                   constant_values = padding)
    else:
        n_windows = floor((n_elems - win_size) / stride) + 1
    shape = n_windows, win_size

    elem_size = A.strides[-1]
    return np.lib.stride_tricks.as_strided(
        A, shape = shape,
        strides = (elem_size * stride, elem_size),
        writeable = False)

def training_data_to_dataset(td, sl, bs):

    SP.print('Creating samples from %d songs.' % len(td.songs))

    # For some reason NumPy thinks the arrays loaded from the pickle
    # cache are views.
    windows = []
    for _, s in td.songs:
        for ss in s:
            for t in ss:
                # Bug checking
                assert len(t[t < 0]) == 0
                assert len(t[t > 45]) == 0
                assert t.dtype == np.uint16
                t = t.copy()
                wins = slide_window(t, sl + 1, sl, None)
                for win in wins:
                    win = win.copy()
                    assert len(win[win > 45]) > 0
                    windows.append(win)
    shuffle(windows)

    SP.print('Created %d sliding windows.' % len(windows))
    for window in windows:
        assert len(window[window < 0]) == 0
        if len(window[window > 45]) > 0:
            print(window)
        assert len(window[window > 45]) == 0

    # Length must be a multiple of bs
    n_samples = (len(windows) // bs) * bs

    SP.print('Truncating to %d samples.' % n_samples)
    windows = windows[:n_samples]

    xs = np.array([e[:-1] for e in windows])
    ys = np.array([e[1:] for e in windows])

    return xs, ys


def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<model>'])

    # Load training data
    train, valid, test = load_training_data(g['code-type'], root_path)
    vocab_size = len(train.encoder.ix2ch)

    args = len(train.songs), len(valid.songs), len(test.songs)
    train_x, train_y = training_data_to_dataset(
        train, g['sequence-length'], g['batch-size'])
    valid_x, valid_y = training_data_to_dataset(
        valid, g['sequence-length'], g['batch-size'])

    # Load the training model
    model = load_training_model(g, vocab_size)

    weights_dir = root_path / 'weights'
    weights_dir.mkdir(exist_ok = True)

    weights_path = weights_dir / weights_file(g)
    if weights_path.exists():
        SP.print('Loading weights from %s.' % weights_path)
        model.load_weights(str(weights_path))
    else:
        SP.print('Weights file not found.')
    model.reset_states()
    model.summary()

    # Logging
    log_path = weights_dir / log_file(g)
    def log_losses(epoch, logs):
        with open(log_path, 'at') as outf:
            outf.write('%d %.5f %.5f\n'
                       % (epoch, logs['loss'], logs['val_loss']))
    cb_epoch_end = LambdaCallback(on_epoch_end = log_losses)

    cb_best = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_weights_only = True,
        save_best_only = True,
        mode = 'min')
    stopping = EarlyStopping(patience = 30, verbose = 1)
    reduce_lr = ReduceLROnPlateau(
        factor = 0.2, patience = 8,
        min_lr = g['learning-rate'] / 100,
        verbose = 1)
    callbacks = [reduce_lr, cb_best, stopping, cb_epoch_end]

    SP.print('Batching samples...')
    model.fit(x = train_x, y = train_y,
              batch_size = g['batch-size'],
              epochs = 200, callbacks = callbacks,
              verbose = 1,
              validation_data = (valid_x, valid_y))

if __name__ == '__main__':
    main()
