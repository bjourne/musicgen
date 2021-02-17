# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
"""
Model training for music generation
===================================

Usage:
    model-trainer.py [options] <code-type> lstm
        <corpus-path> --emb-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-trainer.py [options]
        <code-type> transformer <corpus-path>
    model-trainer.py [options]
        <code-type> gpt2 <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --lr=<f>               learning rate
    --epochs=<i>           epochs to train for
    --seq-len=<i>          training sequence length
    --batch-size=<i>       batch size
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from docopt import docopt
from musicgen.params import ModelParams
from musicgen.tensorflow import (compiled_model_from_params,
                                 sequence_to_samples)
from musicgen.training_data import (flatten_training_data,
                                    load_training_data)
from musicgen.utils import SP
from pathlib import Path
from tensorflow.keras.callbacks import *

import numpy as np
import tensorflow as tf

def training_data_to_dataset(training_data, seq_len, batch_size):
    seq = flatten_training_data(training_data)
    dataset = sequence_to_samples(seq, seq_len)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    return dataset

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])

    # Hyperparameters
    params = ModelParams.from_docopt_args(args)
    seq_len = params.seq_len
    batch_size = params.batch_size

    train, valid, test = load_training_data(params.code_type, path)
    vocab_size = len(train.encoder.ix2ch)

    args = len(train.arrs), len(valid.arrs), len(test.arrs)
    SP.print('Train/valid/test split %d/%d/%d' % args)

    model = compiled_model_from_params(path, params, vocab_size,
                                       None, False)

    log_path = path / params.log_file()
    def log_losses(epoch, logs):
        with open(log_path, 'at') as outf:
            outf.write('%d %.5f %.5f\n'
                       % (epoch, logs['loss'], logs['val_loss']))
    weights_path = path / params.weights_file()
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
        min_lr = params.lr / 100,
        verbose = 1)
    callbacks = [reduce_lr, cb_best, stopping, cb_epoch_end]

    SP.print('Batching samples...')
    train_ds = training_data_to_dataset(train, seq_len, batch_size)
    valid_ds = training_data_to_dataset(valid, seq_len, batch_size)
    model.fit(x = train_ds,
              validation_data = valid_ds,
              epochs = params.epochs, callbacks = callbacks,
              verbose = 1)

if __name__ == '__main__':
    main()
