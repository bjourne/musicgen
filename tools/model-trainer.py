# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""
Model training for music generation
===================================

Usage:
    model-trainer.py [options] <code-type> lstm
        <corpus-path> --emb-size=<i> --batch-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-trainer.py [options]
        <code-type> transformer <corpus-path>
        --dropout=<f> --batch-size=<i>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --lr=<f>               learning rate
    --epochs=<i>           epochs to train for
    --seq-len=<i>          training sequence length
    --seed-idx=<i>         seed index [default: random]
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from docopt import docopt
from musicgen.params import ModelParams
from musicgen.tensorflow import select_strategy
from musicgen.training_data import load_training_data
from musicgen.utils import SP
from pathlib import Path
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *

import numpy as np
import tensorflow as tf

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])
    seed_idx = args['--seed-idx']

    # Hyperparameters
    params = ModelParams.from_docopt_args(args)
    train, valid, test = load_training_data(params.code_type, path)

    args = len(train.arrs), len(valid.arrs), len(test.arrs)
    SP.print('Train/valid/test split %d/%d/%d' % args)

    vocab_size = len(train.encoder.ix2ch)
    with select_strategy().scope():
        model = params.model(vocab_size, None, False)
        optimizer = RMSprop(learning_rate = params.lr)
        loss_fn = SparseCategoricalCrossentropy(from_logits = True)
        model.compile(
            optimizer = optimizer,
            loss = loss_fn,
            metrics = ['sparse_categorical_accuracy'])
    model.summary()

    weights_path = path / params.weights_file()
    if weights_path.exists():
        SP.print('Loading weights from %s...' % weights_path)
        model.load_weights(str(weights_path))
    else:
        SP.print('Weights file %s not found.' % weights_path)

    cb_best = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_weights_only = True,
        save_best_only = True,
        mode = 'min')
    reduce_lr = ReduceLROnPlateau(
        factor = 0.2, patience = 8, min_lr = params.lr / 100)
    stopping = EarlyStopping(patience = 30)
    callbacks = [cb_best, reduce_lr, stopping]
    SP.print('Batching samples...')
    train_ds = train.to_samples(params.seq_len) \
        .batch(params.batch_size, drop_remainder = True)
    valid_ds = valid.to_samples(params.seq_len) \
        .batch(params.batch_size, drop_remainder = True)

    model.fit(x = train_ds,
              validation_data = valid_ds,
              epochs = params.epochs, callbacks = callbacks,
              verbose = 1)

if __name__ == '__main__':
    main()
