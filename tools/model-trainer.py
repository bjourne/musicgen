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
from musicgen.code_generators import get_code_generator
from musicgen.tensorflow import (load_training_model,
                                 log_file, weights_file,
                                 sequence_to_samples)
from musicgen.training_data import load_training_data
from musicgen.utils import SP
from pathlib import Path
from tensorflow.keras.callbacks import *

def training_data_to_dataset(td, seq_len, batch_size):
    ds = sequence_to_samples(td.data, seq_len)
    return ds.batch(batch_size, drop_remainder = True)

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

    args = len(train.meta), len(valid.meta), len(test.meta)
    SP.print('Train/valid/test split %d/%d/%d' % args)

    # Load the training model
    model = load_training_model(g, root_path, vocab_size)

    # Logging
    weights_dir = root_path / 'weights'
    weights_dir.mkdir(exist_ok = True)

    # Logging
    log_path = weights_dir / log_file(g)
    def log_losses(epoch, logs):
        with open(log_path, 'at') as outf:
            outf.write('%d %.5f %.5f\n'
                       % (epoch, logs['loss'], logs['val_loss']))
    cb_epoch_end = LambdaCallback(on_epoch_end = log_losses)

    weights_path = weights_dir / weights_file(g)
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
    train_ds = training_data_to_dataset(
        train, g['sequence-length'], g['batch-size'])
    valid_ds = training_data_to_dataset(
        valid, g['sequence-length'], g['batch-size'])
    model.fit(x = train_ds,
              validation_data = valid_ds,
              epochs = 200, callbacks = callbacks,
              verbose = 1)

if __name__ == '__main__':
    main()
