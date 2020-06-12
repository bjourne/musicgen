# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Polyphonic LSTM

Usage:
    tpu-training.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --relative-pitches     use pcode with relative pitches
"""
from docopt import docopt
from musicgen.code_utils import code_to_string
from musicgen.pcode import (EOS_SILENCE,
                            INSN_SILENCE,
                            load_corpus, load_mod_file,
                            pcode_to_midi_file)
from musicgen.utils import (SP, analyze_code,
                            file_name_for_params, find_subseq,
                            split_train_validate_test)
from musicgen.tf_utils import (generate_sequences,
                               initialize_tpus,
                               sequence_to_batched_dataset)
from pathlib import Path
from random import randrange, shuffle
from tensorflow.data import Dataset
from tensorflow.nn import compute_average_loss
from tensorflow.keras import *
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from time import time
import numpy as np
import tensorflow as tf

# Hyperparameters not from the command line here.
class ExperimentParameters:
    BATCH_SIZE = 32
    EPOCHS = 108
    LEARNING_RATE = 0.005
    EMBEDDING_DIM = 100
    LSTM1_UNITS = 512
    LSTM2_UNITS = 512
    DROPOUT = 0.5
    SEQ_LEN = 256

    def __init__(self, output_path, relative_pitches):
        self.output_path = output_path
        self.relative_pitches = relative_pitches
        self.prefix = 'pcode'

    def weights_path(self):
        fmt = '%s_weights-%03d-%03d-%.5f-%02d-%04d-%04d-%.2f-%s.h5'
        args = (self.prefix,
                self.BATCH_SIZE,
                self.EPOCHS,
                self.LEARNING_RATE,
                self.EMBEDDING_DIM,
                self.LSTM1_UNITS,
                self.LSTM2_UNITS,
                self.DROPOUT,
                self.relative_pitches)
        file_name = fmt % args
        return self.output_path / file_name

    def print(self):
        SP.header('EXPERIMENT PARAMETERS')
        params = [
            ('Batch size', self.BATCH_SIZE),
            ('Epochs', self.EPOCHS),
            ('Learning rate', self.LEARNING_RATE),
            ('Embedding dimension', self.EMBEDDING_DIM),
            ('LSTM1 units', self.LSTM1_UNITS),
            ('LSTM2 units', self.LSTM2_UNITS),
            ('Dropout (both layers)', self.DROPOUT),
            ('Sequence length', self.SEQ_LEN),
            ('Relative pitches', self.relative_pitches),
            ('Output path', self.output_path)]
        for param, value in params:
            SP.print('%-22s: %5s' % (param, value))
        SP.leave()

def create_model(params, vocab_size, batch_size, stateful):
    return Sequential([
        Embedding(
            input_dim = vocab_size,
            output_dim = params.EMBEDDING_DIM,
            batch_input_shape = [batch_size, None]),
        Dropout(0.1),
        LSTM(
            params.LSTM1_UNITS,
            stateful = stateful,
            return_sequences = True,
            dropout = params.DROPOUT),
        LSTM(
            params.LSTM2_UNITS,
            stateful = stateful,
            return_sequences = True,
            dropout = params.DROPOUT),
        TimeDistributed(
            Dense(vocab_size, activation = 'softmax'))
    ])

def create_training_model(params, vocab_size):
    model = create_model(params, vocab_size, None, False)
    #opt = SGD(learning_rate = 0.01)
    opt = RMSprop(learning_rate = params.LEARNING_RATE)
    model.compile(
        optimizer = opt,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['sparse_categorical_accuracy'])
    return model

def do_train(train, validate, vocab_size, params):
    # Must be done before creating the datasets.
    strategy = initialize_tpus()

    # Reshape the raw data into tensorflow Datasets
    ds_train = sequence_to_batched_dataset(train,
                                           params.SEQ_LEN,
                                           params.BATCH_SIZE)
    ds_validate = sequence_to_batched_dataset(validate,
                                              params.SEQ_LEN,
                                              params.BATCH_SIZE)

    if strategy:
        with strategy.scope():
            model = create_training_model(params, vocab_size)
    else:
        model = create_training_model(params, vocab_size)
    model.summary()

    weights_path = params.weights_path()
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(str(weights_path))

    cb_best = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min')
    SP.print('Fitting for %d epochs.' % params.EPOCHS)
    history = model.fit(x = ds_train,
                        validation_data = ds_validate,
                        epochs = params.EPOCHS,
                        callbacks = [cb_best],
                        verbose = 2)
    print(history)

# def get_dataset(raw_dataset, batch_size, seq_len):
#     stride = seq_len - 1
#     def split_input_target(chunk):
#         return chunk[:-1], chunk[1:]
#     def flatten_window(win):
#         return win.batch(seq_len + 1, drop_remainder = True)
#     source = tf.constant(raw_dataset, dtype = tf.int32)
#     return Dataset \
#         .from_tensor_slices(source) \
#         .window(seq_len + 1, stride, drop_remainder = True) \
#         .flat_map(flatten_window) \
#         .map(split_input_target) \
#         .shuffle(10000) \
#         .batch(batch_size, drop_remainder = True)

# Very strange code!
def distribute_dataset(strategy, ds, batch_size_per_replica, seq_len):
    def dataset_fn(ctx):
        return sequence_to_batched_dataset(
            ds, seq_len, batch_size_per_replica)
    return strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

@tf.function
def train_epoch(model, opt, strategy, batch_size, ds, loss_var, acc_var):
    def step_fn(X, Y):
        with tf.GradientTape() as tape:
            Y_hat = model(X, training = True)
            loss = losses.sparse_categorical_crossentropy(Y, Y_hat)
            loss = tf.reduce_sum(loss) / batch_size
        grads = tape.gradient(loss, model.trainable_variables)
        gvs = list(zip(grads, model.trainable_variables))
        capped_gvs = [(tf.clip_by_norm(g, 0.15), v) for (g, v) in gvs]
        opt.apply_gradients(capped_gvs)
        loss_var.update_state(Y, Y_hat)
        acc_var.update_state(Y, Y_hat)

    for X, Y in ds:
        strategy.run(step_fn, args = (X, Y))

@tf.function
def evaluate_epoch(model, strategy, batch_size, ds, loss_var, acc_var):
    def step_fn(X, Y):
        Y_hat = model(X, training = False)
        loss = losses.sparse_categorical_crossentropy(Y, Y_hat)
        loss = tf.reduce_sum(loss) / batch_size
        loss_var.update_state(Y, Y_hat)
        acc_var.update_state(Y, Y_hat)

    for X, Y in ds:
        strategy.run(step_fn, args = (X, Y))

def do_train3(train, valid, vocab_size, params):
    strategy = initialize_tpus()
    with strategy.scope():
        model = create_model(params, vocab_size, None, False)
        model.summary()
        opt = RMSprop(learning_rate = params.LEARNING_RATE)
        train_loss = metrics.SparseCategoricalCrossentropy()
        train_acc = metrics.SparseCategoricalAccuracy()
        valid_loss = metrics.SparseCategoricalCrossentropy()
        valid_acc = metrics.SparseCategoricalAccuracy()
    observers = (train_loss, train_acc, valid_loss, valid_acc)

    batch_size = params.BATCH_SIZE
    seq_len = params.SEQ_LEN
    batch_size_per_replica = batch_size // strategy.num_replicas_in_sync

    train_ds = distribute_dataset(strategy, train,
                                  batch_size, seq_len)
    valid_ds = distribute_dataset(strategy, valid,
                                  batch_size, seq_len)

    fmt = '\-> %3d / %3d - %4db - %3ds - %.4f / %.4f - %.2f / %.2f'
    last_time = time()
    last_n_steps = 0
    n_epochs = params.EPOCHS
    for i in range(n_epochs):
        start = time()
        train_epoch(model, opt, strategy, batch_size,
                    train_ds, train_loss, train_acc)
        evaluate_epoch(model, strategy, batch_size,
                       valid_ds, valid_loss, valid_acc)

        new_time = time()
        new_n_steps = opt.iterations.numpy()
        time_delta = new_time - last_time
        n_steps_delta = new_n_steps - last_n_steps
        results = [obs.result() for obs in observers]
        args = (i + 1, n_epochs, n_steps_delta, time_delta,
                results[0], results[2], results[1], results[3])
        SP.print(fmt % args)
        last_time = new_time
        last_n_steps = new_n_steps
        for obs in observers:
            obs.reset_states()

def do_predict(seq, ix2ch, ch2ix, temps, params):
    batch_size = len(temps)
    SP.header('%d PREDICTIONS' % batch_size)
    model = create_model(params, len(ix2ch), batch_size, True)
    model.load_weights(str(params.weights_path()))
    model.reset_states()

    seed_len = 128

    while True:
        idx = randrange(len(seq) - seed_len)
        seed = seq[idx:idx + seed_len]
        if not list(find_subseq(seed.tolist(), EOS_SILENCE)):
            break
        SP.print('EOS_SILENCE in seed, regenerating.')
    seed_string = code_to_string(ix2ch[ix] for ix in seed)
    SP.print('Seed %s.' % seed_string)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, temps, seed, 500, [])
    # Two bars of silence
    join = np.array([ch2ix[(INSN_SILENCE, 16)]] * 2)
    join = np.repeat(np.expand_dims(join, 0), batch_size, axis = 0)
    seqs = np.hstack((seed, join, seqs))

    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    file_name_fmt = '%s_output-%s-%.2f.mid'
    for temp, seq in zip(temps, seqs):
        args = (params.prefix, params.relative_pitches, temp)
        file_name = file_name_fmt % args
        file_path = params.output_path / file_name
        pcode_to_midi_file(seq, file_path, params.relative_pitches)
    SP.leave()

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train LSTM 1.0')
    SP.enabled = args['--verbose']
    output_path = Path(args['<corpus-path>'])
    relative_pitches = args['--relative-pitches']

    # Load sequence
    if output_path.is_dir():
        ix2ch, ch2ix, seq = load_corpus(output_path, 150,
                                        relative_pitches)
    else:
        ix2ch, ch2ix, seq = load_mod_file(output_path, relative_pitches)
        output_path = Path('.')
    analyze_code(ix2ch, seq)
    params = ExperimentParameters(output_path, relative_pitches)
    params.print()
    vocab_size = len(ix2ch)

    # Split data
    train, validate, test = split_train_validate_test(seq, 0.8, 0.1)

    # Run training and prediction.
    do_train3(train, validate, vocab_size, params)
    # do_train(train, validate, vocab_size, params)
    # do_predict(test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5], params)

if __name__ == '__main__':
    main()
