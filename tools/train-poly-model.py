# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Polyphonic LSTM

Usage:
    train-poly-model.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --relative-pitches     use pcode with relative pitches
"""
from docopt import docopt
from musicgen.code_utils import code_to_string
from musicgen.pcode import (EOS_SILENCE, INSN_SILENCE,
                            load_corpus, load_mod_file,
                            pcode_to_midi_file)
from musicgen.utils import (SP, analyze_code,
                            file_name_for_params, find_subseq,
                            split_train_validate_test)
from musicgen.tf_utils import generate_sequences, select_strategy
from pathlib import Path
from random import randrange
from tensorflow.data import Dataset
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
    EPOCHS = 120
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

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    # grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

def create_model(params, vocab_size, batch_size, stateful):
    inp = Input(shape = (params.SEQ_LEN,),
                batch_size = batch_size,
                dtype = tf.int32)
    embedding = Embedding(input_dim = vocab_size,
                          output_dim = params.EMBEDDING_DIM)
    lstm1 = LSTM(params.LSTM1_UNITS,
                  stateful = stateful,
                  return_sequences = True,
                  dropout = params.DROPOUT)
    lstm2 = LSTM(params.LSTM2_UNITS,
                  stateful = stateful,
                  return_sequences = True,
                  dropout = params.DROPOUT)
    time_dist = TimeDistributed(
        Dense(vocab_size, activation = 'softmax'))
    out = time_dist(lstm2(lstm1(embedding(inp))))
    return MyModel(inputs = [inp], outputs = [out])

def automatic_training(model, train, valid, batch_size, epochs, callbacks):
    train = train.batch(batch_size, drop_remainder = True)
    valid = valid.batch(batch_size, drop_remainder = True)
    model.fit(x = train, validation_data = valid,
              epochs = epochs, callbacks = callbacks,
              verbose = 2)

class LossAccObserver:
    def __init__(self):
        self.loss = metrics.SparseCategoricalCrossentropy()
        self.acc = metrics.SparseCategoricalAccuracy()
    def reset(self):
        self.loss.reset_states()
        self.acc.reset_states()
    def update(self, y, y_hat):
        self.loss.update_state(y, y_hat)
        self.acc.update_state(y, y_hat)

def distribute_dataset(strategy, dataset, batch_size):
    def dataset_fn(ctx):
        return dataset.batch(batch_size, drop_remainder = True)
    return strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

@tf.function
def train_epoch(model, strategy, batch_size, ds, obs):
    def step_fn(x, y):
        y_hat = compute_and_apply_gradients(model, x, y)
        obs.update(y, y_hat)
    for x, y in ds:
        strategy.run(step_fn, args = (x, y))

@tf.function
def evaluate_epoch(model, strategy, ds, obs):
    def step_fn(x, y):
        y_hat = model(x, training = False)
        obs.update(y, y_hat)
    for x, y in ds:
        strategy.run(step_fn, args = (x, y))

def manual_training(model, strategy, train, valid,
                    batch_size, epochs, callbacks):
    with strategy.scope():
        train_obs = LossAccObserver()
        valid_obs = LossAccObserver()

    batch_size_per_replica = batch_size // strategy.num_replicas_in_sync
    train = distribute_dataset(strategy, train, batch_size_per_replica)
    valid = distribute_dataset(strategy, valid, batch_size_per_replica)

    fmt = '\-> %3d / %3d - %4db - %3ds - %.4f / %.4f - %.2f / %.2f %s'
    val_losses = []
    last_time = time()
    last_n_steps = 0
    for i in range(epochs):
        start = time()
        train_epoch(model, strategy, batch_size, train, train_obs)
        evaluate_epoch(model, strategy, valid, valid_obs)
        new_time = time()
        val_loss = valid_obs.loss.result()

        new_n_steps = model.optimizer.iterations.numpy()
        time_delta = new_time - last_time
        n_steps_delta = new_n_steps - last_n_steps
        mark = ' '
        if val_loss < min(val_losses, default = 100):
            mark = '*'
        args = (i + 1, epochs, n_steps_delta, time_delta,
                train_obs.loss.result(), val_loss,
                train_obs.acc.result(), valid_obs.acc.result(), mark)
        print(fmt % args)
        last_time = new_time
        last_n_steps = new_n_steps
        val_losses.append(val_loss)
        train_obs.reset()
        valid_obs.reset()

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

def sequence_to_samples(seq, seq_len):
    stride = seq_len - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(seq_len + 1, drop_remainder = True)
    source = tf.constant(seq, dtype = tf.int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(seq_len + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train LSTM 1.0')
    SP.enabled = args['--verbose']
    output_path = Path(args['<corpus-path>'])
    rel_pitches = args['--relative-pitches']

    # Select strategy
    strategy = select_strategy()

    # Load data.
    if output_path.is_dir():
        ix2ch, ch2ix, seq = load_corpus(output_path, 150, rel_pitches)
    else:
        ix2ch, ch2ix, seq = load_mod_file(output_path, rel_pitches)
        output_path = Path('.')
    analyze_code(ix2ch, seq)
    vocab_size = len(ix2ch)

    # Print parameters.
    params = ExperimentParameters(output_path, rel_pitches)
    params.print()

    # Transform data.
    train, valid, test = split_train_validate_test(seq, 0.8, 0.1)
    train = sequence_to_samples(train, params.SEQ_LEN)
    valid = sequence_to_samples(valid, params.SEQ_LEN)

    # Create model and optimizer
    with strategy.scope():
        model = create_model(params, vocab_size, None, False)
        optimizer = RMSprop(learning_rate = params.LEARNING_RATE)
        model.compile(
            optimizer = optimizer,
            loss = 'sparse_categorical_crossentropy',
            metrics = ['sparse_categorical_accuracy'])
        model.summary()

    # Maybe load weights.
    weights_path = params.weights_path()
    if weights_path.exists():
        SP.print('Loading weights from %s.' % weights_path)
        model.load_weights(str(weights_path))
    cb_best = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_weights_only = True,
        save_best_only = True,
        mode = 'min')

    # Run training and prediction.
    batch_size = params.BATCH_SIZE
    epochs = params.EPOCHS
    callbacks = [cb_best]
    manual_training(model, strategy, train, valid,
                    batch_size, epochs, callbacks)
    # automatic_training(model, train, valid,
    #                    batch_size, epochs, callbacks)
    # do_predict(test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5], params)

if __name__ == '__main__':
    main()
