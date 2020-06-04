# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Polyphonic LSTM using TPU

Usage:
    tpu-training.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --relative-pitches     use pcode with relative pitches
"""
from collections import Counter
from docopt import docopt
from logging import ERROR
from musicgen.pcode import (EOS_SILENCE, INSN_SILENCE,
                            analyze_pcode,
                            load_corpus, load_mod_file,
                            pcode_to_midi_file,
                            pcode_to_string)
from musicgen.utils import SP, file_name_for_params, find_subseq
from os import environ, listdir
from pathlib import Path
from random import randrange, shuffle
from tensorflow import constant, get_logger, int32
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices)
from tensorflow.data import Dataset
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np


# Hyperparameters not from the command line here.
class ExperimentParameters:
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.005
    EMBEDDING_DIM = 32
    LSTM1_UNITS = 128
    LSTM2_UNITS = 128
    DROPOUT = 0.1
    SEQ_LEN = 128

    def __init__(self, output_path, relative_pitches):
        self.output_path = output_path
        self.relative_pitches = relative_pitches

    def weights_path(self):
        fmt = 'weights-%03d-%03d-%.5f-%02d-%03d-%03d-%.2f-%s.h5'
        args = (self.BATCH_SIZE,
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

def lstm_model(params, seq_len, vocab_size, batch_size, stateful):
    return Sequential([
        Embedding(
            input_dim = vocab_size,
            output_dim = params.EMBEDDING_DIM,
            batch_input_shape = [batch_size, None]),
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

def initialize_tpus():

    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        SP.print('TPU not configured.')
        return None
    SP.print('Connecting to TPU at %s.' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    devs = list_logical_devices('TPU')
    assert len(devs) > 0
    SP.header('%d TPU DEVICES' % len(devs))
    for dev in devs:
        SP.print(dev)
    SP.leave()
    strategy = TPUStrategy(resolver)
    return strategy

def create_dataset(seq, params):
    # Make this parameter configurable.
    stride = params.SEQ_LEN - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(params.SEQ_LEN + 1, drop_remainder = True)
    SP.print('Length %d, seq_len %d, batch_size %d.'
             % (len(seq), params.SEQ_LEN, params.BATCH_SIZE))
    source = constant(seq, dtype = int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(params.SEQ_LEN + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000) \
        .batch(params.BATCH_SIZE, drop_remainder = True)

def create_training_model(params, vocab_size):
    model = lstm_model(params, params.SEQ_LEN, vocab_size, None, False)
    opt1 = RMSprop(learning_rate = params.LEARNING_RATE)
    model.compile(
        optimizer = opt1,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['sparse_categorical_accuracy'])
    return model

def do_train(train, validate, vocab_size, params):
    # Must be done before creating the datasets.
    strategy = initialize_tpus()

    # Reshape the raw data into tensorflow Datasets
    ds_train = create_dataset(train, params)
    ds_validate = create_dataset(validate, params)

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
                        verbose = 1)
    print(history)

def generate_sequences(model, dataset, temperatures, seed, length):
    SP.header('TEMPERATURES %s' % temperatures)
    batch_size = len(temperatures)

    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])
    SP.print('Consumed seed %s.' % (seed.shape,))

    preds = [seed[:, -1:]]
    for _ in range(length):
        last_word = preds[-1]
        P = model.predict(last_word)[:, 0, :]

        next_idx = [np.random.choice(P.shape[1], p = P[i])
                    for i in range(batch_size)]
        preds.append(np.asarray(next_idx, dtype = np.int32))

    seqs = []
    for i in range(batch_size):
        seq = [int(preds[j][i]) for j in range(length)]
        seqs.append(seq)
    SP.leave()
    return seqs

def do_predict(seq, ix2ch, ch2ix, temperatures, params):
    batch_size = len(temperatures)
    SP.header('%d PREDICTIONS' % batch_size)
    model = lstm_model(params, 1, len(ix2ch), batch_size, True)
    model.load_weights(str(params.weights_path()))
    model.reset_states()

    while True:
        idx = randrange(len(seq) - params.SEQ_LEN)
        seed = seq[idx:idx + params.SEQ_LEN]
        if not list(find_subseq(seed.tolist(), EOS_SILENCE)):
            break
        SP.print('EOS_SILENCE in seed, regenerating.')
    seed_string = pcode_to_string(ix2ch[ix] for ix in seed)
    SP.print('Seed %s.' % seed_string)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, seq, temperatures, seed, 500)
    # Two bars of silence
    join = np.array([ch2ix[(INSN_SILENCE, 16)]] * 2)
    join = np.repeat(np.expand_dims(join, 0), batch_size, axis = 0)
    seqs = np.hstack((seed, join, seqs))

    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    file_name_fmt = 'tpu-test-%s-%02d.mid'
    for i, seq in enumerate(seqs):
        args = params.relative_pitches, i
        file_name = file_name_fmt % args
        file_path = params.output_path / file_name
        pcode_to_midi_file(seq, file_path, params.relative_pitches)
    SP.leave()

def main():
    args = docopt(__doc__, version = 'Train LSTM Using TPU 1.0')
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
    analyze_pcode(ix2ch, seq)

    params = ExperimentParameters(output_path, relative_pitches)
    params.print()

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

    # Run training and prediction.
    do_train(train, validate, vocab_size, params)
    do_predict(test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5], params)

if __name__ == '__main__':
    main()
