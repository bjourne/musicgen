# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Monophonic model (scode)

Usage:
    train-mono-model2.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --kb-limit=<int>       kb limit [default: 150]
    --relative-pitches     use scode with relative pitches
"""
from docopt import docopt
from musicgen.pcode import pcode_to_string
from musicgen.scode import (INSN_SILENCE, load_corpus,
                            load_mod_file,
                            scode_to_midi_file)
from musicgen.utils import (SP, analyze_code,
                            split_train_validate_test)
from musicgen.tf_utils import (generate_sequences,
                               initialize_tpus,
                               sequence_to_batched_dataset)
from pathlib import Path
from random import randrange
from tensorflow.keras import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np

# Hyperparameters not from the command line here.
class ExperimentParameters:
    BATCH_SIZE = 2
    EPOCHS = 10
    LEARNING_RATE = 0.005
    EMBEDDING_DIM = 32
    LSTM1_UNITS = 128
    LSTM2_UNITS = 128
    DROPOUT = 0.2
    SEQ_LEN = 128

    def __init__(self, output_path, rel_pitches):
        self.output_path = output_path
        self.rel_pitches = rel_pitches
        self.prefix = 'scode'

    def weights_path(self):
        fmt = '%s_weights-%03d-%03d-%.5f-%02d-%03d-%03d-%.2f-%s.h5'
        args = (self.prefix,
                self.BATCH_SIZE,
                self.EPOCHS,
                self.LEARNING_RATE,
                self.EMBEDDING_DIM,
                self.LSTM1_UNITS,
                self.LSTM2_UNITS,
                self.DROPOUT,
                self.rel_pitches)
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
            ('Relative pitches', self.rel_pitches),
            ('Output path', self.output_path)]
        for param, value in params:
            SP.print('%-22s: %5s' % (param, value))
        SP.leave()

def lstm_model(params, vocab_size, batch_size, stateful):
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
            Dense(vocab_size, activation = 'softmax'))])

def create_training_model(params, vocab_size):
    model = lstm_model(params, vocab_size, None, False)
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
    model.fit(x = ds_train,
              validation_data = ds_validate,
              epochs = params.EPOCHS,
              callbacks = [cb_best],
              verbose = 1)

def do_predict(seq, ix2ch, ch2ix, temps, params):
    batch_size = len(temps)
    SP.header('%d PREDICTIONS' % batch_size)
    model = lstm_model(params, len(ix2ch), batch_size, True)

    model.load_weights(str(params.weights_path()))
    model.reset_states()

    # Not sure if we should avoid silence tokens or not.
    idx = randrange(len(seq) - params.SEQ_LEN)
    seed = seq[idx:idx + params.SEQ_LEN]

    seed_string = pcode_to_string(ix2ch[ix] for ix in seed)
    SP.print('Seed: %s.' % seed_string)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, temps, seed, 600, [])

    seqs = np.hstack((seed, seqs))
    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    file_name_fmt = '%s_output-%.2f.mid'
    for temp, seq in zip(temps, seqs):
        file_name = file_name_fmt % (params.prefix, temp)
        file_path = params.output_path / file_name
        SP.print('%.2f -> %s.' % (temp, pcode_to_string(seq)))
        scode_to_midi_file([seq], file_path, params.rel_pitches)
    SP.leave()

def main():
    args = docopt(__doc__, version = 'Monophonic model 2.0')
    SP.enabled = args['--verbose']
    output_path = Path(args['<corpus-path>'])
    kb_limit = int(args['--kb-limit'])
    rel_pitches = args['--relative-pitches']

    if output_path.is_dir():
        ix2ch, ch2ix, seq = load_corpus(output_path, kb_limit,
                                        rel_pitches, True)
    else:
        ix2ch, ch2ix, seq = load_mod_file(output_path, rel_pitches, True)
        output_path = Path('.')
    analyze_code(ix2ch, seq)

    # Setup and print parameters
    params = ExperimentParameters(output_path, rel_pitches)
    params.print()

    # Split data
    train, validate, test = split_train_validate_test(seq, 0.8, 0.1)

    do_train(train, validate, len(ix2ch), params)
    temps = [0.5, 0.8, 1.0, 1.2, 1.5]
    do_predict(test, ix2ch, ch2ix, temps, params)

if __name__ == '__main__':
    main()
