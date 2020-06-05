# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Monophonic model

Usage:
    train-mono-model.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --kb-limit=<int>       kb limit [default: 150]
    --pack-mcode           use packed mcode
    --fraction=<float>     fraction of corpus to use [default: 1.0]
"""
from docopt import docopt
from musicgen.mcode import (INSN_JUMP,
                            load_corpus,
                            load_mod_file,
                            mcode_to_midi_file,
                            mcode_to_string)
from musicgen.utils import (SP,
                            analyze_code,
                            encode_training_sequence,
                            file_name_for_params, flatten,
                            load_pickle_cache,
                            split_train_validate_test)
from musicgen.tf_utils import (generate_sequences,
                               initialize_tpus,
                               sequence_to_batched_dataset)
from pathlib import Path
from random import randrange, shuffle
from tensorflow.keras import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np

class ExperimentParameters:
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128
    LSTM1_UNITS = 256
    LSTM2_UNITS = 256
    DROPOUT = 0.2
    SEQ_LEN = 128

    def __init__(self, output_path, pack_mcode):
        self.output_path = output_path
        self.pack_mcode = pack_mcode

    def weights_path(self):
        fmt = 'mono_weights-%03d-%03d-%.5f-%02d-%03d-%03d-%.2f-%s.h5'
        args = (self.BATCH_SIZE,
                self.EPOCHS,
                self.LEARNING_RATE,
                self.EMBEDDING_DIM,
                self.LSTM1_UNITS,
                self.LSTM2_UNITS,
                self.DROPOUT,
                self.pack_mcode)
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
            ('Pack mcode', self.pack_mcode),
            ('Output path', self.output_path)]
        for param, value in params:
            SP.print('%-22s: %5s' % (param, value))
        SP.leave()

def flatten_corpus(corpus_path, kb_limit, pack_mcode, fraction):
    mcode_mods = load_corpus(corpus_path, kb_limit, pack_mcode)
    n_mods = len(mcode_mods)
    params = (n_mods, kb_limit, pack_mcode, fraction)
    cache_file = file_name_for_params('cached_mcode_flat',
                                      'pickle', params)
    cache_path = corpus_path / cache_file

    def rebuild_fun():
        seqs = [[c[1] for c in mcode_mod.cols]
                for mcode_mod in mcode_mods]
        seqs = flatten(seqs)
        seqs = seqs[:int(len(seqs) * fraction)]
        shuffle(seqs)
        for seq in seqs:
            seq.append((INSN_JUMP, 64))
        return encode_training_sequence(flatten(seqs))
    return load_pickle_cache(cache_path, rebuild_fun)

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

    long_jump_toks = [(INSN_JUMP, 16), (INSN_JUMP, 32), (INSN_JUMP, 64)]
    long_jump_ints = [ch2ix[insn] for insn in long_jump_toks
                      if insn in ch2ix]
    while True:
        idx = randrange(len(seq) - params.SEQ_LEN)
        seed = seq[idx:idx + params.SEQ_LEN]
        if not set(seed) & set(long_jump_ints):
            break
        SP.print('Long jump in seed - skipping.')
    seed_string = mcode_to_string(ix2ch[ix] for ix in seed)
    SP.print('Seed %s.' % seed_string)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, temps, seed, 600, long_jump_ints)

    seqs = np.hstack((seed, seqs))
    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    file_name_fmt = 'mono-%.2f.mid'
    for temp, seq in zip(temps, seqs):
        file_name = file_name_fmt % temp
        file_path = params.output_path / file_name
        mcode_to_midi_file(seq, file_path, 90, None)
    SP.leave()

def main():
    args = docopt(__doc__, version = 'Monophonic model 1.0')
    SP.enabled = args['--verbose']

    output_path = Path(args['<corpus-path>'])
    kb_limit = int(args['--kb-limit'])
    pack_mcode = args['--pack-mcode']
    fraction = float(args['--fraction'])

    if output_path.is_dir():
        ix2ch, ch2ix, seq = flatten_corpus(output_path, kb_limit,
                                           pack_mcode, fraction)
    else:
        ix2ch, ch2ix, seq = load_mod_file(output_path, pack_mcode)
        output_path = Path('.')
    analyze_code(ix2ch, seq)

    # Setup and print parameters
    params = ExperimentParameters(output_path, pack_mcode)
    params.print()

    vocab_size = len(ix2ch)

    # Split data
    train, validate, test = split_train_validate_test(seq, 0.8, 0.1)

    # Run training and prediction.
    temps = [0.5, 0.8, 1.0, 1.2, 1.5]
    #do_train(train, validate, vocab_size, params)
    do_predict(test, ix2ch, ch2ix, temps, params)

if __name__ == '__main__':
    main()
