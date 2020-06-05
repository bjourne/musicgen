# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Monophonic model

Usage:
    train-mono-model.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --win-size=<int>       window size [default: 64]
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
                            load_pickle_cache)
from musicgen.tf_utils import initialize_tpus, sequence_to_batched_dataset
from pathlib import Path
from random import randrange, shuffle
from tensorflow.keras import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np

BATCH_SIZE = 32
SEQ_LEN = 64
LEARNING_RATE = 0.001
EPOCHS = 10

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

def lstm_model(vocab_size, batch_size, stateful):
    return Sequential([
        Embedding(
            input_dim = vocab_size,
            output_dim = 128,
            batch_input_shape = [batch_size, None]),
        LSTM(
            128,
            stateful = stateful,
            return_sequences = True,
            dropout = 0.2),
        LSTM(
            128,
            stateful = stateful,
            return_sequences = True,
            dropout = 0.2),
        TimeDistributed(
            Dense(vocab_size, activation = 'softmax'))
    ])

def create_training_model(vocab_size):
    model = lstm_model(vocab_size, None, False)
    opt1 = RMSprop(learning_rate = LEARNING_RATE)
    model.compile(
        optimizer = opt1,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['sparse_categorical_accuracy'])
    return model

def do_train(output_path, train, validate, vocab_size):
    # Must be done before creating the datasets.
    strategy = initialize_tpus()

    # Reshape the raw data into tensorflow Datasets
    ds_train = sequence_to_batched_dataset(train, SEQ_LEN, BATCH_SIZE)
    ds_validate = sequence_to_batched_dataset(validate,
                                              SEQ_LEN, BATCH_SIZE)

    if strategy:
        with strategy.scope():
            model = create_training_model(vocab_size)
    else:
        model = create_training_model(vocab_size)
    model.summary()

    weights_path = output_path / 'mono_weights.h5'
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
              epochs = EPOCHS,
              callbacks = [cb_best],
              verbose = 1)

def generate_sequences(model, temperatures, seed, length):
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

    SP.leave()
    return [[int(preds[j][i]) for j in range(length)]
            for i in range(batch_size)]

def do_predict(output_path, seq, ix2ch, ch2ix, temperatures):
    batch_size = len(temperatures)
    SP.header('%d PREDICTIONS' % batch_size)
    model = lstm_model(len(ix2ch), batch_size, True)

    weights_path = output_path / 'mono_weights.h5'
    model.load_weights(str(weights_path))
    model.reset_states()

    long_jump = ch2ix[(INSN_JUMP, 64)]
    while True:
        idx = randrange(len(seq) - SEQ_LEN)
        seed = seq[idx:idx + SEQ_LEN]
        if not long_jump in seed:
            break
        SP.print('Long jump in seed - skipping.')
    seed_string = mcode_to_string(ix2ch[ix] for ix in seed)
    SP.print('Seed %s.' % seed_string)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, temperatures, seed, 500)

    seqs = np.hstack((seed, seqs))
    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    file_name_fmt = 'mono-%02d.mid'
    for i, seq in enumerate(seqs):
        file_name = file_name_fmt % i
        file_path = output_path / file_name
        mcode_to_midi_file(seq, file_path, 120, None)
    SP.leave()


def main():
    args = docopt(__doc__, version = 'Monophonic model 1.0')
    SP.enabled = args['--verbose']

    output_path = Path(args['<corpus-path>'])
    win_size = int(args['--win-size'])
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
    # do_predict(output_path, test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5])
    do_train(output_path, train, validate, vocab_size)
    do_predict(output_path, test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5])

if __name__ == '__main__':
    main()
