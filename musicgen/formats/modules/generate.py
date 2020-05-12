# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from collections import namedtuple
from construct import Container
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from musicgen.formats.modules.mycode import (get_sequence,
                                             prettyprint_mycode)
from musicgen.utils import OneHotGenerator, SP
from os import environ
from pathlib import Path
from pickle import dump, load
from random import choice, randint, randrange
from sys import argv
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_model(seq_len, n_chars):
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_len, n_chars)))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam')
    print(model.summary())
    return model

def weighted_sample(probs, temperature):
    probs = np.array(probs).astype('float64')
    probs = np.log(probs) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    probas = np.random.multinomial(1, probs, 1)
    return np.argmax(probas)

def generate_chars(model, seed, n_generate, temp, char2idx, idx2char):
    seq_len = len(seed)
    n_chars = len(char2idx)
    for _ in range(n_generate):
        X = np.zeros((1, seq_len, n_chars))
        for t, idx in enumerate(seed):
            X[0, t, idx] = 1
        P = model.predict(X, verbose = 0)[0]
        if temp is None:
            idx = np.random.choice(len(P), p = P)
        else:
            idx = weighted_sample(P, temp)
        yield idx
        seed = seed[1:] + [idx]

def generate_music(model, n_epoch, seq, seq_len,
                   char2idx, idx2char):
    SP.header('EPOCH', '%d', n_epoch)
    idx = randrange(len(seq) - seq_len)
    seed = list(seq[idx:idx + seq_len])
    for temp in [None, 0.2, 0.5, 1.0, 1.2]:
        fmt = '%s' if temp is None else '%.2f'
        SP.header('TEMPERATURE', fmt, temp)
        mycode = [idx2char[i]
                  for i in generate_chars(model, seed,
                                          100, temp,
                                          char2idx, idx2char)]
        prettyprint_mycode(mycode)
        # rows = sequence_to_rows(generated)
        # for row in rows:
        #     SP.print(row_to_string(row))
        SP.leave()
    SP.leave()

def run_training(corpus_path, model_path, win_size, step, batch_size):
    seq = get_sequence(corpus_path, model_path)


    # Different tokens in sequence
    chars = sorted(set(seq))
    vocab_size = len(chars)
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}

    # Cut sequence
    seq = seq[:len(seq) // 100]

    # Convert to integer sequence
    int_seq = np.array([char2idx[c] for c in seq])

    SP.print(f'Training model with %d tokens and %d characters.',
             (len(seq), vocab_size))

    model = make_model(win_size, vocab_size)

    weights_path = model_path / 'weights.hdf5'
    if weights_path.exists():
        SP.print(f'Loading existing weights from "{weights_path}".')
        model.load_weights(weights_path)

    gen = OneHotGenerator(int_seq, batch_size, win_size, vocab_size)

    cb_checkpoint = ModelCheckpoint(
        str(weights_path),
        monitor = 'loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min'
    )

    def on_epoch_begin(n_epoch, logs):
        generate_music(model, n_epoch, int_seq, win_size,
                       char2idx, idx2char)
    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)

    callbacks = [cb_checkpoint, cb_generate]
    model.fit_generator(generator = gen,
                        steps_per_epoch = len(seq) // batch_size,
                        epochs = 10,
                        verbose = 1,
                        shuffle = False,
                        callbacks = callbacks)

def main_cmd_line():
    parser = ArgumentParser(description = 'Music ML')
    parser.add_argument(
        '--corpus-path', required = True,
        help = 'Path to corpus')
    parser.add_argument(
        '--model-path', required = True,
        help = 'Path to store model and cache.')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'Print information')
    parser.add_argument(
        '--win-size', required = True, type = int,
        help = 'Length of training sequence')
    args = parser.parse_args()
    SP.enabled = args.info

    corpus_path = Path(args.corpus_path)
    model_path = Path(args.model_path)

    # Should step be configurable too?
    win_size = args.win_size
    step = 1

    run_training(corpus_path, model_path, win_size, step, 128)

if __name__ == '__main__':
    main_cmd_line()
