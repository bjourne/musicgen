# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from musicgen.generation import mycode_to_midi_file
from musicgen.keras_utils import OneHotGenerator
from musicgen.mycode import (INSN_PROGRAM,
                             INSN_PLAY,
                             INSN_SAMPLE,
                             INPUT_ARG,
                             corpus_to_mycode)
from musicgen.utils import SP
from random import choice, randrange
import numpy as np

def make_model(seq_len, n_chars):
    model = Sequential()
    model.add(LSTM(128, return_sequences = True,
                   input_shape = (seq_len, n_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'rmsprop')
    print(model.summary())
    return model

def weighted_sample(probs, temperature):
    probs = np.array(probs).astype('float64')
    probs = np.log(probs) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    probas = np.random.multinomial(1, probs, 1)
    return np.argmax(probas)

def generate_seq(model, seed, n_generate, temp, n_chars, skip_chars):
    seq_len = len(seed)
    for _ in range(n_generate):
        X = np.zeros((1, seq_len, n_chars))
        for t, idx in enumerate(seed):
            X[0, t, idx] = 1
        P = model.predict(X, verbose = 0)[0]
        while True:
            if temp is None:
                idx = np.random.choice(len(P), p = P)
            else:
                idx = weighted_sample(P, temp)
            if idx not in skip_chars:
                break
        yield idx
        seed = seed[1:] + [idx]

def generate_mycode(model, seed, n_generate, temp, char2idx, idx2char):
    skip_chars = {(INSN_PROGRAM, 0),
                  (INSN_PLAY, INPUT_ARG),
                  (INSN_SAMPLE, INPUT_ARG)}
    skip_chars = {char2idx[ch] for ch in skip_chars}
    n_chars = len(char2idx)
    seq = generate_seq(model, seed, n_generate, temp, n_chars, skip_chars)
    mycode = [idx2char[i] for i in seq]
    return mycode

def generate_music(model, n_epoch, seq, seq_len,
                   model_path,
                   char2idx, idx2char,
                   programs):
    SP.header('EPOCH', '%d', n_epoch)
    idx = randrange(len(seq) - seq_len)
    seed = list(seq[idx:idx + seq_len])
    for temp in [None, 0.2, 0.5, 1.0, 1.2]:
        fmt = '%s' if temp is None else '%.2f'
        temp_str = fmt % temp
        SP.header('TEMPERATURE %s' % temp_str)
        mycode = generate_mycode(model, seed, 200, temp,
                                 char2idx, idx2char)

        fname = 'gen-%03d-%s.mid' % (n_epoch, temp_str)
        file_path = model_path / fname
        mycode_to_midi_file(mycode, file_path, programs)
        SP.leave()
    SP.leave()

def train_model(corpus_path, win_size, step, batch_size, programs):
    seq = corpus_to_mycode(corpus_path, 150, 24, 20)

    # Different tokens in sequence
    chars = sorted(set(seq))
    vocab_size = len(chars)
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}

    # Cut sequence
    seq = seq[:len(seq) // 10]

    # Convert to integer sequence
    int_seq = np.array([char2idx[c] for c in seq])

    SP.print(f'Training model with %d tokens and %d characters.',
             (len(seq), vocab_size))

    model = make_model(win_size, vocab_size)

    weights_path = corpus_path / 'weights.hdf5'
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
                       corpus_path,
                       char2idx, idx2char,
                       programs)
    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)

    callbacks = [cb_checkpoint, cb_generate]
    model.fit_generator(generator = gen,
                        steps_per_epoch = len(seq) // batch_size,
                        epochs = 10,
                        verbose = 1,
                        shuffle = False,
                        callbacks = callbacks)
