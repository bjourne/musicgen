# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from musicgen.keras_utils import OneHotGenerator
from musicgen.utils import SP
from random import choice
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
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    print(model.summary())
    return model

def weighted_sample(probs, temperature):
    probs = np.array(probs).astype('float64')
    probs = np.log(probs) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    probas = np.random.multinomial(1, probs, 1)
    return np.argmax(probas)

def generate_sequence(model, vocab_size, seed, seq_len, temp, pad_int):
    for _ in range(seq_len):
        X = np.zeros((1, seed.size, vocab_size), dtype = np.int)
        X[0, np.arange(seed.size), seed] = 1
        P = model.predict(X, verbose = 0)[0]
        while True:
            if temp is None:
                idx = np.random.choice(len(P), p = P)
            else:
                idx = weighted_sample(P, temp)
            if idx != pad_int:
                break
        yield idx
        seed = np.roll(seed, -1)
        seed[-1] = idx

def train_model(train, validate,
                weights_path, vocab_size,
                win_size, batch_size, fun):
    n_train = len(train)
    n_validate = len(validate)
    model = make_model(win_size, vocab_size)
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(weights_path)
    else:
        SP.print(f'Weights file {weights_path} not found.')

    train_gen = OneHotGenerator(train, batch_size, win_size, vocab_size)
    validate_gen = OneHotGenerator(validate,
                                   batch_size, win_size, vocab_size)

    cb_checkpoint = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min'
    )
    def on_epoch_begin(epoch, logs):
        fun(model, epoch)
    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)
    model.fit(x = train_gen,
              steps_per_epoch = n_train // batch_size,
              validation_data = validate_gen,
              validation_steps = n_validate // batch_size,
              verbose = 1,
              shuffle = True,
              epochs = 10,
              callbacks = [cb_checkpoint, cb_generate])
