# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence
from musicgen.keras_utils import OneHotGenerator
from musicgen.utils import SP
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

def generate_sequence(model, S, seq_len, temp):
    X = np.expand_dims(S, axis = 0)
    seq = []
    log_lh = 0.0
    for _ in range(seq_len):
        P = model.predict(X, verbose = 0)[0]

        # Extra precision needed to ensure np.sum(P) == 1.0.
        P = P.astype(np.float64)

        # Reweigh probabilities according to temperature.
        P = np.exp(np.log(P) / temp)

        # Renormalize
        P = P / np.sum(P)

        # Faster than np.random.choice
        Y = np.random.multinomial(1, P, 1)[0]
        X = np.roll(X, -1, axis = 1)
        X[0, -1] = Y

        idx = np.argmax(Y)
        seq.append(idx)
        log_lh += np.log(P[idx])
    return log_lh, seq

def train_model(train, validate,
                weights_path, vocab_size,
                win_size, batch_size, fun):
    model = make_model(win_size, vocab_size)
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(weights_path)
    else:
        SP.print(f'Weights file {weights_path} not found.')

    train_gen = OneHotGenerator(train,
                                batch_size, win_size, vocab_size)
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
              steps_per_epoch = len(train_gen),
              validation_data = validate_gen,
              validation_steps = len(validate_gen),
              verbose = 1,
              shuffle = True,
              epochs = 10,
              callbacks = [cb_checkpoint, cb_generate])
