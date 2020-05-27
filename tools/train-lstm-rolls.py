# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""LSTM Model based on piano rolls

Usage:
    train-gain-rolls.py [-v] [--kb-limit=<int>] module <mod>

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --kb-limit=<int>            kb limit [default: 150]
"""
from docopt import docopt
from keras import Input
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Multiply, Permute, RepeatVector,
                          LeakyReLU, LSTM)
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, SGD
from keras.utils import Sequence, to_categorical
from mido import Message, MidiFile, MidiTrack
from musicgen.analyze import sample_props
from musicgen.defs import period_to_idx
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.utils import (SP, file_name_for_params,
                            flatten, load_pickle, save_pickle)
import numpy as np

np.set_printoptions(edgeitems = 20, linewidth = 180)

def make_model(seq_len, n_chars):
    model = Sequential()
    model.add(LSTM(256, return_sequences = True,
                   input_shape = (seq_len, n_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    return model

def make_model2(seq_len, n_chars):
    midi_shape = (seq_len, n_chars)

    input_midi = Input(midi_shape)

    x = LSTM(1024, return_sequences=True,
             unit_forget_bias=True)(input_midi)
    x = LeakyReLU()(x)
    x = BatchNormalization() (x)
    x = Dropout(0.3)(x)

    # compute importance for each step
    attn1 = Dense(1, activation='tanh')(x)
    attn1 = Flatten()(attn1)
    attn1 = Activation('softmax')(attn1)
    attn1 = RepeatVector(1024)(attn1)
    attn1 = Permute([2, 1])(attn1)

    mult1 = Multiply()([x, attn1])
    sent_repr1 = Dense(512)(mult1)

    x = Dense(512)(sent_repr1)
    x = LeakyReLU()(x)
    x = BatchNormalization() (x)
    x = Dropout(0.22)(x)

    x = LSTM(512, return_sequences=True, unit_forget_bias=True)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization() (x)
    x = Dropout(0.22)(x)

    # compute importance for each step
    attn2 = Dense(1, activation='tanh')(x)
    attn2 = Flatten()(attn2)
    attn2 = Activation('softmax')(attn2)
    attn2 = RepeatVector(512)(attn2)
    attn2 = Permute([2, 1])(attn2)

    mult2 = Multiply()([x, attn2])
    sent_repr2 = Dense(256)(mult2)

    x = Dense(256)(sent_repr2)
    x = LeakyReLU()(x)
    x = BatchNormalization() (x)
    x = Dropout(0.22)(x)

    x = LSTM(128, unit_forget_bias=True)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization() (x)
    x = Dropout(0.22)(x)

    x = Dense(39, activation='softmax')(x)

    model = Model(input_midi, x)
    optimizer = SGD(lr=0.007)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer)
    return model

def notes_to_matrix(notes, sample_props, n_rows):
    # Assign columns for percussive instruments.
    percussion = {s : i % 3 for i, s in enumerate(sample_props)
                  if sample_props[s].is_percussive}

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}

    pitch_delta = min(pitch for pitch in pitches)
    SP.print('Transposing %d steps.' % pitch_delta)

    def note_to_triplet(n):
        si = n.sample_idx
        if si in percussion:
            col_idx = percussion[si]
            note_dur = 4
        else:
            col_idx = 3 + n.pitch_idx - pitch_delta
            assert col_idx >= 3
            sample_dur = sample_props[si].note_duration
            # Should be correct since it is set in rows_to_mod_notes
            note_dur = min(n.duration, sample_dur)
        return n.row_idx, col_idx, note_dur
    notes = sorted([note_to_triplet(n) for n in notes])

    M = np.zeros((n_rows + 4, 3 + 36))

    # Fill matrix with notes
    for (row, col, dur) in notes:
        M[row][col] = 1.0
        assert dur > 0
        for fol in range(dur - 1):
            M[row + fol + 1][col] = 0.5

    last_nonzero_row = np.nonzero(M)[0][-1]
    return M[:last_nonzero_row + 1]

def mod_file_to_piano_roll(file_path):
    mod = load_file(file_path)
    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    props = sample_props(mod, notes)
    return notes_to_matrix(notes, props, len(rows))

def stop_note(note, time):
    return Message('note_off', note = note, velocity = 0, time = time)

def start_note(note, time):
    return Message('note_on', note = note, velocity = 127, time = time)

def piano_roll_to_track(mat):
    # Time to emit
    delta = 0
    # Current state of the notes in the roll.
    notes = [False] * len(mat[0])
    for row in mat:
        for i, col in enumerate(row):
            if col == 1.0:
                if notes[i]:
                    # First stop the ringing note
                    yield stop_note(i, delta)
                    delta = 0
                yield start_note(i, delta)
                delta = 0
                notes[i] = True
            elif col == 0:
                if notes[i]:
                    # Stop the ringing note
                    yield stop_note(i, delta)
                    delta = 0
                notes[i] = False
        # ms per row
        delta += 1

def piano_roll_to_midi_file(mat, midi_file):
    row_time = guess_time_ms(mat)

    # Produce drum track
    drum_mat = mat[:,0:3]
    drum_instruments = [36, 31, 40]
    drum_track = list(piano_roll_to_track(drum_mat))
    for msg in drum_track:
        msg.note = drum_instruments[msg.note]
        msg.channel = 9
        msg.time *= row_time

    # Produce piano track
    piano_mat = mat[:,3:]
    piano_track = list(piano_roll_to_track(piano_mat))
    for msg in piano_track:
        msg.note = msg.note + 48
        msg.channel = 1
        msg.time *= row_time

    midi = MidiFile(type = 1)
    midi.tracks.append(MidiTrack(drum_track))
    midi.tracks.append(MidiTrack(piano_track))
    midi.save(midi_file)

def guess_time_ms(mat):
    mat2 = mat[np.count_nonzero(mat == 1.0, axis = 1) > 0]
    zero_ratio = len(mat2) / len(mat)
    row_time = int(160 * zero_ratio)
    SP.print('Guessed row time %d ms.' % row_time)
    return row_time

def main():
    args = docopt(__doc__, version = 'LSTM Model trainer 1.0')
    SP.enabled = args['--verbose']

    mod_file = args['<mod>']
    mat = mod_file_to_piano_roll(mod_file)

    SEQ_LEN = 32

    X = []
    Y = []
    for i in range(0, len(mat) - SEQ_LEN):
        X.append(mat[i:i + SEQ_LEN])
        Y.append(mat[i + SEQ_LEN])
    X = np.array(X)
    Y = np.array(Y)
    print(mat.shape)
    model = make_model3(32, mat.shape[1])
    print(model.summary())

    model.fit(X, Y, batch_size = 32, epochs = 50, shuffle = True)



if __name__ == '__main__':
    main()
