# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""LSTM Model based on piano rolls

Usage:
    train-lstm-rolls.py [-v] [--kb-limit=<i> --win-size=<i>] <corpus>

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --kb-limit=<i>              kb limit [default: 150]
    --win-size=<i>              window size [default: 32]
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
from musicgen.corpus import load_index
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.utils import (SP, file_name_for_params,
                            flatten, load_pickle, save_pickle)
from pathlib import Path
from random import randrange, shuffle
import numpy as np

np.set_printoptions(edgeitems = 20, linewidth = 180)

class WindowGenerator(Sequence):
    def __init__(self, seq, batch_size, win_size):
        self.seq = seq
        self.batch_size = batch_size
        self.win_size = win_size

    def __len__(self):
        n_windows = len(self.seq) - self.win_size
        return int(np.ceil(n_windows / self.batch_size))

    def __getitem__(self, i):
        base = i * self.batch_size

        n_windows = len(self.seq) - self.win_size
        batch_size = min(n_windows - base, self.batch_size)

        X = []
        Y = []
        for i in range(batch_size):
            x = self.seq[base + i:base + i + self.win_size]
            y = self.seq[base + i + self.win_size]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

VOCAB_SIZE = 39

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

    x = Dense(VOCAB_SIZE, activation='softmax')(x)

    model = Model(input_midi, x)
    optimizer = SGD(lr=0.007)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    return model

def notes_to_matrix(notes, sample_props, n_rows):
    # Assign columns for percussive instruments.
    percussion = {s : i % 3 for i, s in enumerate(sample_props)
                  if sample_props[s].is_percussive}

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}

    if not pitches:
        SP.print('No melody.')
        return None

    min_pitch = min(pitch for pitch in pitches)
    max_pitch = max(pitch for pitch in pitches)

    pitch_range = max_pitch - min_pitch
    if pitch_range >= 36:
        SP.print('Pitch range %d too large' % pitch_range)
        return None

    def note_to_triplet(n):
        si = n.sample_idx
        if si in percussion:
            col_idx = percussion[si]
            note_dur = 4
        else:
            col_idx = 3 + n.pitch_idx - min_pitch
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

    # Clip silence.
    last_nonzero_row = np.nonzero(M)[0][-1]
    return M[:last_nonzero_row + 1]

def mod_file_to_piano_roll(file_path):
    SP.header('PARSING %s' % str(file_path))
    try:
        mod = load_file(file_path)
    except PowerPackerModule:
        SP.print('PowerPacker module.')
        return None
    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    props = sample_props(mod, notes)
    mat = notes_to_matrix(notes, props, len(rows))
    SP.leave()
    return mat

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

def load_data_from_disk(corpus_path, mods, win_size):
    file_paths = [corpus_path / mod.genre / mod.fname for mod in mods]
    rolls = [mod_file_to_piano_roll(file_path)
             for file_path in file_paths]
    rolls = [roll for roll in rolls if roll is not None]
    shuffle(rolls)

    padding = np.zeros((win_size, VOCAB_SIZE))
    rolls = [np.vstack((roll, padding)) for roll in rolls]
    seq = np.vstack(rolls)
    print(seq.shape)
    return seq

def load_data(corpus_path, kb_limit, win_size):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]

    size_sum = sum(mod.kb_size for mod in mods)
    params = (size_sum, kb_limit, win_size)
    cache_file = file_name_for_params('cache_rolls', 'npy', params)
    cache_path = corpus_path / cache_file
    if not cache_path.exists():
        seq = load_data_from_disk(corpus_path, mods, win_size)
        with open(cache_path, 'wb') as f:
            np.save(f, seq)
    with open(cache_path, 'rb') as f:
        return np.load(f)

def main():
    args = docopt(__doc__, version = 'LSTM Model trainer 1.0')
    SP.enabled = args['--verbose']
    kb_limit = int(args['--kb-limit'])
    corpus_path = Path(args['<corpus>'])
    win_size = int(args['--win-size'])

    dataset = load_data(corpus_path, kb_limit, win_size)
    n_dataset = len(dataset)

    n_train = int(n_dataset * 0.8)
    n_validate = int(n_dataset * 0.1)
    n_test = n_dataset - n_train - n_validate
    train = dataset[:n_train]
    validate = dataset[n_train:n_train + n_validate]

    batch_size = 256
    train_gen = WindowGenerator(train, batch_size, win_size)
    validate_gen = WindowGenerator(validate, batch_size, win_size)

    model = make_model(win_size, VOCAB_SIZE)
    model.fit(x = train_gen,
             steps_per_epoch = n_train // batch_size,
             validation_data = validate_gen,
             validation_steps = n_validate // batch_size,
             verbose = 1,
             shuffle = True,
             epochs = 10)


if __name__ == '__main__':
    main()
