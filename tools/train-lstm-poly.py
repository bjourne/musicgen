# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Polyphonic LSTM

Usage:
    train-lstm-poly.py [options] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --win-size=<int>       window size [default: 64]
    --kb-limit=<int>       kb limit [default: 150]
    --fraction=<float>     fraction of corpus to use [default: 1.0]
    --relative-pitches     use pcode with relative pitches
"""
from collections import Counter
from docopt import docopt
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding,
                          LSTM)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence, to_categorical
from musicgen.analyze import sample_props
from musicgen.corpus import load_index
from musicgen.generation import notes_to_midi_file
from musicgen.ml import generate_sequence, train_model
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import ModNote, linearize_rows, rows_to_mod_notes
from musicgen.utils import (SP, file_name_for_params,
                            flatten, load_pickle, sort_groupby,
                            save_pickle)
from pathlib import Path
from random import randrange, shuffle
import numpy as np

# This order works best for zodiak_-_gasp.mod
PCODE_MIDI_MAPPING = {
    1 : [-1, 40, 4, 1.0],
    2 : [-1, 36, 4, 1.0],
    3 : [-1, 31, 4, 1.0],
    4 : [1, 48, 4, 1.0]
}

INSN_PITCH = 'P'
INSN_REL_PITCH = 'R'
INSN_SILENCE = 'S'
INSN_DRUM = 'D'
INSN_PROGRAM = 'X'
PAD_TOKEN = (INSN_PROGRAM, 0)

########################################################################
# Encode/Decode
########################################################################
def pcode_to_string(pcode):
    def insn_to_string(insn):
        cmd, arg = insn
        if cmd == INSN_PITCH:
            return '%02d' % arg
        return '%s%s' % (cmd, arg)
    return ' '.join(insn_to_string(insn) for insn in pcode)

def guess_initial_pitch(pcode):
    diffs = [arg for (cmd, arg) in pcode if cmd == INSN_REL_PITCH]
    at_pitch, max_pitch, min_pitch = 0, 0, 0
    for diff in diffs:
        at_pitch += diff
        max_pitch = max(at_pitch, max_pitch)
        min_pitch = min(at_pitch, min_pitch)
    return -min_pitch

def pcode_to_midi_file(pcode, file_path, relative_pitches):
    SP.header('WRITING %s' % file_path)
    if relative_pitches:
        at_pitch = guess_initial_pitch(pcode)

    notes = []
    at = 0
    for cmd, arg in pcode:
        ri = at // 4
        ci = at % 4
        if cmd in (INSN_PITCH, INSN_REL_PITCH, INSN_DRUM):
            if cmd == INSN_DRUM:
                sample_idx = arg + 1
                pitch_idx = 36
            else:
                sample_idx = 4
                if cmd == INSN_PITCH:
                    pitch_idx = arg
                else:
                    at_pitch += arg
                    pitch_idx = at_pitch
            note = ModNote(ri, ci, sample_idx, pitch_idx, 48, -1)
            notes.append(note)
            at += 1
        else:
            at += arg

    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(160 * len(row_indices) / max_row)
    for n in notes:
        n.time_ms = row_time_ms

    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (relative_pitches, row_time_ms))

    # Fix durations
    cols = sort_groupby(notes, lambda n: n.col_idx)
    for _, col in cols:
        col_notes = list(col)
        for n1, n2 in zip(col_notes, col_notes[1:]):
            n1.duration = min(n2.row_idx - n1.row_idx, 16)
        if col_notes:
            last_note = col_notes[-1]
            row_in_page = last_note.row_idx % 64
            last_note.duration = min(64 - row_in_page, 16)
    notes_to_midi_file(notes, file_path, PCODE_MIDI_MAPPING)
    SP.leave()

def guess_percussive_instruments(mod, notes):
    props = sample_props(mod, notes)
    props = [(s, p.n_notes, p.is_percussive) for (s, p) in props.items()
             if p.is_percussive]

    # Sort by the number of notes so that the same instrument
    # assignment is generated every time.
    props = list(reversed(sorted(props, key = lambda x: x[1])))
    percussive_samples = [s for (s, _, _) in props]

    return {s : i % 3 for i, s in enumerate(percussive_samples)}

def mod_file_to_pcode(file_path, relative_pitches):
    SP.header('READING %s' % file_path)
    try:
        mod = load_file(file_path)
    except PowerPackerModule:
        SP.print('PowerPacker module.')
        SP.leave()
        return

    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)

    percussion = guess_percussive_instruments(mod, notes)
    fmt = 'Row time %d ms, guessed percussion: %s.'
    SP.print(fmt % (notes[0].time_ms, percussion))

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}
    if not pitches:
        SP.print('No melody.')
        SP.leave()
        return
    min_pitch = min(pitch for pitch in pitches)
    max_pitch = max(pitch for pitch in pitches)
    pitch_range = max_pitch - min_pitch
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        SP.leave()
        return

    def note_to_event(n):
        si = n.sample_idx
        at = 4 * n.row_idx + n.col_idx
        if si in percussion:
            return at, True, percussion[si]
        return at, False, n.pitch_idx - min_pitch,
    notes = sorted([note_to_event(n) for n in notes])

    if relative_pitches:
        # Make pitches relative
        current_pitch = None
        notes2 = []
        for at, is_drum, pitch in notes:
            if is_drum:
                notes2.append((at, True, pitch))
            else:
                if current_pitch is None:
                    notes2.append((at, False, 0))
                else:
                    notes2.append((at, False, pitch - current_pitch))
                current_pitch = pitch
        notes = notes2

    def produce_silence(delta):
        thresholds = [16, 8, 4, 3, 2, 1]
        for threshold in thresholds:
            while delta >= threshold:
                yield threshold
                delta -= threshold
        assert delta >= -1

    # Begin with pad token
    yield PAD_TOKEN

    at = 0
    last_pitch = None
    for ofs, is_drum, arg in notes:
        delta = ofs - at
        for sil in produce_silence(delta - 1):
            yield INSN_SILENCE, sil
        if is_drum:
            yield INSN_DRUM, arg
        elif relative_pitches:
            yield INSN_REL_PITCH, arg
        else:
            yield INSN_PITCH, arg
        at = ofs
    SP.leave()

########################################################################
# Test encode and decode
########################################################################
def test_encode_decode(mod_file, relative_pitches):
    pcode = list(mod_file_to_pcode(mod_file, relative_pitches))
    pcode_to_midi_file(pcode, 'test.mid', relative_pitches)

########################################################################
# Cache loading
########################################################################
def load_data_from_disk(corpus_path, mods, relative_pitches):
    file_paths = [corpus_path / mod.genre / mod.fname for mod in mods]
    pcodes = [mod_file_to_pcode(fp, relative_pitches)
              for fp in file_paths]
    shuffle(pcodes)
    return flatten(pcodes)

def load_data(corpus_path, kb_limit, relative_pitches):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]
    size_sum = sum(mod.kb_size for mod in mods)
    params = (size_sum, kb_limit, relative_pitches)
    cache_file = file_name_for_params('cache_poly', 'pickle', params)
    cache_path = corpus_path / cache_file
    if not cache_path.exists():
        SP.print('Cache file %s not found.' % cache_file)
        seq = load_data_from_disk(corpus_path, mods, relative_pitches)
        save_pickle(cache_path, seq)
    return load_pickle(cache_path)

########################################################################
# Model definition
########################################################################
def make_model(seq_len, vocab_size):
    # Check if an Embedding layer is worthwhile.
    model = Sequential()
    model.add(LSTM(128, return_sequences = True,
                   input_shape = (seq_len, vocab_size)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    print(model.summary())
    return model

########################################################################
# Data generator
########################################################################
class DataGen(Sequence):
    def __init__(self, seq, batch_size, win_size, vocab_size):
        self.seq = seq
        self.batch_size = batch_size
        self.win_size = win_size
        self.vocab_size = vocab_size

    def __len__(self):
        n_windows = len(self.seq) - self.win_size
        return int(np.ceil(n_windows / self.batch_size))

    def __getitem__(self, i):
        base = i * self.batch_size

        # Fix running over the edge.
        n_windows = len(self.seq) - self.win_size
        batch_size = min(n_windows - base, self.batch_size)

        X = np.zeros((batch_size, self.win_size, self.vocab_size),
                     dtype = np.bool)
        Y = np.zeros(batch_size)
        for i in range(batch_size):
            for j in range(self.win_size):
                X[i, j, self.seq[base + i + j]] = 1
            Y[i] = self.seq[base + i + self.win_size]
        return X, Y


########################################################################
# Generation
########################################################################
def generate_midi_files(model, epoch, seq,
                        vocab_size, win_size,
                        char2idx, idx2char, corpus_path,
                        relative_pitches):
    SP.header('EPOCH', '%d', epoch)
    # Pick a seed that doesn't contain the break token.
    eos = char2idx[PAD_TOKEN]
    while True:
        idx = randrange(len(seq) - win_size)
        seed = np.array(seq[idx:idx + win_size])
        if not eos in seed:
            break
    join_seq = [char2idx[(INSN_SILENCE, 8)]] * 4

    i = randrange(len(seq) - win_size)
    seed = list(seq[i:i + win_size])
    S = to_categorical(seed, vocab_size)
    temps = [0.5, 0.8, 1.0, 1.2, 1.5]
    for temp in temps:
        SP.header('TEMPERATURE %.2f' % temp)
        log_lh, seq = generate_sequence(model, S, 300, temp, eos)

        seq = seed + join_seq + seq
        seq = [idx2char[i] for i in seq]
        SP.print('logLH: %.4f', log_lh)
        SP.print(pcode_to_string(seq))
        file_name = 'pcode-%03d-%.2f.mid' % (epoch, temp)
        file_path = corpus_path / file_name
        pcode_to_midi_file(seq, file_path, relative_pitches)
        SP.leave()
    SP.leave()

def analyze_pcode(pcode):
    counts = Counter(pcode)
    for (cmd, arg), cnt in sorted(counts.items()):
        SP.print('%s %3d %10d' % (cmd, arg, cnt))

def main():
    args = docopt(__doc__, version = 'Train Poly LSTM 1.0')
    SP.enabled = args['--verbose']
    kb_limit = int(args['--kb-limit'])
    corpus_path = Path(args['<corpus-path>'])
    win_size = int(args['--win-size'])
    relative_pitches = args['--relative-pitches']

    # Load dataset
    if corpus_path.is_dir():
        dataset = load_data(corpus_path, kb_limit, relative_pitches)
        output_dir = corpus_path
    else:
        dataset = list(mod_file_to_pcode(corpus_path, relative_pitches))
        output_dir = Path('.')
    n_dataset = len(dataset)
    analyze_pcode(dataset)

    # Convert to integer sequence
    n_dataset = len(dataset)
    chars = sorted(set(dataset))
    vocab_size = len(chars)
    SP.print('%d tokens and %d token types.', (n_dataset, vocab_size))
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}
    dataset = np.array([char2idx[c] for c in dataset])

    # Split data
    n_train = int(n_dataset * 0.8)
    n_validate = int(n_dataset * 0.1)
    n_test = n_dataset - n_train - n_validate
    train = dataset[:n_train]
    validate = dataset[n_train:n_train + n_validate]
    test = dataset[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))

    # Data generators
    batch_size = 128
    train_gen = DataGen(train, batch_size, win_size, vocab_size)
    validate_gen = DataGen(validate, batch_size, win_size, vocab_size)

    # Create model and maybe load from disk.
    model = make_model(win_size, vocab_size)

    params = (win_size, n_train, n_validate, relative_pitches)
    weights_file = file_name_for_params('weights_poly', 'hdf5', params)
    weights_path = output_dir / weights_file
    if weights_path.exists():
        SP.print(f'Loading weights from {weights_path}.')
        model.load_weights(weights_path)
    else:
        SP.print(f'Weights file {weights_path} not found.')

    def on_epoch_begin(epoch, logs):
        generate_midi_files(model, epoch, test,
                            vocab_size, win_size,
                            char2idx, idx2char, output_dir,
                            relative_pitches)
    cb_cp1 = ModelCheckpoint(str(weights_path))
    best_weights_path = '%s-best-{val_loss:.2f}.hdf' % weights_path.stem
    cb_cp2 = ModelCheckpoint(
        str(best_weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min')

    cb_generate = LambdaCallback(on_epoch_begin = on_epoch_begin)
    model.fit(x = train_gen,
              steps_per_epoch = len(train_gen),
              validation_data = validate_gen,
              validation_steps = len(validate_gen),
              verbose = 1,
              shuffle = True,
              epochs = 20,
              callbacks = [cb_cp1, cb_cp2, cb_generate])

if __name__ == '__main__':
    main()
