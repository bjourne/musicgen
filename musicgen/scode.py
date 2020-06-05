# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# SCode stands for simple code
from musicgen.code_utils import guess_percussive_instruments
from musicgen.corpus import load_index
from musicgen.generation import notes_to_midi_file
from musicgen.parser import PowerPackerModule, load_file
from musicgen.pcode import pcode_to_string
from musicgen.rows import ModNote, linearize_rows, rows_to_mod_notes
from musicgen.utils import (SP,
                            encode_training_sequence,
                            file_name_for_params, flatten,
                            load_pickle_cache)
from random import shuffle
import numpy as np

SCODE_MIDI_MAPPING = {
    1 : [-1, 40, 4, 1.0],
    2 : [-1, 36, 4, 1.0],
    3 : [-1, 31, 4, 1.0],
    4 : [1, 48, 4, 1.0]
}

INSN_PITCH = 'P'
INSN_REL_PITCH = 'R'
INSN_SILENCE = 'S'
INSN_DRUM = 'D'
MAX_COMPRESSED_SILENCE = 16

def produce_silence(delta, compress_silence):
    if compress_silence:
        delta = min(delta, MAX_COMPRESSED_SILENCE)
    thresholds = [64, 32, 16, 8, 4, 3, 2, 1]
    for threshold in thresholds:
        while delta >= threshold:
            yield threshold
            delta -= threshold
    assert delta == 0

def mod_notes_to_scode(notes, n_rows, percussion, rel_pitches,
                       compress_silence):

    if rel_pitches:
        # Make pitches relative
        current_pitch = None
        for note in notes:
            if note.sample_idx in percussion:
                continue
            if current_pitch is None:
                rel_pitch = 0
            else:
                rel_pitch = note.pitch_idx - current_pitch
            current_pitch = note.pitch_idx
            note.pitch_idx = rel_pitch

    if not notes:
        for sil in produce_silence(n_rows, compress_silence):
            yield INSN_SILENCE, sil
        return
    at = 0
    for note in notes:
        row_idx = note.row_idx
        sample_idx = note.sample_idx
        delta = row_idx - at
        if delta > 0:
            for sil in produce_silence(delta, compress_silence):
                yield INSN_SILENCE, sil
        if sample_idx in percussion:
            yield INSN_DRUM, percussion[sample_idx]
        elif rel_pitches:
            yield INSN_REL_PITCH, note.pitch_idx
        else:
            yield INSN_PITCH, note.pitch_idx
        at = row_idx + 1
    if at < n_rows:
        for sil in produce_silence(n_rows - at, compress_silence):
            yield INSN_SILENCE, sil

def mod_file_to_scode(file_path, rel_pitches, compress_silence):
    SP.header('READING %s' % file_path)
    try:
        mod = load_file(file_path)
    except PowerPackerModule:
        SP.print('PowerPacker module.')
        SP.leave()
        return None

    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    if not notes:
        SP.print('Empty module.')
        SP.leave()
        return None

    percussion = guess_percussive_instruments(mod, notes)
    fmt = 'Row time %d ms, guessed percussion: %s.'
    SP.print(fmt % (notes[0].time_ms, percussion))

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}
    if not pitches:
        SP.print('No melody.')
        SP.leave()
        return None
    min_pitch = min(pitch for pitch in pitches)
    max_pitch = max(pitch for pitch in pitches)
    pitch_range = max_pitch - min_pitch
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        SP.leave()
        return None

    # Align pitches to 0 base
    for note in notes:
        note.pitch_idx -= min_pitch

    # No groupby.
    cols = [[n for n in notes if n.col_idx == i] for i in range(4)]

    n_rows = len(rows)
    scode = [list(mod_notes_to_scode(col, n_rows, percussion,
                                     rel_pitches, compress_silence))
             for col in cols]
    SP.leave()
    return scode

def guess_initial_pitch(scode):
    diffs = [arg for (cmd, arg) in scode if cmd == INSN_REL_PITCH]
    at_pitch, max_pitch, min_pitch = 0, 0, 0
    for diff in diffs:
        at_pitch += diff
        max_pitch = max(at_pitch, max_pitch)
        min_pitch = min(at_pitch, min_pitch)
    return -min_pitch

def scode_to_mod_notes(scode, ci, rel_pitches):
    if rel_pitches:
        at_pitch = guess_initial_pitch(scode)
    ri = 0
    notes = []
    for cmd, arg in scode:
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
            ri += 1
        elif cmd == INSN_SILENCE:
            ri += arg
        else:
            assert False
    # Fix durations
    for n1, n2 in zip(notes, notes[1:]):
        n1.duration = min(n2.row_idx - n1.row_idx, 16)
    if notes:
        last_note = notes[-1]
        row_in_page = last_note.row_idx % 64
        last_note.duration = min(64 - row_in_page, 16)
    return notes

def scode_to_midi_file(cols, file_path, rel_pitches):
    notes = flatten([scode_to_mod_notes(col, i, rel_pitches)
                     for (i, col) in enumerate(cols)])

    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(240 * len(row_indices) / max_row)
    for n in notes:
        n.time_ms = row_time_ms
    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (rel_pitches, row_time_ms))
    notes_to_midi_file(notes, file_path, SCODE_MIDI_MAPPING)

def test_encode_decode(mod_file, rel_pitches):
    scode = list(mod_file_to_scode(mod_file, rel_pitches, False))
    print(scode)
    scode_to_midi_file(scode, 'test.mid', rel_pitches)

########################################################################
# Cache loading
########################################################################
def pad_with_silence(cols):
    # For now, eight bars of silence is appended to every column, but
    # I should solve this better in the future.
    EOS_SILENCE = [(INSN_SILENCE, MAX_COMPRESSED_SILENCE)] * 2
    for col in cols:
        col.extend(EOS_SILENCE)
    return cols

def build_corpus(corpus_path, mods, rel_pitches, compress_silence):
    file_paths = [corpus_path / mod.genre / mod.fname for mod in mods]
    scode_per_mod = [mod_file_to_scode(fp, rel_pitches, compress_silence)
                     for fp in file_paths]
    scode_per_mod = [cols for cols in scode_per_mod if cols]
    cols = flatten(scode_per_mod)

    avg_len = np.mean([len(c) for c in cols])
    SP.print('Average column length %.2f.' % avg_len)

    shuffle(cols)
    cols = pad_with_silence(cols)
    return encode_training_sequence(flatten(cols))

def load_corpus(corpus_path, kb_limit, rel_pitches, compress_silence):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]
    size_sum = sum(mod.kb_size for mod in mods)
    params = (size_sum, kb_limit, rel_pitches, compress_silence)
    cache_file = file_name_for_params('cached_scode', 'pickle', params)
    cache_path = corpus_path / cache_file
    def rebuild_fun():
        return build_corpus(corpus_path, mods,
                            rel_pitches, compress_silence)
    return load_pickle_cache(cache_path, rebuild_fun)

def load_mod_file(mod_file, rel_pitches, compress_silence):
    cols = mod_file_to_scode(mod_file, rel_pitches, compress_silence)
    cols = pad_with_silence(cols)
    scode = flatten(cols)
    return encode_training_sequence(scode)

if __name__ == '__main__':
    from sys import argv
    from pathlib import Path
    SP.enabled = True
    load_corpus(Path('~/musicgen').expanduser(), 150, False, True)
    #test_encode_decode(argv[1], False)
