# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# SCode stands for simple code
from musicgen.code_utils import (CODE_MIDI_MAPPING,
                                 INSN_PITCH,
                                 INSN_REL_PITCH,
                                 INSN_SILENCE,
                                 INSN_DRUM,
                                 fix_durations,
                                 guess_initial_pitch,
                                 guess_percussive_instruments)
from musicgen.generation import notes_to_midi_file
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import ModNote, linearize_rows, rows_to_mod_notes
from musicgen.utils import SP, flatten
from random import shuffle
import numpy as np

MAX_COMPRESSED_SILENCE = 16
ROW_TIME_FACTOR_MS = 240

def scode_short_pause():
    return [(INSN_SILENCE, 8)]

def scode_long_pause():
    return [(INSN_SILENCE, 16)]

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

    if not notes and not compress_silence:
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
    if at < n_rows and not compress_silence:
        for sil in produce_silence(n_rows - at, compress_silence):
            yield INSN_SILENCE, sil

def mod_file_to_scode4(file_path, rel_pitches, compress_silence):
    try:
        mod = load_file(file_path)
    except PowerPackerModule:
        SP.print('PowerPacker module.')
        return None

    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    if not notes:
        SP.print('Empty module.')
        return None

    percussion = guess_percussive_instruments(mod, notes)
    fmt = 'Row time %d ms, guessed percussion: %s.'
    SP.print(fmt % (notes[0].time_ms, percussion))

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}
    if not pitches:
        SP.print('No melody.')
        return None
    min_pitch = min(pitch for pitch in pitches)
    max_pitch = max(pitch for pitch in pitches)
    pitch_range = max_pitch - min_pitch
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        return None

    # Align pitches to 0 base
    for note in notes:
        note.pitch_idx -= min_pitch

    # No groupby.
    cols = [[n for n in notes if n.col_idx == i] for i in range(4)]

    n_rows = len(rows)
    scode4 = [list(mod_notes_to_scode(col, n_rows, percussion,
                                      rel_pitches, compress_silence))
             for col in cols]
    return scode4

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
    fix_durations(notes)
    return notes

def guess_and_set_row_time(notes):
    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(ROW_TIME_FACTOR_MS * len(row_indices) / max_row)
    row_time_ms = max(row_time_ms, 60)
    for n in notes:
        n.time_ms = row_time_ms
    return row_time_ms

def scode4_to_midi_file(scode4, file_path, rel_pitches):
    notes = flatten([scode_to_mod_notes(scode, i, rel_pitches)
                     for (i, scode) in enumerate(scode4)])
    row_time_ms = guess_and_set_row_time(notes)
    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (rel_pitches, row_time_ms))
    notes_to_midi_file(notes, file_path, CODE_MIDI_MAPPING)

def scode_to_midi_file(scode, file_path, rel_pitches):
    notes = scode_to_mod_notes(scode, 0, rel_pitches)
    row_time_ms = row_time_ms = guess_and_set_row_time(notes)
    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (rel_pitches, row_time_ms))
    notes_to_midi_file(notes, file_path, CODE_MIDI_MAPPING)

def mod_file_to_scode(file_path, rel_pitches):
    # Since the output is monophonic we compress silences.
    scode4 = mod_file_to_scode4(file_path, rel_pitches, True)
    if not scode4:
        return None

    # Add join tokens to the first 3 columns but not the last which is
    # handled by model-trainer.py. Maybe change it in the future.
    long_pause = scode_long_pause()
    for scode in scode4[:-1]:
        scode.extend(long_pause)
    return flatten(scode4)

def test_encode_decode4(mod_file, rel_pitches):
    scode4 = list(mod_file_to_scode4(mod_file, rel_pitches, False))
    scode4_to_midi_file(scode4, 'test4.mid', rel_pitches)

def test_encode_decode(mod_file, rel_pitches):
    scode = list(mod_file_to_scode(mod_file, rel_pitches))
    scode_to_midi_file(scode, 'test.mid', rel_pitches)

if __name__ == '__main__':
    from sys import argv
    SP.enabled = True
    test_encode_decode(argv[1], bool(int(argv[2])))
