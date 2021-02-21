# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
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
from musicgen.rows import ModNote, linearize_rows, rows_to_mod_notes
from musicgen.utils import SP, flatten
from random import shuffle
import numpy as np

MAX_COMPRESSED_SILENCE = 16
ROW_TIME_FACTOR_MS = 240

def pause():
    return [(INSN_SILENCE, 16)]

def produce_silence(delta):
    delta = min(delta, MAX_COMPRESSED_SILENCE)
    thresholds = [64, 32, 16, 8, 4, 3, 2, 1]
    for threshold in thresholds:
        while delta >= threshold:
            yield threshold
            delta -= threshold
    assert delta == 0

def mod_notes_to_scode(notes, percussion):
    at = 0
    for note in notes:
        row_idx = note.row_idx
        sample_idx = note.sample_idx
        delta = row_idx - at
        if delta > 0:
            for sil in produce_silence(delta):
                yield INSN_SILENCE, sil
        if sample_idx in percussion:
            yield INSN_DRUM, percussion[sample_idx]
        else:
            yield INSN_PITCH, note.pitch_idx
        at = row_idx + 1

def to_code(notes, rel_pitches, percussion, min_pitch):
    # Align pitches to 0 base
    for note in notes:
        note.pitch_idx -= min_pitch

    # No groupby.
    cols = [[n for n in notes if n.col_idx == i] for i in range(4)]

    # n_rows = len(rows)
    code4 = [list(mod_notes_to_scode(col, percussion)) for col in cols]
    # Add join tokens to the first 3 columns but not the last which is
    # handled by model-trainer.py. Maybe change it in the future.
    for code in code4[:-1]:
        code.extend(pause())
    code = flatten(code4)

    if rel_pitches:
        current_pitch = None
        rel_code = []
        for cmd, arg in code:
            if cmd != INSN_PITCH:
                rel_code.append((cmd, arg))
            else:
                if current_pitch is None:
                    rel_code.append((INSN_REL_PITCH, 0))
                else:
                    diff = arg - current_pitch
                    rel_code.append((INSN_REL_PITCH, diff))
                current_pitch = arg
        code = rel_code
    return code

def guess_and_set_row_time(notes, row_time_factor):
    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(ROW_TIME_FACTOR_MS * len(row_indices) / max_row)
    row_time_ms = max(row_time_ms, 60)
    for n in notes:
        n.time_ms = row_time_ms
    return row_time_ms

def to_notes(scode, rel_pitches):
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
            note = ModNote(ri, 0, sample_idx, pitch_idx, 48, -1)
            notes.append(note)
            ri += 1
        elif cmd == INSN_SILENCE:
            ri += arg
        else:
            assert False
    fix_durations(notes)
    guess_and_set_row_time(notes, ROW_TIME_FACTOR_MS)
    return notes

def metadata(code):
    meta = {}
    meta['n_toks'] = len(code)

    notes = [(c, a) for (c, a) in code if c != INSN_SILENCE]
    meta['n_notes'] = len(notes)

    mel_notes_abs = [a for (c, a) in code if c == INSN_PITCH]
    mel_notes_rel = [a for (c, a) in code if c == INSN_REL_PITCH]
    rel_pitches = True if mel_notes_rel else False
    mel_notes = mel_notes_rel if rel_pitches else mel_notes_abs

    meta['n_unique_notes'] = len(set(mel_notes))

    at = 0
    lo, hi = 0, 0
    for mel_note in mel_notes:
        if rel_pitches:
            at += mel_note
        else:
            at = mel_note
        lo = min(at, lo)
        hi = max(at, hi)
    meta['pitch_range'] = hi - lo
    return meta

def test_encode_decode(mod_file, rel_pitches):
    mod = load_file(mod_file)
    scode = list(to_code(mod, rel_pitches, True))
    notes = to_notes(scode, rel_pitches)
    notes_to_midi_file(notes, 'test.mid', CODE_MIDI_MAPPING)

if __name__ == '__main__':
    from musicgen.parser import load_file
    from sys import argv
    SP.enabled = True
    test_encode_decode(argv[1], bool(int(argv[2])))
