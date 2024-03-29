# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
#
# PCode stands for parallel or polyphonic code.
from musicgen.code_utils import (BASE_ROW_TIME,
                                 INSN_PITCH,
                                 INSN_REL_PITCH,
                                 INSN_SILENCE,
                                 INSN_DRUM,
                                 fix_durations,
                                 guess_initial_pitch,
                                 guess_percussive_instruments)
from musicgen.rows import ModNote, rows_to_mod_notes
from musicgen.utils import SP, sort_groupby

def pause():
    return [(INSN_SILENCE, 16)] * 4

def to_notes_without_tempo(pcode, rel_pitches):
    if rel_pitches:
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
        elif cmd == INSN_SILENCE:
            at += arg
        else:
            assert False
    return notes

def estimate_row_time(code, rel_pitches):
    notes = to_notes_without_tempo(code, rel_pitches)
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices, default = 0)
    if max_row == 0:
        return 100
    row_time = int(BASE_ROW_TIME * len(row_indices) / max_row)
    return max(row_time, 80)

def to_notes(pcode, rel_pitches, row_time):
    notes = to_notes_without_tempo(pcode, rel_pitches)

    for n in notes:
        n.time_ms = row_time

    fmt = 'Rel pitches: %s, row time: %s.'
    SP.print(fmt % (rel_pitches, row_time))

    # Fix durations
    cols = sort_groupby(notes, lambda n: n.col_idx)
    for _, col in cols:
        fix_durations(list(col))
    return notes

def to_code(notes, rel_pitches, percussion):
    def note_to_event(n):
        si = n.sample_idx
        at = 4 * n.row_idx + n.col_idx
        if si in percussion:
            return at, True, percussion[si]
        return at, False, n.pitch_idx
    notes = sorted([note_to_event(n) for n in notes])

    if rel_pitches:
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

    at = 0
    for ofs, is_drum, arg in notes:
        delta = ofs - at
        for sil in produce_silence(delta - 1):
            yield INSN_SILENCE, sil
        if is_drum:
            yield INSN_DRUM, arg
        elif rel_pitches:
            yield INSN_REL_PITCH, arg
        else:
            yield INSN_PITCH, arg
        at = ofs
