# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
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
    return [(INSN_SILENCE, 16)] * 8

def to_notes(pcode, rel_pitches):
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

    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(BASE_ROW_TIME * len(row_indices) / max_row)
    for n in notes:
        n.time_ms = row_time_ms

    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (rel_pitches, row_time_ms))

    # Fix durations
    cols = sort_groupby(notes, lambda n: n.col_idx)
    for _, col in cols:
        fix_durations(list(col))
    SP.leave()
    return notes

def metadata(pcode):
    meta = {}
    meta['n_toks'] = len(pcode)

    notes = [(c, a) for (c, a) in pcode if c != INSN_SILENCE]
    meta['n_notes'] = len(notes)

    mel_notes_abs = [a for (c, a) in pcode if c == INSN_PITCH]
    mel_notes_rel = [a for (c, a) in pcode if c == INSN_REL_PITCH]
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

def to_code(notes, rel_pitches, percussion, min_pitch):
    def note_to_event(n):
        si = n.sample_idx
        at = 4 * n.row_idx + n.col_idx
        if si in percussion:
            return at, True, percussion[si]
        return at, False, n.pitch_idx - min_pitch,
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
    last_pitch = None
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

########################################################################
# Test encode and decode
########################################################################
def test_encode_decode(mod_file, rel_pitches):
    from musicgen.code_utils import CODE_MIDI_MAPPING
    from musicgen.generation import notes_to_midi_file
    from musicgen.parser import load_file
    from musicgen.rows import linearize_subsongs
    mod = load_file(mod_file)
    subsongs = linearize_subsongs(mod, 1)
    volumes = [header.volume for header in mod.sample_headers]
    for idx, (_, rows) in enumerate(subsongs):
        notes = rows_to_mod_notes(rows, volumes)
        percussion = guess_percussive_instruments(mod, notes)
        if notes:
            fmt = '%d rows, %d ms/row, percussion %s, %d notes'
            args = len(rows), notes[0].time_ms, percussion, len(notes)
            SP.print(fmt % args)
        pitches = {n.pitch_idx for n in notes
                   if n.sample_idx not in percussion}
        min_pitch = min(pitches, default = 0)
        code = to_code2(notes, rel_pitches, percussion, min_pitch)
        notes = to_notes(code, rel_pitches)
        fname = 'test-%02d.mid' % idx
        notes_to_midi_file(notes, fname, CODE_MIDI_MAPPING)

if __name__ == '__main__':
    from sys import argv
    SP.enabled = True
    test_encode_decode(argv[1], bool(int(argv[2])))
