# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Prettyprint and pretty parse.
from musicgen.defs import period_to_idx

NOTES = [
    "C-", "C#", "D-", "D#", "E-", "F-",
    "F#", "G-", "G#", "A-", "A#", "B-"
]
NOTE_TO_IDX = {n : i for i, n in enumerate(NOTES)}

def notestr_to_idx(notestr):
    note = NOTE_TO_IDX[notestr[:2]]
    octave = int(notestr[2])
    return 12 * octave + note

def idx_to_notestr(idx):
    octave = idx // 12
    note = idx % 12
    return NOTES[note] + str(octave)

def period_to_string(period):
    if period == 0:
        return '...'
    idx = period_to_idx(period)
    octave_idx = idx // 12
    note_idx = idx % 12
    note = NOTES[note_idx]
    return '%s%d' % (note, octave_idx)

def cell_to_string(cell):
    note = period_to_string(cell.period)
    effect_val = (cell.effect_cmd << 8) \
                 + (cell.effect_arg1 << 4) + (cell.effect_arg2)
    sample_idx = cell.sample_idx
    sample_str = '%02X' % sample_idx if sample_idx else '..'
    effect_str = '%03X' % effect_val if effect_val else '...'
    return '%s %s %s' % (note, sample_str, effect_str)

def row_to_string(row):
    return '  '.join(map(cell_to_string, row))

def rows_to_string(rows, numbering = False):
    strings = [row_to_string(row) for row in rows]
    if numbering:
        strings = ['#%04d %s' % (i, row) for i, row in enumerate(strings)]
    return '\n'.join(strings)

def pattern_to_string(pattern):
    return '\n'.join(map(row_to_string, pattern.rows))
