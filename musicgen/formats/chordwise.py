# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Parser for the Chordwise encoding.
from musicgen.formats import SAMPLE_FREQ, parse_midi_notes
from sys import argv
import numpy as np

NOTE_OFFSET = 45
NOTE_RANGE = 38

def normalize_notes(notes):
    out = {}
    for ofs, dur, midi in notes:
        idx = midi - NOTE_OFFSET
        while idx < 0:
            idx += 12
        while idx >= NOTE_RANGE:
            idx -= 12
        if (ofs, idx) in out:
            out[ofs, idx] = max(dur, out[ofs, idx])
        else:
            out[ofs, idx] = dur
    return {(ofs, dur, idx) for ((ofs, idx), dur) in out.items()}

def from_midi(fname, is_chamber_music):
    notes = parse_midi_notes(fname)

    # Sort the notes so that earlier notes doesn't overwrite later
    # ones.
    notes = sorted(normalize_notes(notes))
    n_rows = max(ofs + dur for (ofs, dur, _) in notes)
    mat = np.zeros((n_rows + 1, NOTE_RANGE), dtype = int)
    for ofs, dur, idx in notes:
        mat[ofs, idx] = 1
        mat[ofs+1:ofs+dur, idx] = 2
    return mat

if __name__ == '__main__':
    fname = argv[1]
    for row in from_midi(fname, False):
        print(''.join(map(str, row)))
