# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from music21.chord import Chord
from music21.converter import parse
from music21.note import Note

SAMPLE_FREQ = 12

def parse_midi_el(el):

    dur = SAMPLE_FREQ * el.duration.quarterLength
    int_dur = int(dur)
    assert dur == int_dur

    ofs = SAMPLE_FREQ * el.offset
    int_ofs = int(ofs)
    assert ofs == int_ofs

    if isinstance(el, Note):
        return {(int_ofs, int_dur, el.pitch.midi)}
    elif isinstance(el, Chord):
        return {(int_ofs, int_dur, p.midi) for p in el.pitches}
    else:
        return set()

def parse_midi_notes(fname):
    midi = parse(fname)
    return set.union(*[parse_midi_el(e) for e in midi.recurse()])
