# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from music21.chord import Chord
from music21.converter import parse
from music21.note import Note

def parse_midi_el(el):
    dur = int(4 * el.duration.quarterLength)
    ofs = int(4 * el.offset)
    if isinstance(el, Note):
        return {(ofs, dur, el.pitch.midi)}
    elif isinstance(el, Chord):
        return {(ofs, dur, p.midi) for p in el.pitches}
    return set()

def parse_midi_notes(fname):
    midi = parse(fname)
    return set.union(*[parse_midi_el(e) for e in midi.recurse()])
