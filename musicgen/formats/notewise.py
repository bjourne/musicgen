# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Parser for the Notwise encoding.
from music21.chord import Chord
from music21.converter import parse
from music21.duration import Duration
from music21.instrument import Piano, partitionByInstrument
from music21.note import Note
from music21.stream import Stream
from re import match
from sys import argv

def tokens_to_notes(tokens):
    offset = 0
    active_notes = {}
    notes = []

    def handle_wait(arg):
        return offset + 1/8 * arg
    def handle_press(arg):
        note = Note(arg + 33)
        note.offset = offset
        note.storedInstrument = Piano()
        notes.append(note)
        active_notes[arg] = note
    def handle_release(arg):
        note = active_notes[arg]
        note.duration = Duration(offset - note.offset)
        del active_notes[arg]
    handlers = [
        (r'^wait(\d+)$', handle_wait),
        (r'^p(\d+)$', handle_press),
        (r'^endp(\d+)$', handle_release)
    ]
    for tok in tokens:
        for expr, handler in handlers:
            m = match(expr, tok)
            if m:
                res = handler(int(m.group(1)))
                if res is not None:
                    offset = res
                break
        else:
            print(f'No handler for "{tok}"!')
    return notes


def to_midi(text, fname):
    tokens = text.split()
    notes = tokens_to_notes(tokens)
    print(f'Writing {len(notes)} notes to {fname}.')
    stream = Stream(notes)
    stream.write('midi', fp = fname)

if __name__ == '__main__':
    fname = argv[1]
    with open(fname) as f:
        text = f.read()
    to_midi(text, 'out.mid')
