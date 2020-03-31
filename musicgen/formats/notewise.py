# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Parser for the Notewise encoding.
from music21.duration import Duration
from music21.instrument import Piano, partitionByInstrument
from music21.midi import MidiFile
from music21.midi.translate import midiFileToStream
from music21.note import Note
from music21.stream import Stream
from musicgen.formats import parse_midi_notes
from musicgen.formats.chordwise import normalize_notes
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

# Instruments that sound like violins and pianos.

VIOLIN_LIKE = {
    "Bassoon",
    "Violin", "Viola", "Cello", "Violincello", "Violoncello", "Flute",
    "Oboe", "Clarinet", "Recorder", "Voice", "Piccolo",
    "StringInstrument", "Horn", 'Trumpet'
}
PIANO_LIKE = {
    "Piano", "Harp", "Harpsichord", "Organ", ""
}

INSTRUMENT_INDEX = {instrument : 0 if instrument in VIOLIN_LIKE else 0
                    for instrument in VIOLIN_LIKE | PIANO_LIKE}

def notes_to_events(notes):
    for ofs, dur, idx in notes:
        yield ofs, idx, 0
        yield ofs + dur - 1, idx, 1

def from_midi(fname, is_chamber_music):
    parsed_notes = parse_midi_notes(fname)
    normalized_notes = normalize_notes(parsed_notes)
    events = sorted(notes_to_events(normalized_notes))
    at = 0
    for ofs, idx, ev in events:
        wait = ofs - at
        if wait > 0:
            yield f'wait{wait}'
            at = ofs
        evname = 'p' if ev == 0 else 'endp'
        yield f'{evname}{idx}'
    yield 'wait2'

if __name__ == '__main__':
    fname = argv[1]
    events = list(from_midi(fname, False))
    print(' '.join(events))
