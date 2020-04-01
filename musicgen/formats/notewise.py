# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Parser for the Notewise encoding.
from collections import defaultdict
from music21.duration import Duration
from music21.instrument import Piano, partitionByInstrument
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.midi.translate import midiFileToStream
from music21.note import Note
from music21.stream import Stream
from music21.tempo import MetronomeMark
from musicgen.formats import SAMPLE_FREQ, parse_midi_notes
from musicgen.formats.chordwise import NOTE_OFFSET, normalize_notes
from re import match
from sys import argv

def tokens_to_notes(tokens):
    offset = 0
    active_notes = defaultdict(list)
    notes = []

    def handle_wait(arg):
        return offset + arg / SAMPLE_FREQ
    def handle_press(arg):
        note = Note(arg + NOTE_OFFSET)
        note.offset = offset
        #note.storedInstrument = Piano()
        notes.append(note)
        active_notes[arg].append(note)

    def handle_release(arg):
        note = active_notes[arg].pop()
        note.duration = Duration(offset - note.offset)
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
    midi_els = [MetronomeMark('adagio', 60), TimeSignature('4/4')] + notes
    stream = Stream(midi_els)
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
        dur = max(1, dur)
        yield ofs, idx, 0
        yield ofs + dur - 1, idx, 1

def midi_to_relative_events(fname):
    parsed_notes = parse_midi_notes(fname)
    # Notes can overlap.
    normalized_notes = normalize_notes(parsed_notes)
    events = sorted(notes_to_events(normalized_notes))
    at = 0
    for ofs, idx, ev in events:
        wait = ofs - at
        if wait > 0:
            yield 'wait', wait
            at = ofs
        evname = 'press' if ev == 0 else 'release'
        yield evname, idx
    yield 'wait', 2


def from_midi(fname, is_chamber_music):
    name_map = {'press' : 'p',
                'release' : 'endp',
                'wait' : 'wait'}
    rel_evs = midi_to_relative_events(fname)
    for ev_name, arg in rel_evs:
        yield f'{name_map[ev_name]}{arg}'

if __name__ == '__main__':
    if len(argv) != 3 or argv[1] not in {'parse', 'generate'}:
        print(f'usage: python {argv[0]} [parse|generate] input')
        exit(1)
    fname = argv[2]
    if argv[1] == 'parse':
        events = list(from_midi(fname, False))
        print(' '.join(events))
    else:
        with open(fname, 'rt') as f:
            text = f.read()
        to_midi(text, 'out.mid')
