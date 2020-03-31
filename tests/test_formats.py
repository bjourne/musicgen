# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.formats import parse_midi_notes
from musicgen.formats import chordwise
from musicgen.formats import notewise
from os import getcwd
from pathlib import Path

TEST_PATH = Path() / 'tests'

def test_parse_midi_notes():
    path = TEST_PATH / 'jg_bps27.mid'
    notes = parse_midi_notes(path)
    assert len(notes) == len(set(notes))

def test_parse_notwise():
    path = Path() / 'tests' / 'bwv807a.txt'
    with open(path) as f:
        text = f.read()
    tokens = text.split()
    notes = notewise.tokens_to_notes(tokens)
    assert len(notes) == 36854

def test_generate_notewise():
    path = Path() / 'tests' / 'jg_bps27.mid'
    events = list(notewise.from_midi(path, False))
    assert events[:18] == [
        'wait6', 'p19', 'endp19',
        'wait1', 'p21', 'endp21',
        'wait1', 'p7', 'p19', 'p23',
        'wait1', 'p14', 'endp14',
        'wait2', 'endp7', 'p14', 'endp14', 'endp19'
        ]
    assert events[-2:] == ['endp31', 'wait2']


def test_chordwise():
    path = Path() / 'tests' / 'jsb2pin1.mid'
    rows = chordwise.from_midi(str(path), False)
    assert len(rows) == 353
