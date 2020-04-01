# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import defaultdict
from itertools import groupby
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
        'wait18', 'p19', 'wait2', 'endp19',
        'wait1', 'p21', 'wait2', 'endp21',
        'wait1', 'p7', 'p19', 'p23',
        'wait3', 'p14', 'wait2', 'endp14',
        'wait4', 'p14']
    assert events[-2:] == ['endp31', 'wait2']

def test_notewise_press_releases():
    path = Path() / 'tests' / 'jg_bps27.mid'
    rel_evs = list(notewise.midi_to_relative_events(path))
    assert len(rel_evs) == 11549
    ev_names = [ev for (ev, arg) in rel_evs]
    assert ev_names.count('press') == ev_names.count('release')

def test_chordwise():
    path = Path() / 'tests' / 'jsb2pin1.mid'
    rows = chordwise.from_midi(path, False)
    assert len(rows) == 1057
