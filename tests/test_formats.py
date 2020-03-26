# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.formats.notewise import *
from os import getcwd
from pathlib import Path

def test_notewise():
    path = Path() / 'tests' / 'bwv807a.txt'
    with open(path) as f:
        text = f.read()
    tokens = text.split()
    notes = tokens_to_notes(tokens)
    assert len(notes) == 36854
