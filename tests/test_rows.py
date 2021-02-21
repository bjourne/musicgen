# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.parser import load_file
from musicgen.rows import column_to_mod_notes, linearize_subsongs
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_weird_cells():
    mod = load_file(TEST_PATH / 'drive_faster.mod')
    volumes = [header.volume for header in mod.sample_headers]
    notes = column_to_mod_notes(mod.patterns[0].rows, 1, volumes)
    assert len(notes) == 32

def test_broken_mod():
    mod = load_file(TEST_PATH / 'operation_wolf-wolf31.mod')
    volumes = [header.volume for header in mod.sample_headers]
    notes = column_to_mod_notes(mod.patterns[1].rows, 3, volumes)

def test_subsongs():
    mod = load_file(TEST_PATH / 'beast2-ingame-st.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    orders = [o for (o, _) in subsongs]
    assert len(orders) == 6
    assert orders[0] == [0, 1, 2, 3, 1, 2, 3]
    assert orders[2] == [14]

    mod = load_file(TEST_PATH / 'satanic.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    orders = [o for (o, _) in subsongs]
    assert len(orders) == 114

    mod = load_file(TEST_PATH / 'entity.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    assert len(subsongs) == 2

def test_pattern_jump():
    mod = load_file(TEST_PATH / 'wax-rmx.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    assert len(subsongs) == 1
    assert len(subsongs[0][1]) == 2640
