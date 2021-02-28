# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from pathlib import Path
from musicgen.code_utils import (INSN_REL_PITCH,
                                 guess_percussive_instruments)
from musicgen.parser import load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.scode import to_code, to_notes

TEST_PATH = Path() / 'tests' / 'mods'

def test_pitch_range():
    mod = load_file(TEST_PATH / 'im_a_hedgehog.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    rows = subsongs[0][1]
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    percussion = guess_percussive_instruments(mod, notes)
    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}
    min_pitch = min(pitches, default = 0)
    for note in notes:
        note.pitch_idx -= min_pitch

    scode1 = list(to_code(notes, True, percussion, min_pitch))
    notes1 = to_notes(scode1, True)

    scode2 = list(to_code(notes, False, percussion, min_pitch))
    notes2 = to_notes(scode2, False)

    pitches1 = {n.pitch_idx for n in notes1}
    pitches2 = {n.pitch_idx for n in notes2}
    assert pitches1 == pitches2

def test_pitch_hanges():
    mod = load_file(TEST_PATH / 'im_a_hedgehog.mod')
    subsongs = list(linearize_subsongs(mod, 1))
    rows = subsongs[0][1]
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)

    pitches_before = [n.pitch_idx for n in notes]

    scode1 = list(to_code(notes, True, {}, 33))
    pitches_after = [n.pitch_idx for n in notes]
    assert pitches_before == pitches_after
