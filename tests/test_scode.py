# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from pathlib import Path
from musicgen.code_utils import (INSN_REL_PITCH,
                                 guess_percussive_instruments)
from musicgen.parser import load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.scode import metadata, to_code

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

    scode1 = list(to_code(notes, True, percussion, min_pitch))
    meta1 = metadata(scode1)

    rel_pitches = [c for (c, _) in scode1 if c == INSN_REL_PITCH]
    assert len(rel_pitches) > 0

    scode2 = list(to_code(notes, False, percussion, min_pitch))
    meta2 = metadata(scode2)
    assert meta1['pitch_range'] == meta2['pitch_range']
