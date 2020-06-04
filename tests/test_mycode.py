from musicgen.mcode import INSN_JUMP, mod_notes_to_mcode
from musicgen.parser import load_file
from musicgen.rows import linearize_rows, column_to_mod_notes
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_initial_jumps_1():
    mod = load_file(TEST_PATH / '3_way-password.mod')
    rows = linearize_rows(mod)

    volumes = [64] * 32
    notes = column_to_mod_notes(rows, 3, volumes)
    _, seq = mod_notes_to_mcode(notes, {}, len(rows), False)

    n_jumps = len(rows) // 64
    assert seq == [(INSN_JUMP, 64)] * n_jumps

def test_initial_jumps_2():
    mod = load_file(TEST_PATH / 'x-tron.mod')
    rows = linearize_rows(mod)

    volumes = [64] * 32
    notes = column_to_mod_notes(rows, 3, volumes)
    first_pitch, seq = mod_notes_to_mcode(notes, {}, len(rows), False)

    silence_seq = [(INSN_JUMP, 64)] * 38 + [(INSN_JUMP, 32)]
    assert first_pitch is None
    assert len(rows) == 2464
    assert seq == silence_seq
