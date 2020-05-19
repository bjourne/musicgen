from musicgen.mycode import INSN_JUMP, column_to_mycode
from musicgen.parser import load_file
from musicgen.rows import linearize_rows, column_to_mod_notes
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_initial_jumps():
    mod = load_file(TEST_PATH / '3_way-password.mod')
    rows = linearize_rows(mod)
    seq = list(column_to_mycode(rows, 3))
    n_jumps = len(rows) // 64
    assert seq == [(INSN_JUMP, 64)] * n_jumps
