from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_rows_to_string():
    mod = load(TEST_PATH / 'entity.mod')
    str = rows_to_string(mod.patterns[0].rows)
    assert len(str) == 64 * (10 * 4 + 7) - 1

    assert mod.patterns[0].rows[0][0].period == 509

def test_load_samples():
    mod = load(TEST_PATH / 'entity.mod')
    samples = load_samples(mod)
    assert samples[13].repeat_len == 0
