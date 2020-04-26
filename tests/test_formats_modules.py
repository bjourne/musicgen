from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_rows_to_string():
    mod = load_file(TEST_PATH / 'entity.mod')
    str = rows_to_string(mod.patterns[0].rows)
    assert len(str) == 64 * (10 * 4 + 7) - 1

    assert mod.patterns[0].rows[0][0].period == 509

def test_load_samples():
    mod = load_file(TEST_PATH / 'entity.mod')
    samples = load_samples(mod)
    assert samples[13].repeat_len == 0

def test_pattern_jump():
    mod = load_file(TEST_PATH / 'wax-rmx.mod')
    rows = linearize_rows(mod)
    assert len(rows) == 2640

def test_load_stk_module():
    mod = load_file(TEST_PATH / '3ddance.mod')
    assert mod.n_orders == 28

def test_protracker_15_sample_module():
    mod = load_file(TEST_PATH / 'am-fm_-_0ldsk00l_w1z4rd.mod')
    for i in range(15, 31):
        assert len(mod.samples[i].bytes) == 0
