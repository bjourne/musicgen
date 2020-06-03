from musicgen.analyze import sample_props
from musicgen.defs import PERIODS, period_to_idx
from musicgen.parser import load_file
from musicgen.prettyprint import rows_to_string
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.samples import load_samples
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

def test_weird_magic():
    # This mod has the signature "M&K!"
    mod = load_file(TEST_PATH / 'im_a_hedgehog.mod')
    assert mod.n_orders == 13

def test_loading_truncated_module():
    mod = load_file(TEST_PATH / 'after-the-rain.mod')
    assert len(mod.samples[8].bytes) == 7990

def percussive_samples(mod):
    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = list(rows_to_mod_notes(rows, volumes))
    return {sample for (sample, p) in sample_props(mod, notes).items()
            if p.is_percussive}

def test_sample_length():
    mod = load_file(TEST_PATH / 'his_hirsute_ant.mod')
    assert len(mod.samples[0].bytes) == 0x23ca

# Testing heuristics for detecting percussive samples.
def test_percussive_samples():
    mod = load_file(TEST_PATH / 'androidr.mod')
    assert percussive_samples(mod) == {2, 3, 4}

    mod = load_file(TEST_PATH / 'big_blunts.mod')
    assert percussive_samples(mod) == {17, 21}

    mod = load_file(TEST_PATH / 'boner.mod')
    assert percussive_samples(mod) == {1, 2, 3, 6}

    mod = load_file(TEST_PATH / 'lambada.mod')
    assert percussive_samples(mod) == {2, 4, 5, 6}

    mod = load_file(TEST_PATH / 'mist-eek.mod')
    assert percussive_samples(mod) == {10, 11, 12}

    # 1 and 2 are chord samples incorrectly classified as drums.
    mod = load_file(TEST_PATH / 'afro_afro.mod')
    assert percussive_samples(mod) == {1, 2, 5, 6, 7, 8}

def test_period_to_idx():
    idx = period_to_idx(679)
    assert PERIODS[idx] == 678

    idx = period_to_idx(56)
    assert PERIODS[idx] == 57

def test_loading_tricky_mods():
    mod = load_file(TEST_PATH / 'pachabel.mod')
    assert len(mod.samples[2].bytes) == 0x1ee1c
