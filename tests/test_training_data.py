# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.code_utils import INSN_END
from musicgen.corpus import IndexedModule, load_index, save_index
from musicgen.training_data import (TrainingData,
                                    flatten_training_data,
                                    load_and_encode_mod_files,
                                    load_training_data,
                                    pick_song_fragment,
                                    print_histogram)
from musicgen.utils import SP
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp

import numpy as np

TEST_PATH = Path() / 'tests' / 'mods'
TMP_DIR = Path('/tmp/cache_tmp')

def test_load_and_encode():
    SP.enabled = True
    mod_file = TEST_PATH / 'im_a_hedgehog.mod'
    encoder, arrs = load_and_encode_mod_files([mod_file], 'pcode_abs')
    end_idx = encoder.encode_char((INSN_END, 0), False)

    assert len(arrs[0][1]) == 5
    for arr in arrs[0][1]:
        assert arr[-1] == end_idx

def test_pcode_td():
    SP.enabled = True
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    assert len(td.arrs) == 1
    assert td.arrs[0][0] == 'im_a_hedgehog.mod'
    assert len(td.arrs[0][1]) == 5
    for code in td.arrs[0][1]:
        assert td.encoder.decode_char(code[-1]) == (INSN_END, 0)
        assert code[-1] == end_idx
        assert len(code) == 1497

def test_histogram():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    print_histogram(td)

def test_load_training_data():
    load_training_data('pcode_abs', TEST_PATH / 'im_a_hedgehog.mod')

def test_pick_song_fragment():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    enc = td.encoder
    seq = flatten_training_data(td)

    end_tok = enc.encode_char((INSN_END, 0), True)
    assert seq[-1] == end_tok
    i, fragment = pick_song_fragment(seq, 'random', 1200, end_tok)
    assert not end_tok in fragment
    assert len(fragment) == 1200

def test_transposing():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    assert not np.array_equal(td.arrs[0][1][0], td.arrs[0][1][1])

def maybe_build_index():
    if TMP_DIR.exists():
        return
    cat_dir = TMP_DIR / 'category'
    cat_dir.mkdir(exist_ok = True, parents = True)
    mods = [IndexedModule(p.name, i, 'MOD', 0, 4, 'category', 2000, 0,
                          0, 0)
            for i, p in enumerate(TEST_PATH.glob('*.mod'))]
    for mod in mods:
        src = TEST_PATH / mod.fname
        dst = cat_dir / mod.fname
        copyfile(src, dst)

    index = load_index(TMP_DIR)
    for mod in mods:
        index[mod.id] = mod
    save_index(TMP_DIR, index)

def test_code_types():
    SP.enabled = True
    maybe_build_index()
    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.arrs) == 31

    td = TrainingData('scode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.arrs) == 31

def test_scode_rel():
    SP.enabled = True
    maybe_build_index()
    td = TrainingData('scode_rel')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.arrs) == 31

def test_dcode():
    SP.enabled = True
    maybe_build_index()
    td = TrainingData('dcode')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.arrs) == 31

def test_pcode_rel():
    SP.enabled = True
    td = TrainingData('pcode_rel')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.arrs) == 31
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    for name, arrs in td.arrs:
        if name == 'beast2-ingame-st.mod':
            assert len(arrs) == 5
        elif name == 'satanic.mod':
            assert len(arrs) == 114
        elif name == 'entity.mod':
            assert len(arrs) == 2
        else:
            assert len(arrs) == 1
        for arr in arrs:
            assert arr[-1] == end_idx
