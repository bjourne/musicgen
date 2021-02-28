# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.code_utils import INSN_END, INSN_PITCH
from musicgen.corpus import IndexedModule, load_index, save_index
from musicgen.training_data import (ERR_FEW_UNIQUE_PITCHES,
                                    ERR_PARSE_ERROR,
                                    TrainingData,
                                    load_and_encode_mod_files,
                                    load_training_data,
                                    mod_file_to_codes_w_progress,
                                    pick_song_fragment,
                                    print_histogram)
from musicgen.utils import SP
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp

import numpy as np

TEST_PATH = Path() / 'tests' / 'mods'
TMP_DIR = Path('/tmp/cache_tmp')

SP.enabled = True

def test_load_and_encode():
    # This song is perhaps incorrectly classified as dissonant.
    mod_file = TEST_PATH / 'im_a_hedgehog.mod'
    encoder, meta, data = load_and_encode_mod_files(
        [mod_file], 'pcode_abs')
    assert len(data) == 0

def test_pcode_td():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')
    end_idx = td.encoder.encode_char((INSN_END, 0), False)

    assert len(td.meta) == 1
    assert td.meta[0][1] == 'zodiak_-_gasp.mod'
    assert td.data.tolist().count(end_idx) == 5
    assert len(td.data) == 13424 * 5

    code = td.encoder.decode_chars(td.data[:120])
    code = [(c, a) for (c, a) in code if c == INSN_PITCH]
    assert code == [('P', 24), ('P', 24),
                    ('P', 27), ('P', 27),
                    ('P', 24)]


def test_histogram():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    print_histogram(td)

def test_load_training_data():
    load_training_data('pcode_abs', TEST_PATH / 'im_a_hedgehog.mod')

def test_pick_song_fragment():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')

    end_tok = td.encoder.encode_char((INSN_END, 0), True)
    assert td.data[-1] == end_tok
    i, fragment = pick_song_fragment(td, 'random', 1200, False)
    assert not end_tok in fragment
    assert len(fragment) == 1200

def test_transposing():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    indexes = np.where(td.data == end_idx)[0] + 1
    arrs = np.split(td.data, indexes)[:-1]
    for arr in arrs[1:]:
        assert len(arr) == len(arrs[0])
        assert not np.array_equal(arr, arr[0])

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
    maybe_build_index()
    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.meta) == 26

    td = TrainingData('scode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.meta) == 26

def test_scode_rel():
    maybe_build_index()
    td = TrainingData('scode_rel')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.meta) == 26

def test_dcode():
    maybe_build_index()
    td = TrainingData('dcode')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.meta) == 26

def test_pcode_rel():
    maybe_build_index()
    td = TrainingData('pcode_rel')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.meta) == 26
    end_idx = td.encoder.encode_char((INSN_END, 0), False)

    subsongs_per_song = {
        'beast2-ingame-st.mod' : 5,
        'entity.mod' : 2
    }
    sentinel = (None, len(td.data))
    for ((o1, n1), (o2, _)) in zip(td.meta, td.meta[1:] + [sentinel]):
        chunk = td.data[o1:o2]
        n_subsongs = subsongs_per_song.get(n1, 1)
        assert chunk.tolist().count(end_idx) == n_subsongs
        assert chunk[-1] == end_idx

def test_is_learnable():
    file_path = TEST_PATH / 'youve_been_here.mod'
    result = list(mod_file_to_codes_w_progress(1, 1,
                                               file_path, 'pcode_abs'))
    assert len(result) == 1
    assert result[0] == (False, 0, (ERR_FEW_UNIQUE_PITCHES, 2))

def test_split():
    maybe_build_index()
    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    train, valid, test = td.split_3way(0.8, 0.1)
    assert len(train.data) + len(valid.data) + len(test.data) \
        == len(td.data)
    assert len(train.meta) + len(valid.meta) + len(test.meta) \
        == len(td.meta)

def test_packed_module():
    file_path = TEST_PATH / 'mr_kadde_-_con-vers-cert.mod'
    result = list(mod_file_to_codes_w_progress(1, 1,
                                               file_path, 'pcode_abs'))
    assert len(result) == 1
    assert result[0] == (False, 0, (ERR_PARSE_ERROR, 'PowerPackerModule'))
