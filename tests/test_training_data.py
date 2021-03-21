# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.code_generators import get_code_generator
from musicgen.code_utils import INSN_END, INSN_PITCH
from musicgen.corpus import IndexedModule, load_index, save_index
from musicgen.training_data import (CODE_MODULES,
                                    ERR_FEW_UNIQUE_PITCHES,
                                    ERR_PARSE_ERROR,
                                    TrainingData,
                                    abs_ofs_to_rel_ofs,
                                    convert_to_midi,
                                    load_and_encode_mod_files,
                                    load_training_data,
                                    mod_file_to_codes_w_progress,
                                    print_histogram,
                                    random_rel_ofs,
                                    save_generated_sequences)
from musicgen.utils import SP
from pathlib import Path
from shutil import copyfile, rmtree
from tempfile import mkdtemp

import numpy as np

TEST_PATH = Path() / 'tests' / 'mods'
TMP_DIR = Path('/tmp/cache_tmp')

SP.enabled = True

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

def test_load_and_encode():
    # This song is perhaps incorrectly classified as dissonant.
    mod_file = TEST_PATH / 'im_a_hedgehog.mod'
    encoder, songs = load_and_encode_mod_files(
        [mod_file], 'pcode_abs')
    assert len(songs) == 0

def test_pcode_td():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')
    # end_idx = td.encoder.encode_char((INSN_END, 0), False)

    assert len(td.songs) == 1
    assert td.songs[0][0] == 'zodiak_-_gasp.mod'

    first_subsong = td.songs[0][1][0]
    assert sum(transp.shape[0] for transp in first_subsong) == 13423 * 5

    code = td.encoder.decode_chars(first_subsong[0][:120])
    code = [(c, a) for (c, a) in code if c == INSN_PITCH]
    assert code == [('P', 24), ('P', 24),
                    ('P', 27), ('P', 27),
                    ('P', 24)]

    transp0 = first_subsong[0]
    n_code = len(transp0)
    for transp in first_subsong[1:]:
        assert len(transp) == n_code
        assert not np.array_equal(transp, transp0)

def test_histogram():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    print_histogram(td)

def test_load_training_data():
    load_training_data('pcode_abs', TEST_PATH / 'im_a_hedgehog.mod')

def test_pcode_abs():
    maybe_build_index()
    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.songs) == 26

def test_dcode():
    maybe_build_index()
    td = TrainingData('dcode')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.songs) == 26

def test_pcode_rel():
    maybe_build_index()
    td = TrainingData('pcode_rel')
    td.load_disk_cache(TMP_DIR, 150)
    assert len(td.songs) == 26
    # end_idx = td.encoder.encode_char((INSN_END, 0), False)

    subsongs_per_song = {
        'beast2-ingame-st.mod' : 5,
        'entity.mod' : 2
    }
    for name, subsongs in td.songs:
        n_subsongs = subsongs_per_song.get(name, 1)
        assert len(subsongs) == n_subsongs

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

    assert len(train.songs) + len(valid.songs) + len(test.songs) \
        == len(td.songs)
    assert train.songs[0][1][0][0].flags['OWNDATA']
    assert not train.songs[0][1][0][0].base

def test_packed_module():
    file_path = TEST_PATH / 'mr_kadde_-_con-vers-cert.mod'
    result = list(mod_file_to_codes_w_progress(1, 1,
                                               file_path, 'pcode_abs'))
    assert len(result) == 1
    assert result[0] == (False, 0, (ERR_PARSE_ERROR, 'PowerPackerModule'))

def test_convert_to_midi():
    file_path = TEST_PATH / 'zodiak_-_gasp.mod'
    for code_type in CODE_MODULES:
        convert_to_midi(code_type, file_path)

def test_abs_to_rel_ofs():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')
    rel_ofs = abs_ofs_to_rel_ofs(td, 36000)
    assert rel_ofs == (0, 0, 2, 9154)

def test_random_rel_ofs():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'zodiak_-_gasp.mod')
    for _ in range(10):
        s_i, ss_i, t_i, o = random_rel_ofs(td, 10_000)
        transp = td.songs[s_i][ss_i][t_i]
        assert len(transp) < o + 10_000

def test_numpy_format():
    maybe_build_index()
    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)

    assert td.songs[0][1][0][0].flags['OWNDATA'] == True
    assert not td.songs[0][1][0][0].base

def test_save_generated_sequences():
    g = get_code_generator('orig-pcode')

    output_path = TMP_DIR / 'generated'
    if output_path.exists():
        rmtree(output_path)
    output_path.mkdir(parents = True)

    td = TrainingData('pcode_abs')
    td.load_disk_cache(TMP_DIR, 150)
    offsets = [random_rel_ofs(td, 100)]

    seqs = [np.array([1, 2, 3, 4, 5])]
    log_probs = [-100]
    skews = [('top-p', 0.98)]
    save_generated_sequences(g, output_path, td, seqs, offsets,
                             log_probs, skews)

    files = list(output_path.glob('*.pickle.gz'))
    assert len(files) == 1
    parts = files[0].stem.split('-')
    rel_ofs = [int(p) for p in parts[:4]]
    for p in rel_ofs:
        assert p >= 0
