# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.code_utils import INSN_END
from musicgen.training_data import (TrainingData,
                                    flatten_training_data,
                                    load_training_data,
                                    pick_song_fragment,
                                    print_histogram)
from musicgen.utils import SP
from pathlib import Path
import numpy as np


TEST_PATH = Path() / 'tests' / 'mods'

def test_pcode_td():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    assert len(td.arrs) == 1
    assert td.arrs[0][0] == 'im_a_hedgehog.mod'
    assert len(td.arrs[0][1]) == 5
    for code in td.arrs[0][1]:
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
    i, fragment = pick_song_fragment(seq, 'random', 1200, end_tok)
    assert not end_tok in fragment
    assert len(fragment) == 1200

def test_transposing():
    td = TrainingData('pcode_abs')
    td.load_mod_file(TEST_PATH / 'im_a_hedgehog.mod')
    assert not np.array_equal(td.arrs[0][1][0], td.arrs[0][1][1])
