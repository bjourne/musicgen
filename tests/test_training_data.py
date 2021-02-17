from pathlib import Path
from musicgen.training_data import (TrainingData,
                                    flatten_training_data,
                                    load_training_data,
                                    print_histogram)
from musicgen.utils import SP

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
