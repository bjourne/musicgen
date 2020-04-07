from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_rows_to_string():
    mod = load(TEST_PATH / 'entity.mod')
    str = rows_to_string(mod.patterns[0].rows)
