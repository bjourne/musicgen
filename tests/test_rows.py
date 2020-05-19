from musicgen.parser import load_file
from musicgen.rows import column_to_mod_notes
from pathlib import Path

TEST_PATH = Path() / 'tests' / 'mods'

def test_weird_cells():
    mod = load_file(TEST_PATH / 'drive_faster.mod')
    volumes = [header.volume for header in mod.sample_headers]
    notes = column_to_mod_notes(mod.patterns[0].rows, 1, volumes)
    assert len(notes) == 32
