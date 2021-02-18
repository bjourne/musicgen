from pathlib import Path
from musicgen.code_utils import INSN_REL_PITCH
from musicgen.parser import load_file
from musicgen.scode import metadata, to_code

TEST_PATH = Path() / 'tests' / 'mods'

def test_pitch_range():
    mod = load_file(TEST_PATH / 'im_a_hedgehog.mod')
    scode1 = list(to_code(mod, True, True))
    meta1 = metadata(scode1)

    rel_pitches = [c for (c, _) in scode1 if c == INSN_REL_PITCH]
    assert len(rel_pitches) > 0

    scode2 = list(to_code(mod, False, True))
    meta2 = metadata(scode2)
    assert meta1['pitch_range'] == meta2['pitch_range']
