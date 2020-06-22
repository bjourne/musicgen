from musicgen.code_utils import (INSN_PITCH,
                                 INSN_REL_PITCH,
                                 INSN_SILENCE)
from musicgen.pcode import is_pcode_learnable

def test_abs_is_learnable():
    assert not is_pcode_learnable([])


    pcode = [(INSN_PITCH, 0), (INSN_SILENCE, 4),
            (INSN_PITCH, 0), (INSN_SILENCE, 4)] * 10 \
            + [(INSN_PITCH, 2), (INSN_SILENCE, 4)] * 10
    assert not is_pcode_learnable(pcode)

    pcode = [(INSN_PITCH, 0), (INSN_PITCH, 1),
             (INSN_PITCH, 2), (INSN_PITCH, 3),
             (INSN_PITCH, 4), (INSN_PITCH, 5)]
    assert not is_pcode_learnable(pcode)

    # Pitch range to large
    pcode = [(INSN_PITCH, i) for i in range(100)]
    assert not is_pcode_learnable(pcode)

def test_rel_is_learnable():
    pcode = [(INSN_REL_PITCH, 1), (INSN_REL_PITCH, -1)] * 20
    assert not is_pcode_learnable(pcode)

    pcode = [(INSN_REL_PITCH, 1), (INSN_REL_PITCH, -1)] * 10 \
        + [(INSN_REL_PITCH, 20), (INSN_REL_PITCH, 20)]
    assert not is_pcode_learnable(pcode)
