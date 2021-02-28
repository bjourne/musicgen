# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.code_utils import INSN_PITCH, INSN_REL_PITCH, INSN_SILENCE
from musicgen.pcode import to_code, to_notes
from musicgen.rows import ModNote
from musicgen.training_data import (ERR_FEW_NOTES,
                                    ERR_PITCH_RANGE,
                                    training_error)

def error_for_pcode(pcode, rel_pitches):
    notes = to_notes(pcode, rel_pitches)
    return training_error(notes, {})

def test_abs_is_learnable():
    err = error_for_pcode([], False)
    assert err == (ERR_FEW_NOTES, 0)

    pcode = [(INSN_PITCH, 0), (INSN_SILENCE, 4),
            (INSN_PITCH, 0), (INSN_SILENCE, 4)] * 10 \
            + [(INSN_PITCH, 2), (INSN_SILENCE, 4)] * 10
    err = error_for_pcode(pcode, False)
    assert err == (ERR_FEW_NOTES, 30)

    pcode = [(INSN_PITCH, 0), (INSN_PITCH, 1),
             (INSN_PITCH, 2), (INSN_PITCH, 3),
             (INSN_PITCH, 4), (INSN_PITCH, 5)]
    err = error_for_pcode(pcode, False)
    assert err == (ERR_FEW_NOTES, 6)

    # Pitch range to large
    pcode = [(INSN_PITCH, i) for i in range(100)]
    err = error_for_pcode(pcode, False)
    assert err == (ERR_PITCH_RANGE, 99)

def test_rel_is_learnable():
    pcode = [(INSN_REL_PITCH, 1), (INSN_REL_PITCH, -1)] * 20
    err = error_for_pcode(pcode, True)
    assert err

    pcode = [(INSN_REL_PITCH, 1), (INSN_REL_PITCH, -1)] * 10 \
        + [(INSN_REL_PITCH, 20), (INSN_REL_PITCH, 20)]
    err = error_for_pcode(pcode, True)
    assert err

def test_to_code():
    # Don't convert pitches
    notes = [ModNote(0, 0, 1, 10, 1.0, 0.0),
             ModNote(0, 1, 1, 50, 1.0, 0.0)]
    code = list(to_code(notes, False, {}, 0))
    assert code == [(INSN_PITCH, 10), (INSN_PITCH, 50)]
