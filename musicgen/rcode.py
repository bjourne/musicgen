# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Exactly like pcode_abs except that silence is not compressed.
from musicgen import pcode, code_utils
from musicgen.code_utils import INSN_SILENCE

def to_code(notes, percussion):
    for cmd, arg in pcode.to_code(notes, False, percussion):
        if cmd == INSN_SILENCE:
            for _ in range(arg):
                yield INSN_SILENCE, 1
        else:
            yield cmd, arg

def estimate_row_time(code):
    return pcode.estimate_row_time(code, False)

def to_notes(code, row_time):
    return pcode.to_notes(code, False, row_time)

def is_transposable():
    return True

def code_transpositions(code):
    return code_utils.code_transpositions(code)

def normalize_pitches(code):
    return code_utils.normalize_pitches(code)

def pause():
    return [(INSN_SILENCE, 1)] * 4 * 16
