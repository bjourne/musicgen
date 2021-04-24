# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Two rcode tokes at once. Piggybacking on dcode module.
from musicgen import code_utils, dcode, rcode
from musicgen.code_utils import INSN_SILENCE

def to_code(notes, percussion):
    code = list(rcode.to_code(notes, percussion))
    if len(code) % 2 == 1:
        code.append((INSN_SILENCE, 1))
    return dcode.from_pcode(code)

def to_notes(code, row_time):
    return dcode.to_notes(code, row_time)

def is_transposable():
    return True

def code_transpositions(code):
    return dcode.code_transpositions(code)

def code_transpositions(code):
    return dcode.code_transpositions(code)

def estimate_row_time(code):
    return dcode.estimate_row_time(code)

def pause():
    return dcode.from_pcode(rcode.pause())

# Using pcode as the "default" format
def from_pcode(code):
    code = list(rcode.from_pcode(code))
    return dcode.from_pcode(code)

def to_pcode(code):
    return rcode.to_pcode(dcode.to_pcode(code))
