# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Two rcode tokes at once. Piggybacking on dcode module.
from musicgen import code_utils, rcode
from musicgen.code_utils import INSN_SILENCE
from musicgen.dcode import dcode_to_pcode, pcode_to_dcode

def to_code(notes, percussion):
    code = list(rcode.to_code(notes, percussion))
    if len(code) % 2 == 1:
        code.append((INSN_SILENCE, 1))
    return pcode_to_dcode(code)

def is_transposable():
    return True

def code_transpositions(code):
    code = list(dcode_to_pcode(code))
    codes = code_utils.code_transpositions(code)
    return [list(pcode_to_dcode(code)) for code in codes]

def code_transpositions(code):
    code = list(dcode_to_pcode(code))
    codes = code_utils.code_transpositions(code)
    return [list(pcode_to_dcode(code)) for code in codes]
