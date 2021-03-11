# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# DCode stands for double parallel code.
from musicgen import code_utils, pcode
from musicgen.code_utils import INSN_END, INSN_SILENCE
from musicgen.utils import SP

def pause():
    return pcode_to_dcode(pcode.pause())

def dcode_to_pcode(code):
    for cmd, arg in code:
        cmd1, cmd2 = cmd[0], cmd[-1]
        arg1, arg2 = arg
        yield cmd1, arg1
        yield cmd2, arg2

def pcode_to_dcode(code):
    assert len(code) % 2 == 0
    for ((cmd1, arg1), (cmd2, arg2)) in zip(code[::2], code[1::2]):
        yield cmd1 + '-' + cmd2, (arg1, arg2)

def to_code(mod, percussion):
    code = list(pcode.to_code(mod, False, percussion))
    if len(code) % 2 == 1:
        code.append((INSN_SILENCE, 1))
    return pcode_to_dcode(code)

def to_notes(code, row_time):
    return pcode.to_notes(list(dcode_to_pcode(code)), False, row_time)

def is_transposable():
    return True

def code_transpositions(code):
    code = list(dcode_to_pcode(code))
    codes = code_utils.code_transpositions(code)
    return [list(pcode_to_dcode(code)) for code in codes]

def normalize_pitches(code):
    code = list(dcode_to_pcode(code))
    code = code_utils.normalize_pitches(code)
    return list(pcode_to_dcode(code))

def estimate_row_time(code):
    code = list(dcode_to_pcode(code))
    return pcode.estimate_row_time(code, False)
