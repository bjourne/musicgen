# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Exactly like pcode_abs except that silence is not compressed.
from itertools import groupby
from musicgen import pcode, code_utils
from musicgen.code_utils import INSN_SILENCE

def from_pcode(code):
    for cmd, arg in code:
        if cmd == INSN_SILENCE:
            for _ in range(arg):
                yield INSN_SILENCE, 1
        else:
            yield cmd, arg

def to_pcode(code):
    def produce_silence(delta):
        thresholds = [16, 8, 4, 3, 2, 1]
        for threshold in thresholds:
            while delta >= threshold:
                yield threshold
                delta -= threshold
        assert delta >= -1
    for grp, els in groupby(code, lambda x: x):
        if grp == (INSN_SILENCE, 1):
            for sil in produce_silence(len(list(els))):
                yield INSN_SILENCE, sil
        else:
            for el in els:
                yield el

def to_code(notes, percussion):
    return from_pcode(pcode.to_code(notes, False, percussion))

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
