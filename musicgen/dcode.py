# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# DCode stands for double parallel code.
from musicgen import code_utils, pcode
from musicgen.code_utils import INSN_SILENCE
from musicgen.utils import SP

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

def to_code(mod, percussion, min_pitch):
    code = list(pcode.to_code(mod, False, percussion, min_pitch))
    if len(code) % 2 == 1:
        code.append((INSN_SILENCE, 1))
    return pcode_to_dcode(code)

def to_notes(code):
    return pcode.to_notes(dcode_to_pcode(code), False)

def metadata(code):
    return pcode.metadata(list(dcode_to_pcode(code)))

def is_transposable():
    return True

def transpose_code(code):
    code = list(dcode_to_pcode(code))
    codes = code_utils.transpose_code(code)
    return [list(pcode_to_dcode(code)) for code in codes]

def test_encode_decode(mod_file):
    from musicgen.code_utils import CODE_MIDI_MAPPING
    from musicgen.generation import notes_to_midi_file
    from musicgen.parser import load_file
    mod = load_file(mod_file)
    code = to_code(mod)
    notes = to_notes(code)
    notes_to_midi_file(notes, 'test.mid', CODE_MIDI_MAPPING)

if __name__ == '__main__':
    from sys import argv
    SP.enabled = True
    test_encode_decode(argv[1])
