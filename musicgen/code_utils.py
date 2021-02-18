# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
#
# Utils common to all encodings.
from musicgen.analyze import sample_props

# Fun melodic instruments:
#   20 (organ)
#   27 (jazz guitar)
#  105 (sitar)
CODE_MIDI_MAPPING = {
    1 : [-1, 36, 4, 1.0],
    2 : [-1, 40, 4, 1.0],
    3 : [-1, 31, 4, 1.0],
    4 : [1, 48, 4, 1.0]
    #4 : [1, 54, 3, 1.0]
}
BASE_ROW_TIME = 160

# Standard instructions
INSN_PITCH = 'P'
INSN_REL_PITCH = 'R'
INSN_SILENCE = 'S'
INSN_DRUM = 'D'
INSN_END = 'X'

def insn_to_string(insn):
    cmd, arg = insn
    if cmd == INSN_PITCH:
        return '%02d' % arg
    return '%s%s' % (cmd, arg)

def code_to_string(pcode):
    return ' '.join(insn_to_string(insn) for insn in pcode)

def guess_percussive_instruments(mod, notes):
    props = sample_props(mod, notes)
    props = [(s, p.n_notes, p.is_percussive) for (s, p) in props.items()
             if p.is_percussive]

    # Sort by the number of notes so that the same instrument
    # assignment is generated every time.
    props = list(reversed(sorted(props, key = lambda x: x[1])))
    percussive_samples = [s for (s, _, _) in props]

    return {s : i % 3 for i, s in enumerate(percussive_samples)}

def guess_initial_pitch(scode):
    diffs = [arg for (cmd, arg) in scode if cmd == INSN_REL_PITCH]
    at_pitch, max_pitch, min_pitch = 0, 0, 0
    for diff in diffs:
        at_pitch += diff
        max_pitch = max(at_pitch, max_pitch)
        min_pitch = min(at_pitch, min_pitch)
    return -min_pitch

def fix_durations(notes):
    '''
    Notes must be in the same column.
    '''
    for n1, n2 in zip(notes, notes[1:]):
        n1.duration = min(n2.row_idx - n1.row_idx, 16)
    if notes:
        last_note = notes[-1]
        row_in_page = last_note.row_idx % 64
        last_note.duration = min(64 - row_in_page, 16)
