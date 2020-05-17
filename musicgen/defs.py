# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from bisect import bisect_left, bisect_right
from collections import namedtuple
from musicgen.utils import SP

PERIODS = [
    1712, 1616, 1525, 1440, 1357, 1281, 1209, 1141, 1077, 1017, 961, 907,
    856,   808,  762,  720,  678,  640,  604,  570,  538,  508, 480, 453,
    428,   404,  381,  360,  339,  320,  302,  285,  269,  254, 240, 226,
    214,   202,  190,  180,  170,  160,  151,  143,  135,  127, 120, 113,
    107,   101,   95,   90,   85,   80,   76,   71,   67,   64,  60,  57,
]
PERIOD_TO_IDX = {p : i for i, p in enumerate(PERIODS)}

def period_to_idx(period):
    # This is a little hacky. Some cells have incorrect period values.
    idx = PERIOD_TO_IDX.get(period)
    if idx is None:
        rev_periods = list(reversed(PERIODS))
        idx = bisect_left(rev_periods, period)

        if idx > 0:
            pass
        else:
            idx = 1
        idx = 60 - idx
    assert idx is not None
    return idx

NOTE_FREQS = [
    4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65,
    5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13
]

SAMPLE_RATE = 44100
AMIGA_SAMPLE_RATE = 16574.27

DEFAULT_TEMPO = 125
DEFAULT_SPEED = 0x06
DEFAULT_VOLUME = 0x20

TRACK_VOLUME = 0x20
MASTER_VOLUME = 0x40

def freq_for_idx(idx):
    octave = idx // 12
    note = idx % 12
    return NOTE_FREQS[note]/2**(8-octave)

FREQS = [freq_for_idx(idx) for idx in range(len(PERIODS))]
BASE_NOTE_IDX = 36
BASE_FREQ = freq_for_idx(BASE_NOTE_IDX)

# Enum for effect commands. Python's builtin Enum class doesn't play
# well with Construct, hence why they are constants.
EFFECT_CMD_JUMP_TO_OFFSET = 11
EFFECT_CMD_SET_VOLUME     = 12
EFFECT_CMD_JUMP_TO_ROW    = 13
EFFECT_CMD_FINETUNING     = 14
EFFECT_CMD_UPDATE_TIMING  = 15


Note = namedtuple('Note', ['col_idx', 'row_idx',
                           'sample_idx', 'note_idx',
                           'vol', 'time_ms'])
