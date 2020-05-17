# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from bisect import bisect_left, bisect_right
from collections import namedtuple
from musicgen.utils import SP
import numpy as np

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

NOTES = [
    "C-", "C#", "D-", "D#", "E-", "F-",
    "F#", "G-", "G#", "A-", "A#", "B-"
]
NOTE_TO_IDX = {n : i for i, n in enumerate(NOTES)}

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

def freq_for_index(idx):
    octave = idx // 12
    note = idx % 12
    return NOTE_FREQS[note]/2**(8-octave)

def notestr_to_index(notestr):
    note = NOTE_TO_IDX[notestr[:2]]
    octave = int(notestr[2])
    return 12 * octave + note

def index_to_notestr(idx):
    octave = idx // 12
    note = idx % 12
    return NOTES[note] + str(octave)

FREQS = [freq_for_index(idx) for idx in range(len(PERIODS))]
BASE_NOTE_IDX = notestr_to_index('C-3')
BASE_FREQ = freq_for_index(BASE_NOTE_IDX)

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


########################################################################
# Sample management
########################################################################
class Sample:
    def __init__(self, bytes, repeat_from, repeat_len):
        arr = np.frombuffer(bytes, dtype = np.int8)
        arr = arr.astype(np.int16) * 256

        ratio = SAMPLE_RATE / AMIGA_SAMPLE_RATE
        n_samples = int(arr.size * ratio)
        if arr.size:
            x_old = np.linspace(0, 1, arr.size)
            x_new = np.linspace(0, 1, n_samples)
            arr = np.interp(x_new, x_old, arr)
        self.arr = arr.astype(np.float)
        if repeat_len > 1:
            self.repeat_from = round(2 * repeat_from * ratio)
            self.repeat_len = round(2 * repeat_len * ratio)
        else:
            self.repeat_from = self.repeat_len = 0

def load_samples(mod):
    return [Sample(data.bytes, header.repeat_from, header.repeat_len)
            for data, header in zip(mod.samples, mod.sample_headers)]

########################################################################
# Mathy stuff
########################################################################
def repeat_sample(sample, arr, dur_s):
    '''
    arr is the frequency interpolated sample array.
    '''
    # Duration in nr of samples if repeating
    n_samples = int(SAMPLE_RATE * dur_s)

    # Repeating
    repeat_from = sample.repeat_from
    repeat_len = sample.repeat_len

    ratio = arr.size / sample.arr.size
    repeat_from = int(repeat_from * ratio)
    repeat_len = int(repeat_len * ratio)

    if repeat_len:
        repeat_to = repeat_from + repeat_len
        n_tail = arr.size - repeat_to
        repeat_body_len = n_samples - arr.size + repeat_len
        if repeat_body_len > repeat_len:
            n_repeats = int(repeat_body_len / repeat_len)
            head = arr[:repeat_from]
            tail = arr[repeat_to:]
            body = arr[repeat_from:repeat_to]
            repeated_body = np.tile(body, n_repeats)
            arr = np.concatenate((head, repeated_body, tail))
    return arr
