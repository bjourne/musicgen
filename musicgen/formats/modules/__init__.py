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

def period_to_string(period):
    if period == 0:
        return '...'
    idx = period_to_idx(period)
    octave_idx = idx // 12
    note_idx = idx % 12
    note = NOTES[note_idx]
    return '%s%d' % (note, octave_idx)

def cell_to_string(cell):
    note = period_to_string(cell.period)
    effect_val = (cell.effect_cmd << 8) \
                 + (cell.effect_arg1 << 4) + (cell.effect_arg2)
    sample_idx = cell.sample_idx
    sample_str = '%02X' % sample_idx if sample_idx else '..'
    effect_str = '%03X' % effect_val if effect_val else '...'
    return '%s %s %s' % (note, sample_str, effect_str)

def row_to_string(row):
    return '  '.join(map(cell_to_string, row))

def rows_to_string(rows, numbering = False):
    strings = [row_to_string(row) for row in rows]
    if numbering:
        strings = ['#%04d %s' % (i, row) for i, row in enumerate(strings)]
    return '\n'.join(strings)

def pattern_to_string(pattern):
    return '\n'.join(map(row_to_string, pattern.rows))

def calc_row_time(tempo, speed):
    return (60 / tempo / 4) * speed / 6

def update_timings(row, tempo, speed):
    for cell in row:
        if cell.effect_cmd == EFFECT_CMD_UPDATE_TIMING:
            val = 16 * cell.effect_arg1 + cell.effect_arg2
            if val <= 0x1f:
                speed = val
            else:
                tempo = val
    return tempo, speed

########################################################################
# Data extraction
########################################################################
def linearize_rows(mod):
    table_idx = 0
    rows = []
    jumps_taken = set()
    next_from = 0
    while table_idx < mod.n_orders:
        pattern_idx = mod.pattern_table[table_idx]
        pattern = mod.patterns[pattern_idx]
        assert len(pattern.rows) == 64
        for i in range(next_from, 64):
            row = pattern.rows[i]
            rows.append(row)
            jump = False
            jump_loc = table_idx, i
            for cell in row:
                cmd = cell.effect_cmd
                arg1 = cell.effect_arg1
                arg2 = cell.effect_arg2
                if cmd == 0xb and not jump_loc in jumps_taken:
                        table_idx = 16 * arg1 + arg2 - 1
                        jumps_taken.add(jump_loc)
                        next_from = 0
                        jump = True
                elif cmd == 0xd:
                    next_from = 10 * arg1 + arg2
                    jump = True
            if jump:
                break
            else:
                next_from = 0
        table_idx += 1
    return rows

def mod_note_volume(mod, cell):
    if cell.effect_cmd == 12:
        return (cell.effect_arg1 << 4) + cell.effect_arg2
    return mod.sample_headers[cell.sample_idx - 1].volume

Note = namedtuple('Note', ['col_idx', 'row_idx',
                           'sample_idx', 'note_idx',
                           'vol', 'time_ms'])

# If the sample is present but not the period, it is ambiguous whether
# the cell represents repeating the note or bending a ringing
# note. For most but not all MODs, the MIDI representation becomes
# better if such cells are assumed to be bends.
#
# Worse : blu_angel_-_dream.mod
# Better: agnostic.mod
def notes_in_rows(mod, rows):
    '''
    Order is (col, row, sample, note, vol, ms)
    '''
    tempo = DEFAULT_TEMPO
    speed = DEFAULT_SPEED
    channel_periods = {}
    channel_samples = {}
    for row_idx, row in enumerate(rows):
        tempo, speed = update_timings(row, tempo, speed)
        time_ms = int(calc_row_time(tempo, speed) * 1000)
        for col_idx, cell in enumerate(row):
            # Period may be missing.
            sample_idx = cell.sample_idx
            period = cell.period
            # Neither sample nor note, skipping
            if not sample_idx and not period:
                continue

            # Corrupt mod with bad sample_idx
            if not sample_idx <= 0x1f:
                continue

            # Sample but no note
            if sample_idx and not period:
                fmt = 'Missing period for sample %2d at cell %4d:%d.'
                SP.print(fmt % (sample_idx, row_idx, col_idx))
                continue

            # Period but no sample
            if period and not sample_idx:
                sample_idx = channel_samples.get(col_idx)
                if sample_idx is None:
                    fmt = 'Missing sample at cell %4d:%d and ' \
                        + 'no channel sample. MOD bug?'
                    print(fmt % (row_idx, col_idx))
                    continue
                fmt = 'Using last sample at cell %4d:%d'
                SP.print(fmt % (row_idx, col_idx))
                sample_idx = channel_samples[col_idx]

            note_idx = period_to_idx(period)
            assert 0 <= note_idx < 60
            vol = mod_note_volume(mod, cell)
            yield Note(col_idx, row_idx,
                       sample_idx, note_idx,
                       vol, time_ms)
            channel_samples[col_idx] = sample_idx
            channel_periods[col_idx] = period

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
