# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
import numpy as np

PERIODS = [
    1712, 1616, 1525, 1440, 1357, 1281, 1209, 1141, 1077, 1017, 961, 907,
    856,   808,  762,  720,  678,  640,  604,  570,  538,  508, 480, 453,
    428,   404,  381,  360,  339,  320,  302,  285,  269,  254, 240, 226,
    214,   202,  190,  180,  170,  160,  151,  143,  135,  127, 120, 113,
    107,   101,   95,   90,   85,   80,   76,   71,   67,   64,  60,  57,
]
PERIOD_TO_IDX = {p : i for i, p in enumerate(PERIODS)}

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

def period_to_string(period):
    if period == 0:
        return '---'
    idx = PERIOD_TO_IDX[period]
    octave_idx = idx // 12
    note_idx = idx % 12
    note = NOTES[note_idx]
    return '%s%d' % (note, octave_idx)

def cell_to_string(cell):
    note = period_to_string(cell.period)
    effect_val = (cell.effect_cmd << 8) \
                 + (cell.effect_arg1 << 4) + (cell.effect_arg2)
    return '%s %02X %03X' % (note, cell.sample_idx, effect_val)

def row_to_string(row):
    return '  '.join(map(cell_to_string, row))

def rows_to_string(rows, numbering = False):
    strings = [row_to_string(row) for row in rows]
    if numbering:
        strings = ['#%03d %s' % (i, row) for i, row in enumerate(strings)]
    return '\n'.join(strings)

def pattern_to_string(pattern):
    return '\n'.join(map(row_to_string, pattern.rows))

def calc_row_time(tempo, speed):
    return (60 / tempo / 4) * speed / 6

def update_timings(row, tempo, speed):
    for cell in row:
        if cell.effect_cmd == 15:
            val = 16 * cell.effect_arg1 + cell.effect_arg2
            if val <= 0x1f:
                speed = val
            else:
                tempo = val
    return tempo, speed

def linearize_rows(mod):
    table_idx = 0
    rows = []
    jumps_taken = set()
    while table_idx < mod.n_played_patterns:
        pattern_idx = mod.pattern_table[table_idx]
        pattern = mod.patterns[pattern_idx]
        for i, row in enumerate(pattern.rows):
            rows.append(row)
            jump = False
            for cell in row:
                cmd = cell.effect_cmd
                if cmd == 0xb:
                    jump_loc = table_idx, i
                    if not jump_loc in jumps_taken:
                        table_idx = 16 * cell.effect_arg1 \
                            + cell.effect_arg2 - 1
                        jump = True
                        jumps_taken.add(jump_loc)
                elif cmd == 0xd:
                    next_from = 10 * cell.effect_arg1 + cell.effect_arg2
                    assert not next_from
                    jump = True
            if jump:
                break
        table_idx += 1
    return rows

########################################################################
# Sample management
########################################################################
class Sample:
    def __init__(self, bytes, repeat_from, repeat_len):
        arr = np.frombuffer(bytes, dtype = np.int8)
        arr = arr.astype(np.int16) * 256

        ratio = SAMPLE_RATE / AMIGA_SAMPLE_RATE
        n_samples = int(arr.size * ratio)

        x_old = np.linspace(0, 1, arr.size)
        x_new = np.linspace(0, 1, n_samples)
        if arr.size:
            arr = np.interp(x_new, x_old, arr)
        self.arr = arr.astype(np.float)
        self.repeat_from = round(2 * repeat_from * ratio)
        self.repeat_len = round(2 * repeat_len * ratio)

def load_samples(mod):
    return [Sample(data.bytes, header.repeat_from, header.repeat_len)
            for data, header in zip(mod.samples, mod.sample_headers)]
