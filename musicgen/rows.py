# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import defaultdict
from musicgen.defs import (DEFAULT_SPEED, DEFAULT_TEMPO,
                           EFFECT_CMD_UPDATE_TIMING,
                           EFFECT_CMD_SET_VOLUME,
                           Note,
                           period_to_idx)
from musicgen.utils import SP, flatten

def linearize_subsongs(mod, n_allowed_jumps):
    jump_counts = defaultdict(int)
    table_idx = 0
    next_from = 0
    play_order = []
    rows = []

    # Used orders pevents subsongs from looping into each other.
    used_orders = set()
    while table_idx < mod.n_orders:
        play_order.append(table_idx)
        pattern_idx = mod.pattern_table[table_idx]
        pattern = mod.patterns[pattern_idx]
        for i in range(next_from, 64):
            row = pattern.rows[i]
            rows.append(row)
            jump = False
            for cell in row:
                cmd = cell.effect_cmd
                arg1 = cell.effect_arg1
                arg2 = cell.effect_arg2
                if cmd == 0xb:
                    loc = table_idx, i
                    target_idx = 16 * arg1 + arg2
                    jump_count = jump_counts[loc]
                    if (jump_count < n_allowed_jumps and
                        target_idx not in used_orders):
                        jump_counts[loc] += 1
                        table_idx = target_idx - 1
                        next_from = 0
                        jump = True
                    else:
                        used_orders.update(play_order)
                        yield play_order, rows
                        play_order, rows = [], []
                elif cmd == 0xd:
                    next_from = 10 * arg1 + arg2
                    jump = True
            if jump:
                break
            else:
                next_from = 0
        table_idx += 1
    if play_order:
        yield play_order, rows

def update_timings(row, tempo, speed):
    for cell in row:
        if cell.effect_cmd == EFFECT_CMD_UPDATE_TIMING:
            val = 16 * cell.effect_arg1 + cell.effect_arg2
            if val <= 0x1f:
                speed = val
            else:
                tempo = val
    return tempo, speed

def calc_row_time(tempo, speed):
    return (60 / tempo / 4) * speed / 6

def mod_note_volume(default, cell):
    if cell.effect_cmd == EFFECT_CMD_SET_VOLUME:
        return (cell.effect_arg1 << 4) + cell.effect_arg2
    return default

class ModNote:
    def __init__(self, row_idx, col_idx, sample_idx, pitch_idx,
                 vol, time_ms):
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.sample_idx = sample_idx
        self.pitch_idx = pitch_idx
        self.vol = vol
        self.time_ms = time_ms
        self.duration = None

    def __str__(self):
        return '%s<%d, %d>' % (self.__class__.__name__,
                               self.row_idx, self.col_idx)

    def __repr__(self):
        fmt = '%s<%d, %d, %d, %d, %d>'
        return fmt % (self.__class__.__name__,
                      self.row_idx, self.col_idx,
                      self.sample_idx,
                      self.pitch_idx, self.duration)



# If the sample is present but not the period, it is ambiguous whether
# the cell represents repeating the note or bending a ringing
# note. For most but not all MODs, the MIDI representation becomes
# better if such cells are assumed to be bends.
#
# Worse : blu_angel_-_dream.mod
# Better: agnostic.mod
def column_to_mod_notes(rows, col_idx, volumes):
    tempo = DEFAULT_TEMPO
    speed = DEFAULT_SPEED
    col_period = None
    col_sample_idx = None
    notes = []
    for row_idx, row in enumerate(rows):
        tempo, speed = update_timings(row, tempo, speed)
        time_ms = int(calc_row_time(tempo, speed) * 1000)
        cell = row[col_idx]

        sample_idx = cell.sample_idx
        period = cell.period
        # Neither sample nor note, skipping
        if not sample_idx and not period:
            continue

        if sample_idx:
            col_sample_idx = sample_idx
        if period:
            col_period_idx = period

        # Sample but no note, we skip those.
        if sample_idx and not period:
            continue

        # Period but no sample
        if period and not sample_idx:
            sample_idx = col_sample_idx
            if sample_idx is None:
                fmt = 'Missing sample at cell %4d:%d and ' \
                    + 'no channel sample. MOD bug?'
                SP.print(fmt % (row_idx, col_idx))
                continue
            # fmt = 'Using last sample at cell %4d:%d'
            # SP.print(fmt % (row_idx, col_idx))

        vol_idx = sample_idx - 1
        if not 0 <= vol_idx < len(volumes):
            fmt = 'Sample %d out of bounds at cell %4d:%d. MOD bug?'
            SP.print(fmt % (sample_idx, row_idx, col_idx))
            continue
        vol = mod_note_volume(volumes[vol_idx], cell)
        pitch_idx = period_to_idx(period)
        assert 0 <= pitch_idx < 60

        note = ModNote(row_idx, col_idx,
                       sample_idx, pitch_idx,
                       vol, time_ms)
        notes.append(note)

    # Add durations
    for n1, n2 in zip(notes, notes[1:]):
        n1.duration = n2.row_idx - n1.row_idx
    if notes:
        notes[-1].duration = len(rows) - notes[-1].row_idx
    return notes

def rows_to_mod_notes(rows, volumes):
    return flatten([column_to_mod_notes(rows, i, volumes)
                    for i in range(4)])

if __name__ == '__main__':
    from sys import argv
    from musicgen.parser import load_file
    mod = load_file(argv[1])
    for play_order, rows in linearize_subsongs(mod, 1):
        print(play_order)
