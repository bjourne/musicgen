# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.defs import (DEFAULT_SPEED, DEFAULT_TEMPO,
                           EFFECT_CMD_UPDATE_TIMING,
                           EFFECT_CMD_SET_VOLUME,
                           Note,
                           period_to_idx)
from musicgen.utils import SP

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

# If the sample is present but not the period, it is ambiguous whether
# the cell represents repeating the note or bending a ringing
# note. For most but not all MODs, the MIDI representation becomes
# better if such cells are assumed to be bends.
#
# Worse : blu_angel_-_dream.mod
# Better: agnostic.mod
def rows_to_notes(rows, volumes):
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
            vol = mod_note_volume(volumes[sample_idx - 1], cell)
            yield Note(col_idx, row_idx,
                       sample_idx, note_idx,
                       vol, time_ms)
            channel_samples[col_idx] = sample_idx
            channel_periods[col_idx] = period
