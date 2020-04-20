# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import *

def is_melody(melody, min_length, min_notes, max_repeat):
    notes = [n for (s, r, n) in melody]
    if len(notes) < min_length:
        return False
    if len(set(notes)) < min_notes:
        return False
    note_groups = groupby(notes)
    if any(len(list(grp)) > max_repeat for (_, grp) in note_groups):
        return False
    return True

def chunk_notes(notes, max_dist):
    buf = []
    at_row = None
    at_time = None
    for col, row, s, n, vol, time_ms in notes:
        if at_row is None:
            at_row = row
        if at_time is None:
            at_time = time_ms
        # Either notes are to far apart or the tempo changes.
        if (row - at_row > max_dist) or at_time != time_ms:
            yield buf
            buf = []
        buf.append((col, row, s, n))
        at_row = row
        at_time = time_ms
    yield buf

def note_melodies(notes, min_length, min_notes, max_repeat, max_dist):
    # Sort and group by column
    notes = sorted(notes, key = lambda r: r[0])
    notes_per_col = groupby(notes, lambda r: r[0])
    seen = set()
    for col, group1 in notes_per_col:
        notes_per_sample = groupby(list(group1), lambda r: r[2])
        for sample, group2 in notes_per_sample:
            for group3 in chunk_notes(group2, max_dist):
                melody = list(group3)
                base_row = melody[0][1]
                melody = tuple((s, r - base_row, n)
                               for (c, r, s, n) in melody)
                ok = is_melody(melody, min_length, min_notes, max_repeat)
                if melody in seen or not ok:
                    continue
                yield melody
                seen.add(melody)

def build_cell_row(sample, note, effect_cmd, effect_arg):
    if note == -1:
        period = 0
    else:
        period = PERIODS[note]
    sample_lo = sample & 0xf
    sample_hi = sample >> 4
    effect_arg1 = effect_arg & 0xf
    effect_arg2 = effect_arg >> 4
    cell = dict(period = period,
                sample_lo = sample_lo,
                sample_hi = sample_hi,
                effect_cmd = effect_cmd,
                effect_arg1 = effect_arg1,
                effect_arg2 = effect_arg2)
    zero_cell = dict(period = 0,
                     sample_lo = 0,
                     sample_hi = 0,
                     effect_cmd = 0,
                     effect_arg1 = 0,
                     effect_arg2 = 0)
    return [cell, zero_cell, zero_cell, zero_cell]

ZERO_CELL_ROW = build_cell_row(0, -1, 0, 0)
ZERO_CELL_ROW_SILENCE = build_cell_row(0, -1, 0xc, 0)

def melody_to_rows(melody, end_length):
    at = 0
    for sample, rel_row, period in melody:
        while at < rel_row:
            yield ZERO_CELL_ROW
            at += 1
        yield build_cell_row(sample, period, 0, 0)
        at += 1
    for _ in range(end_length // 2):
        yield ZERO_CELL_ROW
    for _ in range(end_length // 2):
        yield ZERO_CELL_ROW_SILENCE


def rows_to_pattern(rows):
    n = len(rows)
    rows = rows + [ZERO_CELL_ROW] * (64 - n)
    return rows

def rows_to_patterns(rows):
    for i in range(0, len(rows), 64):
        yield dict(rows = rows_to_pattern(rows[i:i + 64]))

def main():
    parser = ArgumentParser(
        description = 'Isolate melodies in MOD files')
    parser.add_argument('input_module', type = FileType('rb'))
    parser.add_argument('output_module', type = FileType('wb'))
    parser.add_argument(
        '--trailer', type = int, required = True,
        help = 'length of melody trailer')
    parser.add_argument(
        '--min-length', type = int,
        help = 'minimum number of notes in a melody')
    parser.add_argument(
        '--min-unique', type = int,
        help = 'minimum number of unique notes in a melody')
    parser.add_argument(
        '--max-repeat', type = int, required = True,
        help = 'maximum number of repeated notes in a melody')
    parser.add_argument(
        '--max-distance', type = int, required = True,
        help = 'maximum distance between notes in a melody')
    args = parser.parse_args()

    with args.input_module as inf:
        mod = Module.parse(inf.read())

    rows = linearize_rows(mod)
    notes = notes_in_rows(mod, rows)
    melodies = note_melodies(notes,
                             args.min_length,
                             args.min_unique,
                             args.max_repeat,
                             args.max_distance)
    rows_out = sum([list(melody_to_rows(melody, args.trailer))
                    for melody in melodies], [])

    patterns = list(rows_to_patterns(rows_out))
    n_patterns = len(patterns)

    pattern_table = list(range(n_patterns)) + [0] * (128 - n_patterns)

    mod_out = dict(title = mod.title,
                   sample_headers = mod.sample_headers,
                   n_played_patterns = n_patterns,
                   pattern_table = bytearray(pattern_table),
                   patterns = patterns,
                   samples = mod.samples)
    with args.output_module as outf:
        outf.write(Module.build(mod_out))

if __name__ == '__main__':
    main()
