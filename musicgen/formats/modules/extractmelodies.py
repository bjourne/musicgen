# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import *

def group_completion(buf,
                     last_sample, last_time,
                     sample, time, row_delta,
                     max_distance, mu_factor, mu_threshold):

    if last_sample != sample:
        return True, 'sample_change (%d -> %d)' % (last_sample, sample)
    if last_time != time:
        return True, 'tempo_change'
    if row_delta > max_distance:
        return True, 'notes_apart'

    # Rows containing notes
    rows = [i for i, cell in enumerate(buf) if cell.sample_idx != 0]
    n_rows = len(rows)
    if n_rows < mu_threshold:
        return False, 'less_than_mu_threshold'

    diff_sum = sum(y - x for (x, y) in zip(rows, rows[1:]))
    avg_diff = diff_sum / (n_rows - 1)

    if row_delta > avg_diff * mu_factor:
        return True, 'exceeds_mu_factor'
    return False, 'no'

def extract_sample_groups(rows, col_idx,
                          max_distance, mu_factor, mu_threshold):
    tempo = DEFAULT_TEMPO
    speed = DEFAULT_SPEED
    last_time = None
    last_sample = None
    last_row = None
    buf = []

    for row_idx, row in enumerate(rows):
        tempo, speed = update_timings(row, tempo, speed)
        time = int(calc_row_time(tempo, speed) * 1000)
        cell = row[col_idx]
        sample = cell.sample_idx
        if last_sample is None:
            last_sample = sample
        if last_time is None:
            last_time = time
        if last_row is None:
            last_row = row_idx

        if sample != 0:
            row_delta = row_idx - last_row
            status, msg = group_completion(
                buf,
                last_sample, last_time,
                sample, time,
                row_delta,
                max_distance, mu_factor, mu_threshold)
            if status:
                yield buf, msg
                buf = []
            last_sample = sample
            last_row = row_idx
            last_time = time

        # Remove pattern jumps and tempo changes
        if cell.effect_cmd in (0xb, 0xd, 0xf):
            cell.effect_cmd = 0
            cell.effect_arg1 = 0
            cell.effect_arg2 = 0

        buf.append(cell)
    yield buf, 'last_one'

def is_group_melody(cells, min_length, min_notes, max_repeat):
    notes = [c.period for c in cells if c.sample_idx != 0]
    if len(notes) < min_length:
        return False
    if len(set(notes)) < min_notes:
        return False
    note_groups = groupby(notes)
    if any(len(list(grp)) > max_repeat for (_, grp) in note_groups):
        return False
    return True

def filter_duplicate_melodies(melodies):
    seen = set()
    for melody in melodies:
        signature = tuple((c.sample_idx, i, c.period)
                          for i,c in enumerate(melody)
                          if c.sample_idx != 0)
        if not signature in seen:
            yield melody
        seen.add(signature)

def build_cell(sample, note, effect_cmd, effect_arg):
    if note == -1:
        period = 0
    else:
        period = PERIODS[note]
    sample_lo = sample & 0xf
    sample_hi = sample >> 4
    effect_arg1 = effect_arg & 0xf
    effect_arg2 = effect_arg >> 4
    return dict(period = period,
                sample_lo = sample_lo,
                sample_hi = sample_hi,
                effect_cmd = effect_cmd,
                effect_arg1 = effect_arg1,
                effect_arg2 = effect_arg2)

ZERO_CELL = build_cell(0, -1, 0, 0)
ZERO_CELL_SILENCE = build_cell(0, -1, 0xc, 0)

def fix_trailer(cells, end_length):
    fixed_cells = []
    buf = []
    for cell in cells:
        if cell.sample_idx != 0 or cell.effect_cmd != 0:
            fixed_cells.extend(buf)
            buf = []
        buf.append(cell)
    end_length_2 = end_length // 2
    return fixed_cells \
        + [ZERO_CELL] * end_length_2 + [ZERO_CELL_SILENCE] * end_length_2

def rows_to_pattern(rows):
    zero_cell_row = [ZERO_CELL] * 4
    n = len(rows)
    rows = rows + [zero_cell_row] * (64 - n)
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
    parser.add_argument(
        '--mu-factor', type = float, default = 3.0,
        help = 'maximum factor of mean note distance allowed')
    parser.add_argument(
        '--mu-threshold', type = int, default = 5,
        help = 'minimum number of notes before measuring the mu factor')

    args = parser.parse_args()
    with args.input_module as inf:
        mod = Module.parse(inf.read())

    rows = linearize_rows(mod)

    groups = sum(
        [list(extract_sample_groups(rows, col_idx,
                                    args.max_distance,
                                    args.mu_factor,
                                    args.mu_threshold))
         for col_idx in range(4)], [])

    groups = [group for (group, msg) in groups
              if is_group_melody(group,
                                 args.min_length,
                                 args.min_unique,
                                 args.max_repeat)]

    groups = filter_duplicate_melodies(groups)

    groups = [fix_trailer(group, args.trailer) for group in groups]
    cells = sum(groups, [])
    rows = [[c, ZERO_CELL, ZERO_CELL, ZERO_CELL] for c in cells]

    patterns = list(rows_to_patterns(rows))
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
