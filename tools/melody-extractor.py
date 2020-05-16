# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from construct import Container
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file, save_file
from sys import exit

def group_completion(buf, cell,
                     last_sample, sample,
                     last_timing, timing,
                     row_delta,
                     max_distance, mu_factor, mu_threshold):

    if last_sample != sample:
        return True, 'sample_change (%d -> %d)' % (last_sample, sample)
    if last_timing != timing:
        return True, 'tempo_change'

    if row_delta > max_distance:
        return True, 'notes_apart'

    notes = [period_to_idx(c.period) for c in buf if c.sample_idx != 0]
    notes.append(period_to_idx(cell.period))
    hi_octave = max(notes) // 12
    lo_octave = min(notes) // 12
    if hi_octave - lo_octave >= 2:
        return True, 'two_octaves'

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

def build_cell(sample, note, effect_cmd, effect_arg):
    if note == -1:
        period = 0
    else:
        period = PERIODS[note]
    sample_lo = sample & 0xf
    sample_hi = sample >> 4
    effect_arg1 = effect_arg >> 4
    effect_arg2 = effect_arg & 0xf
    return dict(period = period,
                sample_lo = sample_lo,
                sample_hi = sample_hi,
                sample_idx = sample_hi << 4 + sample_lo,
                effect_cmd = effect_cmd,
                effect_arg1 = effect_arg1,
                effect_arg2 = effect_arg2)

def is_silence(cell):
    return (cell.effect_cmd == 0xc
            and cell.effect_arg1 == 0
            and cell.effect_arg2 == 0)

# def is_note(cell):
#     return cell.sample_idx != 0 and not is_silence(cell)

def extract_sample_groups(rows, col_idx,
                          max_distance, mu_factor, mu_threshold):
    # Tempo and speed
    timing = DEFAULT_TEMPO, DEFAULT_SPEED
    last_timing = None
    last_sample = None
    last_sample_row = None
    buf = []
    current_period = None
    for row_idx, row in enumerate(rows):
        timing = update_timings(row, *timing)
        cell = row[col_idx]
        sample = cell.sample_idx

        if sample != 0:
            if last_sample is None:
                last_sample = sample
            if last_sample_row is None:
                last_sample_row = row_idx
            if last_timing is None:
                last_timing = timing

            # What a hack.
            if current_period is None:
                current_period = cell.period
                if current_period == 0:
                    current_period = PERIODS[BASE_NOTE_IDX]
            if cell.period:
                current_period = cell.period
            cell.period = current_period

            row_delta = row_idx - last_sample_row
            status, msg = group_completion(
                buf, cell,
                last_sample, sample,
                last_timing, timing,
                row_delta,
                max_distance, mu_factor, mu_threshold)
            if status:
                # Emit timing info.
                speed = last_timing[1]
                timing_cell = Container(build_cell(0, -1, 0xf, speed))
                assert buf[0].sample_idx != 0
                yield [timing_cell] + buf, msg
                buf = []
            last_sample = sample
            last_sample_row = row_idx
            last_timing = timing

        # Remove pattern jumps
        if cell.effect_cmd in (0xb, 0xd):
            cell.effect_cmd = 0
            cell.effect_arg1 = 0
            cell.effect_arg2 = 0
        if sample != 0 or buf:
            buf.append(cell)
    yield buf, 'last_one'

def is_melody(cells, min_length, min_notes, max_repeat):
    notes = [c.period for c in cells if c.sample_idx != 0]
    if len(notes) < min_length:
        return False
    if len(set(notes)) < min_notes:
        return False
    note_groups = groupby(notes)
    if any(len(list(grp)) > max_repeat for (_, grp) in note_groups):
        return False
    return True

def move_to_c(cells):
    notes = [period_to_idx(c.period) for c in cells if c.period != 0]

    first_note_in_octave = notes[0] % 12
    if first_note_in_octave <= 6:
        delta = -first_note_in_octave
    else:
        delta = 12 - first_note_in_octave

    c_offset = notes[0] % 12
    for c in cells:
        if c.period != 0:
            new_note = period_to_idx(c.period) + delta
            assert 0 <= new_note < len(PERIODS)
            c.period = PERIODS[new_note]
    return cells

def filter_duplicate_melodies(melodies):
    seen = set()
    for melody in melodies:
        signature = tuple((c.sample_idx, i, c.period)
                          for i,c in enumerate(melody)
                          if c.sample_idx != 0)
        if not signature in seen:
            yield melody
        seen.add(signature)

ZERO_CELL = build_cell(0, -1, 0, 0)
ZERO_CELL_SILENCE = build_cell(0, -1, 0xc, 0)

def add_trailer(melody, end_length):
    end_length_2 = end_length // 2
    return melody + [ZERO_CELL] * end_length_2 \
        + [ZERO_CELL_SILENCE] * end_length_2

def remove_ending_silence(melody):
    last_idx = [i for i, c in enumerate(melody)
                if c.sample_idx != 0 or c.effect_cmd != 0][-1]
    return melody[:last_idx + 1]

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
    parser.add_argument(
        '--transpose',
        help = 'transpose all melodies to c',
        type = eval,
        choices = [True, False],
        default = 'True')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'print debug stuff')

    args = parser.parse_args()
    args.input_module.close()
    args.output_module.close()
    mod = load_file(args.input_module.name)
    rows = linearize_rows(mod)

    melodies = sum(
        [list(extract_sample_groups(rows, col_idx,
                                    args.max_distance,
                                    args.mu_factor,
                                    args.mu_threshold))
         for col_idx in range(4)], [])

    melodies = [melody for (melody, msg) in melodies
                if is_melody(melody,
                             args.min_length,
                             args.min_unique,
                             args.max_repeat)]

    if args.transpose:
        melodies = [move_to_c(melody) for melody in melodies]

    melodies = [remove_ending_silence(melody)
                for melody in filter_duplicate_melodies(melodies)]
    if args.info:
        print(f'=== {len(melodies)} MELODIES ===')
        for melody in melodies:
            print('\n'.join(cell_to_string(c) for c in melody))
            print()
    melodies = [add_trailer(melody, args.trailer) for melody in melodies]
    if not melodies:
        fmt = 'Sorry, found no melodies in "%s"!'
        print(fmt % args.input_module.name)
        exit(1)

    cells = sum(melodies, [])
    rows = [[c, ZERO_CELL, ZERO_CELL, ZERO_CELL] for c in cells]

    patterns = list(rows_to_patterns(rows))
    n_patterns = len(patterns)

    pattern_table = list(range(n_patterns)) + [0] * (128 - n_patterns)

    mod_out = dict(title = mod.title,
                   sample_headers = mod.sample_headers,
                   n_orders = n_patterns,
                   restart_pos = 0,
                   pattern_table = bytearray(pattern_table),
                   initials = 'M.K.'.encode('utf-8'),
                   patterns = patterns,
                   samples = mod.samples)
    save_file(args.output_module.name, mod_out)

if __name__ == '__main__':
    main()
