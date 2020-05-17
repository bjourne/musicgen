# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# MyCode is a bad name for the internal data format I'm using.
from construct import Container
from itertools import groupby, takewhile
from musicgen.corpus import load_index
from musicgen.defs import Note, period_to_idx
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import linearize_rows
from musicgen.utils import SP, find_best_split, find_best_split2
from pathlib import Path
from pickle import dump, load
from random import randint

########################################################################
# Native format to mycode
########################################################################
def column_to_notes(rows, col_idx):
    for row_idx, row in enumerate(rows):
        cell = row[col_idx]
        if not cell.period:
            continue
        note_idx = period_to_idx(cell.period)
        sample_idx = cell.sample_idx
        if not sample_idx <= 0x1f:
            fmt = 'Skipping invalid sample %d in module.'
            SP.print(fmt, sample_idx)
            continue
        yield row_idx, note_idx, sample_idx

def produce_jumps(delta):
    thresholds = [64, 32, 16, 8, 4, 3, 2, 1]
    for threshold in thresholds:
        while delta >= threshold:
            yield threshold
            delta -= threshold
    assert delta == 0

def duration_and_jump(duration, row_delta):
    if duration is None:
        if row_delta > 8:
            return 8, row_delta - 8
        return row_delta, 0
    else:
        if row_delta > 8:
            return duration, row_delta - duration
        return row_delta, 0

INSN_JUMP = 'J'
INSN_DUR = 'D'
INSN_PLAY = 'P'
INSN_SAMPLE = 'S'
INSN_PROGRAM = 'X'
INSN_REPEAT = 'R'
INSN_END_BLOCK = 'E'

# 100 is a magic number and means the value is either random or
# required to be inputted from the outside.
INPUT_ARG = 100

def get_sample_val(last_sample_idx, sample_idx):
    if last_sample_idx is None:
        return INPUT_ARG
    elif sample_idx != last_sample_idx:
        return sample_idx - last_sample_idx
    return None

def get_play_val(last_note_idx, note_idx):
    if last_note_idx is None:
        return INPUT_ARG
    return note_idx - last_note_idx

def column_to_mycode(rows, col_idx):
    notes = list(column_to_notes(rows, col_idx))

    # Add two dummy notes
    notes = [(0, 0, 0)] + notes + [(len(rows), 0, 0)]

    # Compute row deltas
    notes = [(r2 - r1, n1, s1) for ((r1, n1, s1), (r2, n2, s2))
             in zip(notes, notes[1:])]

    # Maybe emit jumps
    for jump in produce_jumps(notes[0][0]):
        yield INSN_JUMP, jump

    last_duration = None
    last_note_idx = None
    last_sample_idx = None
    for row_delta, note_idx, sample_idx in notes[1:]:
        duration, jump = duration_and_jump(last_duration, row_delta)
        if duration != last_duration:
            yield INSN_DUR, duration
        last_duration = duration

        sample_val = get_sample_val(last_sample_idx, sample_idx)
        if sample_val is not None:
            yield INSN_SAMPLE, sample_val

        play_val = get_play_val(last_note_idx, note_idx)
        yield INSN_PLAY, play_val
        for jmp in produce_jumps(jump):
            yield INSN_JUMP, jmp
        last_note_idx = note_idx
        last_sample_idx = sample_idx

def pack_mycode(mycode):
    starti, width, n_reps = find_best_split2(mycode)
    if n_reps == 1:
        return mycode
    SP.print('Packing code of length %d.', len(mycode))
    intro = pack_mycode(mycode[:starti])
    loop = pack_mycode(mycode[starti:starti + width])
    outro = pack_mycode(mycode[starti + width * n_reps:])
    return intro + [(INSN_REPEAT, n_reps)] + loop \
        + [(INSN_END_BLOCK, 0)] + outro

def rows_to_mycode(rows):
    for col_idx in range(4):
        mycode = list(column_to_mycode(rows, col_idx))
        # Still to slow.
        # mycode = pack_mycode(mycode)
        for ev in mycode:
            yield ev
        yield INSN_PROGRAM, 0

def mod_file_to_mycode(fname):
    SP.print(str(fname))
    try:
        mod = load_file(fname)
    except PowerPackerModule:
        SP.print('Skipping PP20 module.')
        return []
    rows = linearize_rows(mod)
    return rows_to_mycode(rows)

########################################################################
# Mycode to native format
########################################################################
def build_cell(sample, note, effect_cmd, effect_arg):
    if note == -1:
        period = 0
    else:
        period = PERIODS[note]
    sample_lo = sample & 0xf
    sample_hi = sample >> 4
    sample_idx = (sample_hi << 4) + sample_lo
    effect_arg1 = effect_arg >> 4
    effect_arg2 = effect_arg & 0xf
    return Container(dict(period = period,
                          sample_lo = sample_lo,
                          sample_hi = sample_hi,
                          sample_idx = sample_idx,
                          effect_cmd = effect_cmd,
                          effect_arg1 = effect_arg1,
                          effect_arg2 = effect_arg2))

ZERO_CELL = build_cell(0, -1, 0, 0)

########################################################################
# Guessing logic
########################################################################
def find_subseq(seq, subseq):
    l = len(subseq)
    for i in range(len(seq) - l + 1):
        if seq[i:i+l] == subseq:
            yield i

def guess_initial_duration(seq):
    prefix = list(takewhile(lambda x: x[0] != INSN_DUR, seq))
    if prefix == seq:
        # No duration token in sequence so we pick the default one.
        SP.print('No duration changes.')
        return 2

    play_in_prefix = any(x[0] == INSN_PLAY for x in prefix)
    if not prefix or not play_in_prefix:
        # Initial duration doesn't matter if there are no notes in the
        # prefix.
        SP.print('No play instructions in prefix.')
        return 2

    # Is the prefix present in another part of the sequence?
    last_index = list(find_subseq(seq, prefix))[-1]
    if last_index != 0:
        # If so the last duration token before the last occurrence is
        # the initial duration.
        dur = [arg for (cmd, arg) in seq[:last_index]
               if cmd == INSN_DUR][-1]
        return dur

    # Take the second duration if there is any. If there isn't
    durs = [arg for (cmd, arg) in seq if cmd == INSN_DUR]
    if len(durs) == 1:
        return 2 if dur[0] == 1 else dur[0] - 1
    return durs[1]

def guess_initial_note(sample):
    at_note = 0
    min_note = 0
    max_note = 0
    for cmd, arg in sample:
        if cmd == INSN_PLAY:
            at_note += arg
            max_note = max(at_note, max_note)
            min_note = min(at_note, min_note)
    return -min_note + 12

def guess_initial_sample(seq):
    at_sample = 0
    min_sample = 0
    max_sample = 0
    for cmd, arg in seq:
        if cmd == INSN_SAMPLE:
            at_sample += arg
            max_sample = max(at_sample, max_sample)
            min_sample = min(at_sample, min_sample)
    return -min_sample + 1

def mycode_to_notes(seq):
    print(seq)
    # First make some educated guesses
    note = guess_initial_note(seq)
    sample = guess_initial_sample(seq)
    duration = guess_initial_duration(seq)
    assert duration is not None

    SP.print('Guessed initial note %d, sample %d, and duration %d.',
             (note, sample, duration))

    row = 0
    for cmd, arg in seq:
        assert arg != INPUT_ARG
        if cmd == INSN_PLAY:
            note += arg
            if not 0 <= note < 60:
                SP.print('Fixing bad note %d.', note)
                note = note % 60
            yield Note(0, row, sample, note, 64, 100)
            row += duration
        elif cmd == INSN_SAMPLE:
            sample += arg
        elif cmd == INSN_DUR:
            duration = arg
        elif cmd == INSN_JUMP:
            row += arg
        else:
            assert False

# I have to work on this. OOB notes and samples and missing durations
# should be handled better.
def mycode_to_column(seq, sample, note):
    row_idx = 0
    col = []

    duration = None
    first_note = True
    first_sample = True

    if sample is None:
        sample = randint(3, 10)
    if note is None:
        note = randint(27, 30)

    for cmd, arg in seq:
        if cmd == INSN_JUMP:
            for _ in range(arg):
                yield ZERO_CELL
        elif cmd == INSN_DUR:
            duration = arg
        elif cmd == INSN_SAMPLE:
            if first_sample:
                first_sample = False
            else:
                if arg == INPUT_ARG:
                    sample = randint(3, 10)
                else:
                    sample += arg
        elif cmd == INSN_PLAY:
            if first_note:
                first_note = False
            else:
                if arg == INPUT_ARG:
                    note = randint(25, 35)
                else:
                    note += arg
            # Should solve this better
            if not (1 <= sample <= 31):
                SP.print('Sample %d oob.', sample)
                sample = randint(3, 10)
            if not (0 <= note < 60):
                SP.print('Note %d oob.', note)
                note = randint(25, 35)
            yield build_cell(sample, note, 0, 0)
            if duration is None:
                SP.print('No duration, assuming 2.')
                duration = 2
            for _ in range(duration - 1):
                yield ZERO_CELL
        else:
            assert False

def mycode_to_rows(seq, col_defs):
    def pred(x):
        return x == (INSN_PROGRAM, 0)
    parts = [list(group) for k, group
             in groupby(seq, pred) if not k]
    if not col_defs:
        col_defs = [(None, None)] * len(parts)
    cols = [list(mycode_to_column(part, sample, note))
            for part, (sample, note) in zip(parts, col_defs)]

    # Pad with missing cols
    cols = cols + [[] for _ in range(4 - len(cols))]

    # Pad with empty cells
    max_len = max(len(col) for col in cols)
    cols = [col + [ZERO_CELL] * (max_len - len(col))
            for col in cols]
    return zip(*cols)

########################################################################
# Pretty printing (maybe not useful)
########################################################################
def pretty_insn(ind, cmd, arg):
    if cmd in (INSN_JUMP, INSN_PLAY, INSN_DUR, INSN_SAMPLE, INSN_REPEAT):
        str = '%-2s %4d' % (cmd.upper(), arg)
    else:
        str = cmd.upper()
    return ' ' * ind + str

def prettyprint_mycode(mycode):
    ind = 0
    for cmd, arg in mycode:
        if cmd == INSN_END_BLOCK:
           ind -= 2
        print(pretty_insn(ind, cmd, arg))
        if cmd == INSN_REPEAT:
            ind += 2

########################################################################
# Cache generation
########################################################################
def disk_corpus_to_mycode(corpus_path, mods):
    SP.header('PARSING', '%d modules', len(mods))
    fnames = [corpus_path / mod.genre / mod.fname for mod in mods]
    seq = sum([list(mod_file_to_mycode(fname))
               for fname in fnames], [])
    SP.leave()
    return seq

def load_cache(cache_path):
    assert cache_path.exists()
    with open(cache_path, 'rb') as f:
        return load(f)

def limit_argument(cmd, arg, max_note_delta, max_sample_delta):
    if cmd == INSN_PLAY:
        return cmd, INPUT_ARG if abs(arg) > max_note_delta else arg
    elif cmd == INSN_SAMPLE:
        return cmd, INPUT_ARG if abs(arg) > max_sample_delta else arg
    return cmd, arg

def limit_arguments(seq, max_note_delta, max_sample_delta):
    return [limit_argument(cmd, arg, max_note_delta, max_sample_delta)
            for (cmd, arg) in seq]

def cache_file_name(mods, kb_limit, max_note_delta, max_sample_delta):
    size_sum = sum(mod.kb_size for mod in mods)
    fmt = 'cache-%010d-%04d-%02d-%02d.pickle'
    return fmt % (size_sum, kb_limit, max_note_delta, max_sample_delta)

def corpus_to_mycode(corpus_path, kb_limit,
                     max_note_delta, max_sample_delta):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]

    cache_file = cache_file_name(mods, kb_limit,
                                 max_note_delta, max_sample_delta)

    # size_sum = sum(mod.kb_size for mod in mods)
    # cache_file = 'cache-%04d-%010d.pickle' % (kb_limit, size_sum)
    cache_path = corpus_path / cache_file
    if not cache_path.exists():
        seq = disk_corpus_to_mycode(corpus_path, mods)

        # Filter out uncommon tokens.
        seq = limit_arguments(seq, 26, 20)


        SP.print('Saving cache...')
        with open(cache_path, 'wb') as f:
            dump(seq, f)
    else:
        SP.print('Using cache at %s.', cache_path)
    return load_cache(cache_path)

########################################################################
# Debugging and analysis
########################################################################
def check_mycode(rows1, mycode):
    col_defs = [next(column_to_notes(rows1, col_idx), (0, 0, 0))
                for col_idx in range(4)]
    col_defs = [(sample, note) for (_, note, sample) in col_defs]

    rows2 = list(mycode_to_rows(mycode, col_defs))
    assert len(rows1) == len(rows2)
    for row1, row2 in zip(rows1, rows2):
        for cell1, cell2 in zip(row1, row2):
            note1 = period_to_idx(cell1.period)
            note2 = period_to_idx(cell2.period)
            assert note1 == note2
            # Samples without periods skipped
            if not cell1.period and not cell2.period:
                continue
            if not cell1.sample_idx == cell2.sample_idx:
                print('Diff at row %d' % row_idx)
            assert cell1.sample_idx == cell2.sample_idx

########################################################################

def main():
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(
        description = 'MyCode MOD analyzer')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'Print information')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--module', type = FileType('rb'),
        help = 'Path to module to analyze')
    group.add_argument(
        '--corpus-path',
        help = 'Path to corpus')
    parser.add_argument(
        '--model-path',
        help = 'Path to store model and cache.')
    args = parser.parse_args()
    SP.enabled = args.info

    if args.module:
        args.module.close()
        mod = load_file(args.module.name)
        rows = linearize_rows(mod)
        mycode = list(rows_to_mycode(rows))
        print(rows_to_string(mycode_to_rows(mycode, None)))
        # prettyprint_mycode(mycode)
        # check_mycode(rows, mycode)
    elif args.corpus_path:
        corpus_path = Path(args.corpus_path)
        model_path = Path(args.model_path)
        mycode = get_sequence(corpus_path, model_path)
    analyze_mycode(mycode)

if __name__ == '__main__':
    main()
