# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# MyCode is a bad name for the internal data format I'm using.
from itertools import takewhile
from musicgen.analyze import sample_props
from musicgen.corpus import load_index
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import ModNote, linearize_rows, column_to_mod_notes
from musicgen.ssrs import find_min_ssr
from musicgen.utils import (SP, file_name_for_params,
                            flatten,
                            load_pickle, save_pickle)
from time import time

def produce_jumps(delta, do_pack):
    # Limit jumps to 8 rows to save vocabulary space.
    if do_pack:
        delta = max(delta, 8)
    thresholds = [64, 32, 16, 8, 4, 3, 2, 1]
    for threshold in thresholds:
        while delta >= threshold:
            yield threshold
            delta -= threshold
    assert delta == 0

def produce_pitches(delta):
    while delta >= 12:
        yield 12
        delta -= 12
    while delta <= -12:
        yield -12
        delta += 12
    if delta != 0:
        yield delta

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
INSN_PROGRAM = 'X'
INSN_PITCH = 'I'

INSN_BLOCK = 'B'
INSN_REPEAT = 'R'

def bin_repeats(reps):
    thresholds = [16, 8, 4, 3, 2]
    for thr in thresholds:
        if reps >= thr:
            return thr
    return 1

MIN_PACK = 4
def pack_mycode(seq):
    if len(seq) <= MIN_PACK:
        return seq
    start, w, reps = find_min_ssr(seq)

    if reps == 1 or reps * w < MIN_PACK:
        return seq

    p1 = pack_mycode(seq[:start])
    p2 = pack_mycode(seq[start:start + w])
    p3 = pack_mycode(seq[start + w*reps:])

    block_tok = (INSN_BLOCK, 0)
    rep_tok = (INSN_REPEAT, bin_repeats(reps))

    return p1 + [block_tok] + p2 + [rep_tok] + p3

def mod_notes_to_mycode(notes, instruments, n_rows, do_pack):
    seq = []
    first_jump = notes[0].row_idx if notes else n_rows
    for jump in produce_jumps(first_jump, do_pack):
        seq.append((INSN_JUMP, jump))

    last_duration = None
    last_pitch_idx = None
    first_pitch = None
    for note in notes:
        # Maybe update duration
        duration, jump = duration_and_jump(last_duration, note.duration)
        if duration != last_duration:
            seq.append((INSN_DUR, duration))
        last_duration = duration

        instrument = instruments[note.sample_idx]

        if instrument == 1:
            if last_pitch_idx is None:
                first_pitch = note.pitch_idx
            else:
                pitch_delta = note.pitch_idx - last_pitch_idx
                for pitch in produce_pitches(pitch_delta):
                    seq.append((INSN_PITCH, pitch))
            last_pitch_idx = note.pitch_idx
        seq.append((INSN_PLAY, instrument))
        for jump in produce_jumps(jump, do_pack):
            seq.append((INSN_JUMP, jump))
    if do_pack:
        seq = pack_mycode(seq)
    return first_pitch, seq

def unpack_block(seq, top_level):
    seq2 = []
    while seq:
        cmd, arg = seq.pop(0)
        if cmd == INSN_BLOCK:
            seq2.extend(unpack_block(seq, False))
        elif cmd == INSN_REPEAT:
            if not top_level:
                return seq2 * arg
        else:
            seq2.append((cmd, arg))
    return seq2

def mycode_to_mod_notes(seq, col_idx, time_ms, pitch_idx, dur):
    n_packed = len(seq)
    seq = unpack_block(seq, True)
    n_after = len(seq)
    SP.print('Unpacking %d symbols to %d.' % (n_packed, n_after))

    row_idx = 0
    notes = []
    blocks = []
    for i, (cmd, arg) in enumerate(seq):
        if cmd == INSN_JUMP:
            row_idx += arg
        elif cmd == INSN_DUR:
            dur = arg
        elif cmd == INSN_PITCH:
            pitch_idx += arg
        elif cmd == INSN_PLAY:
            pitch_to_use = 0
            if arg == 1:
                assert pitch_idx is not None
                pitch_to_use = pitch_idx
            note = ModNote(row_idx, col_idx,
                           arg, pitch_to_use, 64, time_ms)
            # I think this should work.
            note.duration = dur
            notes.append(note)
            row_idx += dur
        else:
            assert False
    return notes

class MyCodedModule:
    def __init__(self, name, time_ms, cols):
        self.name = name
        self.time_ms = time_ms
        self.cols = cols

def mod_file_to_mycode(file_path, do_pack):
    SP.print(str(file_path))
    mod = load_file(file_path)
    rows = linearize_rows(mod)

    # Don't care about volumes
    volumes = [64] * len(mod.sample_headers)
    col_notes = [column_to_mod_notes(rows, i, volumes) for i in range(4)]
    all_notes = flatten(col_notes)
    instruments = {}
    n_perc = 0
    for sample_idx, props in sample_props(mod, all_notes):
        if not props.is_percussive:
            instruments[sample_idx] = 1
        else:
            instruments[sample_idx] = 2 + n_perc
            n_perc = (n_perc + 1) % 3

    time_ms = all_notes[0].time_ms
    n_rows = len(rows)
    cols = [mod_notes_to_mycode(notes, instruments, n_rows, do_pack)
            for notes in col_notes]
    return MyCodedModule(file_path.name, time_ms, cols)

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

    if not prefix or not (INSN_PLAY, 1) in prefix:
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
        return 2 if durs[0] == 1 else durs[0] - 1
    return durs[1]

def guess_initial_pitch(seq):
    at_note = 0
    min_note = 0
    max_note = 0
    for cmd, arg in seq:
        if cmd == INSN_PITCH:
            at_note += arg
            max_note = max(at_note, max_note)
            min_note = min(at_note, min_note)
    return -min_note + 12

########################################################################
# Cache generation
########################################################################
def mod_file_to_mycode_safe(fname, do_pack):
    try:
        return mod_file_to_mycode(fname, do_pack)
    except PowerPackerModule:
        SP.print('Skipping PP20 module.')
        return None

def disk_corpus_to_mycode_mods(corpus_path, mods, do_pack):
    start = time()
    SP.header('PARSING', '%d modules', len(mods))
    fnames = [corpus_path / mod.genre / mod.fname for mod in mods]
    seq = [mod_file_to_mycode_safe(fname, do_pack) for fname in fnames]
    seq = [e for e in seq if e]
    SP.leave()
    delta = time() - start
    SP.print('Parsed corpus in %.2f seconds.', delta)
    return seq

def corpus_to_mycode_mods(corpus_path, kb_limit, do_pack):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]

    size_sum = sum(mod.kb_size for mod in mods)
    params = (size_sum, kb_limit, do_pack)
    cache_file = file_name_for_params('mycode_cache', 'pickle', params)
    print(cache_file)
    cache_path = corpus_path / cache_file
    if not cache_path.exists():
        seq = disk_corpus_to_mycode_mods(corpus_path, mods, do_pack)
        SP.print('Saving MyCode cache...')
        save_pickle(cache_path, seq)
    else:
        SP.print('Loading MyCode cache from %s.', cache_path)
    return load_pickle(cache_path)
