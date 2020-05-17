# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""MIDI file generator

Usage:
    generate-midi.py [-hvo MIDI] [--programs=<seq>] module
        [--midi-mapping=<json>] <mod>
    generate-midi.py [-hvo MIDI] [--programs=<seq>] cache
        [--length=<len> --index=<index>] <cache>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    -o FILE --output FILE  output file [default: test.mid]
    --midi-mapping=<json>  instrument mapping [default: auto]
    --length=<len>         length of code to sample [default: 100]
    --index=<index>        index in cache file [default: random]
    --programs=<seq>       melodic and percussive programs
                           [default: 1,36:40,36,31]
"""
from collections import namedtuple
from docopt import docopt
from itertools import groupby, takewhile
from json import load
from mido import Message, MidiFile, MidiTrack
from musicgen.formats.modules import *
from musicgen.formats.modules.analyze import sample_props
from musicgen.formats.modules.mycode import (INSN_DUR,
                                             INSN_JUMP,
                                             INSN_PLAY,
                                             INSN_PROGRAM,
                                             INSN_SAMPLE,
                                             load_cache,
                                             mycode_to_notes)
from musicgen.formats.modules.parser import load_file
from musicgen.utils import SP, parse_comma_list, sort_groupby
from pathlib import Path
from random import choice, randrange

def note_duration(notes, i, row_idx, note_dur):
    if i < len(notes) - 1:
        next_row_idx = notes[i + 1][1]
        return min(next_row_idx - row_idx, note_dur)
    return note_dur

def midi_notes(notes, midi_mapping):
    offset_ms = 0
    last_row_idx = 0
    for i in range(len(notes)):
        col_idx, row_idx, sample, note_idx, vol, row_time_ms = notes[i]
        row_delta = row_idx - last_row_idx

        # Update time
        offset_ms += row_delta * row_time_ms
        last_row_idx = row_idx

        program, midi_idx_base, note_dur, vol_adj = midi_mapping[sample]
        note_dur = note_duration(notes, i, row_idx, note_dur)

        # -2 indicates filtered notes.
        if program == -2:
            continue

        # Drum track
        if program == -1:
            midi_idx = midi_idx_base
            col_idx = 9
            program = None
        else:
            midi_idx = midi_idx_base + note_idx
        velocity = int(min((vol / 64) * 127 * vol_adj, 127))

        note_on = offset_ms
        note_off = offset_ms + note_dur * row_time_ms

        yield col_idx, note_on, 1, program, midi_idx, velocity

        # We don't need note offs for drums but it doesn't hurt.
        yield col_idx, note_off, 0, program, midi_idx, 0

def midi_notes_to_track(channel, notes):
    prev = 0
    current_program = None
    for _, ofs, on_off, program, midi_idx, velocity in notes:
        rel_ofs = ofs - prev
        if program != current_program:
            current_program = program
            yield Message('program_change', program = program,
                          time = rel_ofs,
                          channel = channel)
            rel_ofs = 0

        if on_off == 1:
            yield Message('note_on',
                          note = midi_idx,
                          velocity = velocity,
                          time = rel_ofs,
                          channel = channel)
        else:
            yield Message('note_off',
                          note = midi_idx,
                          velocity = 0,
                          time = rel_ofs,
                          channel = channel)
        prev = ofs

def assign_instruments(samples, programs):
    mel_i = 0
    perc_i = 0
    midi_mapping = {}
    for sample_idx, is_percussive in samples:
        if is_percussive:
            program = programs.percussive[perc_i]
            midi_mapping[sample_idx] = [-1, program, 4, 1.0]
            perc_i = (perc_i + 1) % len(programs.percussive)
        else:
            program, base = programs.melodic[mel_i]
            midi_mapping[sample_idx] = [program, base, 4, 1.0]
            mel_i = (mel_i + 1) % len(programs.melodic)
    return midi_mapping

def notes_to_midi_file(notes, midi_file, midi_mapping):
    SP.print('%d notes in MOD file.', len(notes))
    SP.header('MIDI MAPPING', '%d samples', len(midi_mapping))
    SP.print('sample midi base dur vol')
    fmt = '%6d %4d %4d %3d %3.1f'
    for sample_idx, midi_def in midi_mapping.items():
        SP.print(fmt, (sample_idx,) + tuple(midi_def))
    SP.leave()

    notes_per_channel = sort_groupby(notes, lambda n: n.col_idx)
    notes_per_channel = [list(grp) for (_, grp) in notes_per_channel]
    notes_per_channel = [list(midi_notes(notes, midi_mapping))
                         for notes in notes_per_channel]
    notes = sorted(sum(notes_per_channel, []))

    # Group by column (9 for drums)
    note_groups = groupby(notes, lambda el: el[0])

    midi = MidiFile(type = 1)
    for i, (channel, note_group) in enumerate(note_groups):
        track = list(midi_notes_to_track(channel, note_group))
        track = MidiTrack(track)
        midi.tracks.append(track)
    midi.save(midi_file)

def mod_file_to_midi_file(mod_file, midi_file,
                          midi_mapping, programs):
    mod = load_file(mod_file)
    rows = linearize_rows(mod)

    # Get volumes
    volumes = [header.volume for header in mod.sample_headers]

    notes = list(rows_to_notes(rows, volumes))

    # Generate midi mapping if needed.
    if midi_mapping == 'auto':
        props = sample_props(mod, notes)
        samples = [(sample_idx, props.is_percussive)
                   for (sample_idx, props) in props]
        midi_mapping = assign_instruments(samples, programs)


    notes_to_midi_file(notes, midi_file, midi_mapping)

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

def cache_file_to_midi_file(cache_file, midi_file,
                            code_index, code_length,
                            programs):
    seq = load_cache(cache_file)
    if code_index == 'random':
        while True:
            code_index = randrange(len(seq) - code_length)
            subseq = seq[code_index:code_index + code_length]

            long_jump = any(arg >= 64 for (cmd, arg) in subseq
                           if cmd == INSN_JUMP)
            if not (INSN_PROGRAM, 0) in subseq and not long_jump:
                break
    else:
        code_index = int(code_index)
        subseq = seq[code_index:code_index + code_length]
    SP.print('Selected index %d from cache of length %d.',
             (code_index, len(seq)))

    # Deduce initial note and duration
    print(subseq)
    note = guess_initial_note(subseq)
    sample = guess_initial_sample(subseq)
    duration = guess_initial_duration(subseq)
    assert duration is not None

    SP.print('Guessed initial note %d, sample %d, and duration %d.',
             (note, sample, duration))

    # We produce notes from the mycode
    notes = list(mycode_to_notes(subseq, note, sample, duration))

    groups = [(n.sample_idx, n.note_idx) for n in notes]
    groups = sort_groupby(groups, lambda n: n[0])
    samples = [(idx, len(set(grp)) == 1) for (idx, grp) in groups]
    midi_mapping = assign_instruments(samples, programs)
    notes_to_midi_file(notes, midi_file, midi_mapping)

Programs = namedtuple('Programs', ['melodic', 'percussive'])

def parse_programs(s):
    parts = s.split(':')
    mels, percs = parts[:-1], parts[-1]
    perc_progs = [int(p) for p in percs.split(',')]
    mel_progs = [[int(p) for p in m.split(',')] for m in mels]
    return Programs(mel_progs, perc_progs)

def main():
    args = docopt(__doc__, version = 'MIDI file generator 1.0')
    SP.enabled = args['--verbose']

    mod_file = args['<mod>']
    cache_file = args['<cache>']
    midi_file = args['--output']
    programs = parse_programs(args['--programs'])
    if mod_file:
        midi_mapping = args['--midi-mapping']
        if midi_mapping != 'auto':
            with open(midi_mapping, 'r') as f:
                midi_mapping = load(f)
            midi_mapping = {int(k) : v for (k, v) in midi_mapping.items()}
        mod_file_to_midi_file(mod_file, midi_file,
                              midi_mapping, programs)
    elif cache_file:
        cache_file = Path(cache_file)
        code_length = int(args['--length'])
        code_index = args['--index']
        cache_file_to_midi_file(cache_file, midi_file,
                                code_index, code_length,
                                programs)

if __name__ == '__main__':
    main()
