# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from collections import defaultdict
from itertools import groupby
from json import load
from mido import Message, MidiFile, MidiTrack
from musicgen.formats.modules import *
from musicgen.formats.modules.analyze import classify_samples
from musicgen.formats.modules.parser import load_file

def note_duration(notes, i, row_idx, note_dur):
    if i < len(notes) - 1:
        next_row_idx = notes[i + 1][1]
        return min(next_row_idx - row_idx, note_dur)
    return note_dur

def midi_notes(conv_info, notes):
    offset_ms = 0
    last_row_idx = 0
    for i in range(len(notes)):
        col_idx, row_idx, sample, note_idx, vol, row_time_ms = notes[i]
        row_delta = row_idx - last_row_idx

        # Update time
        offset_ms += row_delta * row_time_ms
        last_row_idx = row_idx

        program, midi_idx_base, note_dur, vol_adj = conv_info[sample]
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

def generate_conv_info(notes):
    props = classify_samples(notes)
    print(props)
    lookup = {True : [-1, 31, 4, 1.0],
              False : [1, 48, 4, 1.0]}
    return {sample : lookup[is_percussive]
            for (sample, is_percussive) in props}

def main():
    parser = ArgumentParser(description='Module stripper')

    parser.add_argument('module', type = FileType('rb'))
    parser.add_argument('midi', type = FileType('wb'))

    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--json', type = FileType('r'),
        help = 'JSON configuration to guide the conversion')
    group.add_argument(
        '--auto', action = 'store_true',
        help = 'Automatic instrument mapping')

    args = parser.parse_args()
    args.module.close()
    args.midi.close()

    mod = load_file(args.module.name)
    rows = linearize_rows(mod)
    print(rows_to_string(rows))

    # Extract mod notes and sort/groupby channel
    notes = list(notes_in_rows(mod, rows))
    notes = sorted(notes, key = lambda x: x[0])
    notes_per_channel = groupby(notes, key = lambda x: x[0])
    notes_per_channel = [list(grp) for (_, grp) in notes_per_channel]

    # Load or generate conversion
    if args.auto:
        conv_info = generate_conv_info(notes)
    else:
        conv_info = load(args.json)
        conv_info = {int(k) : v for (k, v) in conv_info.items()}

    # Convert to midi notes
    notes_per_channel = [list(midi_notes(conv_info, notes))
                         for notes in notes_per_channel]
    notes = sorted(sum(notes_per_channel, []))

    note_groups = groupby(notes, lambda el: el[0])

    midi = MidiFile(type = 1)
    for i, (channel, note_group) in enumerate(note_groups):
        track = list(midi_notes_to_track(channel, note_group))
        track = MidiTrack(track)
        midi.tracks.append(track)
    midi.save(args.midi.name)

if __name__ == '__main__':
    main()
