# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Utilities for MIDI generation
from collections import namedtuple
from itertools import groupby
from mido import Message, MidiFile, MidiTrack
from musicgen.formats.modules.mycode import mycode_to_notes
from musicgen.utils import SP, sort_groupby

Programs = namedtuple('Programs', ['melodic', 'percussive'])

def parse_programs(s):
    parts = s.split(':')
    mels, percs = parts[:-1], parts[-1]
    perc_progs = [int(p) for p in percs.split(',')]
    mel_progs = [[int(p) for p in m.split(',')] for m in mels]
    return Programs(mel_progs, perc_progs)

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

def notes_to_midi_file(notes, midi_file, midi_mapping):
    SP.print('%d notes to convert.', len(notes))
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

def mycode_to_midi_file(seq, midi_file, programs):
    notes = list(mycode_to_notes(seq))

    groups = [(n.sample_idx, n.note_idx) for n in notes]
    groups = sort_groupby(groups, lambda n: n[0])
    samples = [(idx, len(set(grp)) == 1) for (idx, grp) in groups]
    midi_mapping = assign_instruments(samples, programs)
    notes_to_midi_file(notes, midi_file, midi_mapping)
