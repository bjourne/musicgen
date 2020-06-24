# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Utilities for MIDI generation
from collections import namedtuple
from itertools import groupby
from mido import Message, MidiFile, MidiTrack
from musicgen.utils import SP, flatten, sort_groupby
from os import system
from pathlib import Path
from tempfile import mkdtemp

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
    for sample_idx, is_percussive, duration in samples:
        if is_percussive:
            program = programs.percussive[perc_i]
            midi_mapping[sample_idx] = [-1, program, 4, 1.0]
            perc_i = (perc_i + 1) % len(programs.percussive)
        else:
            program, base = programs.melodic[mel_i]
            midi_mapping[sample_idx] = [program, base, duration, 1.0]
            mel_i = (mel_i + 1) % len(programs.melodic)
    return midi_mapping

def mod_notes_to_midi_notes(notes, midi_mapping):
    offset_ms = 0
    last_row_idx = 0
    for n in notes:
        row_delta = n.row_idx - last_row_idx

        # Update time
        offset_ms += row_delta * n.time_ms
        last_row_idx = n.row_idx

        program, midi_idx_base, note_dur, vol_adj \
            = midi_mapping[n.sample_idx]

        # Note duration is the minimum...
        note_dur = min(note_dur, n.duration)

        # -2 indicates filtered notes.
        if program == -2:
            continue

        # Note velocity
        velocity = int(min((n.vol / 64) * 127 * vol_adj, 127))

        # On and off offsets
        note_on = offset_ms
        note_off = offset_ms + note_dur * n.time_ms

        # Clamp the pitch in case the network generates garbage.
        midi_idx = midi_idx_base + n.pitch_idx
        if not 0 <= midi_idx < 120:
            SP.print('Fixing midi note %d.', midi_idx)
            midi_idx = min(max(midi_idx, 0), 120)

        # Drum track/melodic
        if program == -1:
            yield 9, note_on, 1, None, midi_idx_base, velocity
        else:
            yield n.col_idx, note_on, 1, program, midi_idx, velocity
            yield n.col_idx, note_off, 0, program, midi_idx, 0

def midi_notes_to_track(channel, notes):
    prev = 0
    current_program = None
    for _, ofs, on_off, program, midi_idx, velocity in notes:
        rel_ofs = ofs - prev
        if program != current_program:
            current_program = program
            assert program is not None
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
    SP.print('sample midi base dur  vol')
    fmt = '%6d %4d %4d %3d %3.2f'
    for sample_idx, midi_def in midi_mapping.items():
        SP.print(fmt, (sample_idx,) + tuple(midi_def))
    SP.leave()

    notes_per_channel = sort_groupby(notes, lambda n: n.col_idx)
    notes_per_channel = [list(grp) for (_, grp) in notes_per_channel]
    notes_per_channel = [
        list(mod_notes_to_midi_notes(notes, midi_mapping))
        for notes in notes_per_channel]
    notes = sorted(flatten(notes_per_channel))
    SP.print('Produced %d midi notes (on/offs).' % len(notes))

    # Group by column (9 for drums)
    note_groups = groupby(notes, lambda el: el[0])

    tracks = [MidiTrack(list(midi_notes_to_track(channel, note_group)))
              for (channel, note_group) in note_groups]

    midi = MidiFile(type = 1)
    midi.tracks = tracks
    midi.save(midi_file)

def notes_to_audio_file(notes, audio_file, midi_mapping, stereo):
    type = 'stereo' if stereo else 'mono'
    SP.header('%d NOTES TO %s (%s)' % (len(notes), audio_file, type))

    temp_dir = mkdtemp()
    temp_dir = Path(temp_dir)

    if stereo:
        left_notes = [n for n in notes if n.col_idx in {0, 3}]
        right_notes = [n for n in notes if n.col_idx in {1, 2}]
        for notes, side in [(left_notes, 'L'), (right_notes, 'R')]:
            mid = temp_dir / (side + '.mid')
            notes_to_midi_file(notes, mid, midi_mapping)
            system('timidity %s -OwM' % mid)
        SP.print('Generating stereo output using sox.')
        fmt = 'sox -M -c 1 %s -c 1 %s %s'
        system(fmt % (temp_dir / 'L.wav', temp_dir / 'R.wav', audio_file))
    else:
        mid = temp_dir / 't.mid'
        notes_to_midi_file(notes, mid, midi_mapping)
        system('timidity %s -OwM' % mid)
        system('sox %s %s' % (temp_dir / 't.wav', audio_file))
    SP.leave()
