from argparse import ArgumentParser, FileType
from collections import defaultdict
from itertools import groupby
from json import load
from mido import Message, MidiFile, MidiTrack
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module

# Default midi index for the note C-1.
MIDI_C1_IDX = 24

# Default convert to MIDI instrument 1, with a C-1 note base.
DEFAULT_CONV = (1, MIDI_C1_IDX, 4, 1.0)

def mod_note_volume(mod, cell):
    if cell.effect_cmd == 12:
        return (cell.effect_arg1 << 4) + cell.effect_arg2
    return mod.sample_headers[cell.sample_idx - 1].volume

def mod_notes(mod, rows, col_idx):
    tempo = DEFAULT_TEMPO
    speed = DEFAULT_SPEED
    for row_idx, row in enumerate(rows):
        tempo, speed = update_timings(row, tempo, speed)
        row_time_ms = int(calc_row_time(tempo, speed) * 1000)

        cell = row[col_idx]
        period = cell.period
        if period == 0 or cell.sample_idx == 0:
            continue
        period_idx = period_to_idx(period)
        volume = mod_note_volume(mod, cell)
        yield (cell.sample_idx, row_idx, col_idx,
               period_idx, volume, row_time_ms)

def note_duration(notes, i, row_idx, note_dur):
    if i < len(notes) - 1:
        next_row_idx = notes[i + 1][1]
        return min(next_row_idx - row_idx, note_dur)
    return note_dur

def midi_notes(conv_info, notes):
    offset_ms = 0
    last_row_idx = 0
    for i in range(len(notes)):
        sample_idx, row_idx, col_idx, period_idx, volume, row_time_ms \
            = notes[i]
        row_delta = row_idx - last_row_idx

        # Update time
        offset_ms += row_delta * row_time_ms
        last_row_idx = row_idx

        program, midi_idx_base, note_dur, vol_adj = conv_info[sample_idx]
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
            midi_idx = midi_idx_base + period_idx
        velocity = int(min((volume / 64) * 127 * vol_adj, 127))

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

def main():
    parser = ArgumentParser(description='Module stripper')
    parser.add_argument('--json', type = FileType('r'))
    parser.add_argument('module', type = FileType('rb'))
    parser.add_argument('midi', type = FileType('wb'))
    args = parser.parse_args()
    args.midi.close()

    with args.module as inf:
        mod = Module.parse(inf.read())

    if not args.json:
        conv_info = {
            idx : [1, MIDI_C1_IDX, 4, 1.0] for idx in range(1, 32)
        }
    else:
        conv_info = load(args.json)
        # convert to integer keys
        conv_info = {int(k) : v for (k, v) in conv_info.items()}

    rows = linearize_rows(mod)
    print(rows_to_string(rows))

    # Extract mod notes
    notes_per_channel = [list(mod_notes(mod, rows, i)) for i in range(4)]

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
