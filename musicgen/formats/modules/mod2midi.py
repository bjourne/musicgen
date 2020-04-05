from argparse import ArgumentParser, FileType
from collections import defaultdict
from itertools import groupby
from mido import Message, MidiFile, MidiTrack
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module

# Default midi index for the note C-1.
MIDI_C1_IDX = 24

# Default convert to MIDI instrument 1, with a C-1 note base.
DEFAULT_CONV = (1, MIDI_C1_IDX, 4, 1.0)

def mod_note_velocity(mod, cell):
    if cell.effect_cmd == 12:
        return (cell.effect_arg1 << 4) + cell.effect_arg2
    return mod.sample_headers[cell.sample_idx - 1].volume

def mod_notes(mod, rows):
    for i, row in enumerate(rows):
        for cell in row:
            if cell.period == 0:
                continue
            velocity = mod_note_velocity(mod, cell)
            yield cell.sample_idx, i, cell.period, velocity

def midi_notes(conv_info, notes):
    row_time_ms = int(calc_row_time(DEFAULT_TEMPO, DEFAULT_SPEED) * 1000)
    for sample_idx, row_idx, period, volume in notes:
        program, midi_idx_base, note_dur, vol_adj = conv_info.get(
            sample_idx, DEFAULT_CONV)
        note_dur_ms = row_time_ms * note_dur
        midi_idx = midi_idx_base + PERIOD_TO_IDX[period]
        velocity = int((volume / 64) * 127 * vol_adj)
        note_on = row_time_ms * row_idx
        note_off = note_on + note_dur_ms
        yield program, note_on, 0, midi_idx, velocity
        yield program, note_off, 1, midi_idx, 0

def midi_notes_to_track(channel, program, notes):
    yield Message('program_change', program = program, time = 0,
                  channel = channel)
    prev = 0
    for _, ofs, on_off, midi_idx, velocity in notes:
        rel_ofs = ofs - prev
        if on_off == 0:
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
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    args = parser.parse_args()
    args.output.close()

    with args.input as inf:
        mod = Module.parse(inf.read())

    rows = linearize_rows(mod)[:64]
    print(rows_to_string(rows))

    # Extract mod notes
    notes = list(mod_notes(mod, rows))

    # This part should be configurable/learnable.
    conv_info = {
        4 : (30, 12, 16, 0.5),
        13 : (62, 36, 4, 1.0),
        14 : (62, 24, 4, 1.0),
    }

    # Convert to midi notes
    notes = list(sorted(midi_notes(conv_info, notes)))

    note_groups = groupby(notes, lambda el: el[0])
    midi = MidiFile(type = 1)
    for i, (program, note_group) in enumerate(note_groups):
        track = list(midi_notes_to_track(i, program, note_group))
        print('track', track)
        midi.tracks.append(track)
    midi.save(args.output.name)

if __name__ == '__main__':
    main()
