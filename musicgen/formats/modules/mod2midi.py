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

def mod_note_volume(mod, cell):
    if cell.effect_cmd == 12:
        return (cell.effect_arg1 << 4) + cell.effect_arg2
    return mod.sample_headers[cell.sample_idx - 1].volume

def mod_notes(mod, rows, col_idx):
    tempo = DEFAULT_TEMPO
    speed = DEFAULT_SPEED
    for i, row in enumerate(rows):
        tempo, speed = update_timings(row, tempo, speed)
        row_time_ms = int(calc_row_time(tempo, speed) * 1000)

        cell = row[col_idx]
        period = cell.period
        if period == 0 or cell.sample_idx == 0:
            continue
        period_idx = PERIOD_TO_IDX[period]
        volume = mod_note_volume(mod, cell)
        yield cell.sample_idx, i, period_idx, volume, row_time_ms

def note_duration(notes, i, row_idx, note_dur):
    if i < len(notes) - 1:
        next_row_idx = notes[i + 1][1]
        return min(next_row_idx - row_idx, note_dur)
    return note_dur

def midi_notes(conv_info, notes):
    offset_ms = 0
    last_row_idx = 0
    for i in range(len(notes)):
        sample_idx, row_idx, period_idx, volume, row_time_ms = notes[i]
        row_delta = row_idx - last_row_idx

        # Update time
        offset_ms += row_delta * row_time_ms
        last_row_idx = row_idx

        program, midi_idx_base, note_dur, vol_adj = conv_info[sample_idx]
        note_dur = note_duration(notes, i, row_idx, note_dur)

        # Handling for drums
        if program == -1:
            midi_idx = midi_idx_base
        else:
            midi_idx = midi_idx_base + period_idx
        velocity = min(int((volume / 64) * 127 * vol_adj), 127)

        note_on = offset_ms
        note_off = offset_ms + note_dur * row_time_ms

        yield program, note_on, 1, midi_idx, velocity
        yield program, note_off, 0, midi_idx, 0

def midi_notes_to_track(channel, program, notes):
    if program == -1:
        channel = 9
    else:
        yield Message('program_change', program = program, time = 0,
                      channel = channel)
    prev = 0
    for _, ofs, on_off, midi_idx, velocity in notes:
        rel_ofs = ofs - prev
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
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    args = parser.parse_args()
    args.output.close()

    with args.input as inf:
        mod = Module.parse(inf.read())

    rows = linearize_rows(mod)
    #print(len(rows))
    # print(rows_to_string(rows))

    # Extract mod notes
    notes_per_channel = [list(mod_notes(mod, rows, i)) for i in range(4)]

    # This part should be configurable/learnable.
    lead = 82
    lead_ofs = 36
    conv_info = {
        # 119 = reverse
        # Maybe 17, 82, 83
        1 : (lead, lead_ofs, 3, 1.0),

        2 : (lead, lead_ofs - 12, 4, 1.0),

        3 : (17, 48, 24, 1.0),

        4 : (30, 12, 12, 0.6),
        # 41 = Low Tom 2
        5 : (-1, 41, 2, 0.75),
        6 : (-1, 38, 2, 1.0),

        7 : (118, 12, 2, 0.5),
        # 42
        8 : (-1, 42, 2, 0.75),
        # 43 maybe
        9 : (117, 0, 6, 1.0),

        10 : (-1, 49, 2, 0.75),

        # Note sure...
        11 : (lead, lead_ofs + 6, 3, 1.0),

        12 : (119, 48, 4, 1.0),

        13 : (62, 36, 4, 1.0),
        14 : (62, 24, 4, 1.0),
        15 : (64, 36, 4, 1.0),
        16 : (64, 36, 6, 1.0),
        17 : (64, 24, 4, 1.0),
        18 : (64, 24, 6, 1.0)
    }

    # Convert to midi notes
    notes_per_channel = [list(midi_notes(conv_info, notes))
                         for notes in notes_per_channel]
    notes = sorted(sum(notes_per_channel, []))

    note_groups = groupby(notes, lambda el: el[0])
    midi = MidiFile(type = 1)
    for i, (program, note_group) in enumerate(note_groups):
        track = list(midi_notes_to_track(i, program, note_group))
        track = MidiTrack(track)
        midi.tracks.append(track)
    midi.save(args.output.name)

if __name__ == '__main__':
    main()
