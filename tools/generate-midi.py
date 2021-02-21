# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""MIDI file generator

Usage:
    generate-midi.py [-hv] [--programs=<seq>] module
        [--midi-mapping=<json>] <mod>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --midi-mapping=<json>  instrument mapping [default: auto]
    --programs=<seq>       melodic and percussive programs
                           [default: 1,36:40,36,31]
"""
from docopt import docopt
from json import load
from musicgen.analyze import sample_props
from musicgen.generation import (assign_instruments, notes_to_midi_file,
                                 parse_programs)
from musicgen.parser import load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.utils import SP
from pathlib import Path

def main():
    args = docopt(__doc__, version = 'MIDI file generator 1.0')
    SP.enabled = args['--verbose']

    # Parse
    mod_file = args['<mod>']
    programs = parse_programs(args['--programs'])

    mod_file = Path(mod_file)
    midi_mapping = args['--midi-mapping']
    if midi_mapping != 'auto':
        with open(midi_mapping, 'r') as f:
            midi_mapping = load(f)
        midi_mapping = {int(k) : v
                        for (k, v) in midi_mapping.items()}

    mod = load_file(mod_file)
    subsongs = linearize_subsongs(mod, 1)
    volumes = [header.volume for header in mod.sample_headers]
    for idx, (_, rows) in enumerate(subsongs):
        notes = rows_to_mod_notes(rows, volumes)
        if midi_mapping == 'auto':
            props = sample_props(mod, notes)
            samples = [(sample_idx, props.is_percussive,
                        props.note_duration)
                       for (sample_idx, props) in props.items()]
            midi_mapping = assign_instruments(samples, programs)
        midi_file = 'test-%02d.mid' % idx
        notes_to_midi_file(notes, midi_file, midi_mapping)

if __name__ == '__main__':
    main()
