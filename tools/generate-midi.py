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
from docopt import docopt
from itertools import groupby, takewhile
from json import load
from musicgen.analyze import sample_props
from musicgen.generation import (assign_instruments,
                                 mycode_to_midi_file,
                                 notes_to_midi_file,
                                 parse_programs)
from musicgen.mycode import (INSN_JUMP,
                             INSN_PROGRAM,
                             load_cache,
                             mycode_to_notes)
from musicgen.parser import load_file
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.utils import SP, sort_groupby
from pathlib import Path
from random import randrange

def mod_file_to_midi_file(mod_file, midi_file,
                          midi_mapping, programs):
    mod = load_file(mod_file)
    rows = linearize_rows(mod)

    # Get volumes
    volumes = [header.volume for header in mod.sample_headers]

    notes = rows_to_mod_notes(rows, volumes)

    # Generate midi mapping if needed.
    if midi_mapping == 'auto':
        props = sample_props(mod, notes)
        samples = [(sample_idx, props.is_percussive, props.note_duration)
                   for (sample_idx, props) in props]
        midi_mapping = assign_instruments(samples, programs)

    notes_to_midi_file(notes, midi_file, midi_mapping)

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
            if long_jump:
                SP.print('Long jump in seq.')
                continue
            if (INSN_PROGRAM, 0) in subseq:
                SP.print('Program start in seq.')
                continue
            break
    else:
        code_index = int(code_index)
        subseq = seq[code_index:code_index + code_length]
    SP.print('Selected index %d from cache of length %d.',
             (code_index, len(seq)))

    mycode_to_midi_file(subseq, midi_file, programs)

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
