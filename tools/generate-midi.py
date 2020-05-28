# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""MIDI file generator

Usage:
    generate-midi.py [-hvo MIDI] [--programs=<seq>] module
        [--midi-mapping=<json> --use-mycode] <mod>
    generate-midi.py [-hvo MIDI] [--programs=<seq>] cache
        [--length=<len> --location=<index> --guess] <cache>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    -o FILE --output FILE  output file [default: test.mid]
    --midi-mapping=<json>  instrument mapping [default: auto]
    --use-mycode           use the mycode intermediate format
    --guess                guess initial pitch and duration
    --length=<len>         length of code to sample [default: 100]
    --location=<index>     location in cache file [default: random]
    --programs=<seq>       melodic and percussive programs
                           [default: 1,36:40,36,31]
"""
from docopt import docopt
from json import load
from musicgen.analyze import sample_props
from musicgen.generation import (MYCODE_MIDI_MAPPING,
                                 assign_instruments,
                                 mycode_to_midi_file,
                                 notes_to_midi_file,
                                 parse_programs)
from musicgen.mycode import (INSN_JUMP,
                             INSN_PROGRAM,
                             mycode_to_mod_notes,
                             mod_file_to_mycode)
from musicgen.parser import load_file
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.utils import SP, flatten, load_pickle
from pathlib import Path
from random import randrange

def mod_file_to_midi_file(mod_file, midi_file,
                          midi_mapping, programs):
    mod = load_file(mod_file)
    rows = linearize_rows(mod)

    # Get notes
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)

    # Generate midi mapping if needed.
    if midi_mapping == 'auto':
        props = sample_props(mod, notes)
        samples = [(sample_idx, props.is_percussive, props.note_duration)
                   for (sample_idx, props) in props.items()]
        midi_mapping = assign_instruments(samples, programs)

    notes_to_midi_file(notes, midi_file, midi_mapping)

def mod_file_to_midi_file_using_mycode(mod_file, midi_file):
    # Can't use packing here since it is not exact.
    mycode = mod_file_to_mycode(mod_file, False)
    time_ms = mycode.time_ms
    SP.print('Rowtime %d ms.' % time_ms)
    notes = [mycode_to_mod_notes(seq, i, time_ms, pitch_idx, None)
             for i, (pitch_idx, seq)
             in enumerate(mycode.cols)]
    notes = flatten(notes)
    notes_to_midi_file(notes, midi_file, MYCODE_MIDI_MAPPING)

def random_cache_location(mycode_mods, n_insns):
    long_jump_tok = INSN_JUMP, 64
    stop_tok = INSN_PROGRAM, 0
    while True:
        i = randrange(len(mycode_mods))
        mycode_mod = mycode_mods[i]
        j = randrange(4)
        _, col = mycode_mod.cols[j]
        n = len(col)
        if n - n_insns <= 0:
            SP.print('Only %d instructions in column.', n)
            continue
        k = randrange(n - n_insns)
        subseq = col[k:k + n_insns]
        if not long_jump_tok in subseq and not stop_tok in subseq:
            return i, j, k

def cache_file_to_midi_file(cache_file, midi_file,
                            loc, n_insns,
                            programs,
                            guess):
    mycode_mods = load_pickle(cache_file)
    if loc == 'random':
        mod_idx, col_idx, seq_idx = random_cache_location(mycode_mods,
                                                          n_insns)
    else:
        mod_idx, col_idx, seq_idx = [int(i) for i in loc.split(':')]
        col_idx -= 1

    mycode_mod = mycode_mods[mod_idx]
    pitch_idx, seq = mycode_mod.cols[col_idx]
    seq = seq[seq_idx:seq_idx + n_insns]

    fmt = '%d instructions from "%s" %d:%d:%d.'
    SP.print(fmt, (n_insns, mycode_mod.name,
                   mod_idx, col_idx + 1, seq_idx))
    if guess:
        pitch_idx = None
    mycode_to_midi_file(seq, midi_file, mycode_mod.time_ms, pitch_idx)

def main():
    args = docopt(__doc__, version = 'MIDI file generator 1.0')
    SP.enabled = args['--verbose']

    # Parse
    mod_file = args['<mod>']
    cache_file = args['<cache>']
    midi_file = args['--output']
    programs = parse_programs(args['--programs'])

    if mod_file:
        mod_file = Path(mod_file)
        if args['--use-mycode']:
            mod_file_to_midi_file_using_mycode(mod_file, midi_file)
        else:
            midi_mapping = args['--midi-mapping']
            if midi_mapping != 'auto':
                with open(midi_mapping, 'r') as f:
                    midi_mapping = load(f)
                midi_mapping = {int(k) : v
                                for (k, v) in midi_mapping.items()}
            mod_file_to_midi_file(mod_file, midi_file,
                                  midi_mapping, programs)
    elif cache_file:
        cache_file = Path(cache_file)
        n_insns = int(args['--length'])
        location = args['--location']
        cache_file_to_midi_file(cache_file, midi_file,
                                location, n_insns,
                                programs, args['--guess'])

if __name__ == '__main__':
    main()
