# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
'''
Module analyzer
===============
Prints descriptive statistics for modules

Usage:
    module-analyzer.py [options] <mod>

Options:
    -h --help              show this screen
    -v --verbose           print more output
'''
from collections import Counter
from docopt import docopt
from musicgen.analyze import dissonant_chords, sample_props
from musicgen.parser import UnsupportedModule, load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.utils import SP, sort_groupby
from pathlib import Path
from termtables import print as tt_print
from termtables.styles import ascii_booktabs, booktabs

def main():
    args = docopt(__doc__, version = 'MIDI file generator 1.0')
    SP.enabled = args['--verbose']
    file_path = args['<mod>']

    mod = load_file(file_path)
    rows = list(linearize_subsongs(mod, 1))[0][1]
    n_rows = len(rows)
    sample_headers = mod.sample_headers
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    props = sample_props(mod, notes)
    mel_notes = {n for n in notes
                 if not props[n.sample_idx].is_percussive}
    perc_notes = {n for n in notes
                  if props[n.sample_idx].is_percussive}
    pitches = {n.pitch_idx for n in mel_notes}
    n_unique_mel_notes = len(pitches)
    pitch_range = max(pitches) - min(pitches)
    header = [
        '#',
        'MC freq',
        'Notes', 'Uniq',
        'Longest rep',
        'Size',
        'Dur',
        'Repeat pct',
        'Max ringout',
        'Perc?'
    ]
    row_fmt = [
        '%2d',
        '%.2f',
        '%3d', '%2d',
        '%3d', '%5d', '%2d',
        '%.2f',
        '%.2f',
        '%s'
    ]

    # Make a table
    rows = [(sample,) + p for (sample, p) in props.items()]
    rows = [[fmt % col for (col, fmt) in zip(row, row_fmt)]
            for row in rows]
    tt_print(rows,
             padding = (0, 0, 0, 0),
             alignment = 'rrrrrrrrrc',
             style = ascii_booktabs,
             header = header)

    n_chords, n_diss_chords = dissonant_chords(mel_notes)
    diss_frac = n_diss_chords / n_chords if n_chords else 0.0

    header = ['Item', 'Value']
    rows = [
        ['Rows', n_rows],
        ['Melodic notes', len(mel_notes)],
        ['Percussive notes', len(perc_notes)],
        ['Unique melodic notes', n_unique_mel_notes],
        ['Pitch range', pitch_range],
        ['Chords', n_chords],
        ['Chord dissonance', '%.2f' % diss_frac]
    ]
    tt_print(rows,
             padding = (0, 0, 0, 0),
             alignment = 'lr',
             style = ascii_booktabs)

if __name__ == '__main__':
    main()
