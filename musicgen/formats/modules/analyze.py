# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, defaultdict, namedtuple
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file

HEADER = [
    'Sample',
    'Common freq',
    'N. Notes', 'N. Uniq',
    'Longest rep',
    'Size',
    'Note dur',
    'Repeat pct',
    'Percussive?'
]
ROW_FORMAT = ['%2d',
              '%.2f',
              '%3d', '%2d',
              '%3d', '%5d', '%2d',
              '%.2f',
              '%s']

SampleProps = namedtuple('SampleProps', [
    'most_common_freq',
    'n_notes',
    'n_unique_notes',
    'len_longest_repeating_seq',
    'size',
    'note_duration',
    'repeat_pct',
    'is_percussive'])

def relative_counter(seq):
    counter = Counter(seq)
    tot = len(seq)
    return {el : freq / tot for el, freq in counter.items()}

def bin_distance(dist):
    if dist >= 16:
        return 16
    elif dist >= 8:
        return 8
    elif dist >= 4:
        return 4
    elif dist >= 3:
        return 3
    elif dist >= 2:
        return 2
    return 1

def get_sample_props(header, notes, distances):
    assert len(distances) == len(notes)

    # Get all piches
    pitches = [n.note_idx for n in notes]

    # Homogenize distances
    distances = [bin_distance(d) for d in distances]
    counter = relative_counter(distances)
    distances = [d for (d, freq) in counter.items() if freq >= 0.05]
    duration = max(distances)

    # Compute header size and repeat pct
    size = header.size * 2
    if header.repeat_len > 2:
        repeat_pct = header.repeat_from / header.size
    else:
        repeat_pct = 1.0

    # Compute the length of the longest sequence of repeating pitches.
    longest_rep = max(len(list(group))
                      for (p, group) in groupby(pitches))

    counter = relative_counter(pitches)
    most_common_freq = max(counter.values())
    n_unique = len(counter)

    # Guess whether the sample is for a percussive instrument.
    is_percussive = False
    if repeat_pct == 1.0:
        if most_common_freq > 0.9 and n_unique <= 2:
            is_percussive = True
        if most_common_freq > 0.8 and longest_rep >= 50:
            is_percussive = True

    return SampleProps(most_common_freq,
                       len(notes),
                       n_unique,
                       longest_rep,
                       size,
                       duration,
                       repeat_pct,
                       is_percussive)

def sample_props(mod, notes):
    # Note distances
    distances = defaultdict(list)
    for col_idx, group in sort_groupby(notes, lambda n: n.col_idx):
        last_row = None
        for note in group:
            if last_row is None:
                delta = 0
            else:
                delta = note.row_idx - last_row
            assert delta >= 0
            distances[note.sample_idx].append(delta)
            last_row = note.row_idx

    # Group by sample
    grouped = sort_groupby(notes, lambda n: n.sample_idx)
    grouped = [(sample, get_sample_props(
        mod.sample_headers[sample - 1],
        list(group), distances[sample]))
               for sample, group in grouped]
    return grouped

def main():
    from argparse import ArgumentParser, FileType
    from termtables import print as tt_print
    from termtables.styles import ascii_booktabs, booktabs

    parser = ArgumentParser(
        description = 'Analyze MOD files')
    parser.add_argument('module', type = FileType('rb'))
    args = parser.parse_args()
    args.module.close()

    mod = load_file(args.module.name)
    rows = linearize_rows(mod)
    notes = list(notes_in_rows(mod, rows))
    props = sample_props(mod, notes)

    # Make a table
    rows = [(sample,) + p for (sample, p) in props]
    rows = [[fmt % col for (col, fmt) in zip(row, ROW_FORMAT)]
            for row in rows]
    tt_print(rows,
             padding = (0, 0, 0, 0),
             alignment = 'rrrrrrrrc',
             style = ascii_booktabs,
             header = HEADER)

if __name__ == '__main__':
    main()
