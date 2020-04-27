# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, namedtuple
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file

SampleProps = namedtuple('SampleProps', [
    'most_common_freq',
    'n_unique_notes',
    'len_longest_repeating_seq',
    'is_percussive'])

def get_sample_props(notes):
    # Get all piches
    pitches = [n.note_idx for n in notes]

    # Compute the length of the longest sequence of repeating pitches.
    groups = groupby(pitches)
    longest_rep = max(len(list(group)) for (p, group) in groups)

    counter = Counter(pitches)
    sample, freq = counter.most_common(1)[0]
    common_rel_freq = freq / len(pitches)
    n_unique = len(counter)

    is_percussive = ((common_rel_freq > 0.9 and n_unique <= 2)
                     or (common_rel_freq > 0.8 and longest_rep >= 50))

    return SampleProps(common_rel_freq,
                       n_unique,
                       longest_rep,
                       is_percussive)

def sample_props(notes):
    # Group by sample
    notes = sorted(notes, key = lambda n: n.sample_idx)
    grouped = groupby(notes, key = lambda n: n.sample_idx)
    grouped = [(sample, get_sample_props(group))
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
    notes = notes_in_rows(mod, rows)
    props = sample_props(notes)

    # Make a table
    rows = [[sample,
             '%.3f' % p.most_common_freq,
             p.n_unique_notes,
             p.len_longest_repeating_seq,
             p.is_percussive]
            for (sample, p) in props]
    header = ['Sample', 'Common freq.', '#Uniq', 'Len rep.', 'Percussive?']
    tt_print(rows,
             padding = (0, 0, 0, 0),
             alignment = 'rrrrc',
             style = ascii_booktabs,
             header = header)

if __name__ == '__main__':
    main()
