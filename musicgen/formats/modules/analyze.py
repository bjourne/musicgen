# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, namedtuple
from itertools import groupby
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file

SampleProps = namedtuple('SampleProps',
                         ['most_common_freq', 'n_unique_notes'])

def get_sample_props(notes):
    notes = [n.note_idx for n in notes]
    counter = Counter(notes)
    sample, freq = counter.most_common(1)[0]
    return SampleProps(freq / len(notes), len(counter))

def sample_props(notes):
    # Group by sample
    notes = sorted(notes, key = lambda n: n.sample_idx)
    grouped = groupby(notes, key = lambda n: n.sample_idx)
    grouped = [(sample, get_sample_props(group))
               for sample, group in grouped]
    return grouped

def is_percussive(props):
    return props.most_common_freq > 0.9 and props.n_unique_notes <= 2

def classify_samples(notes):
    props = sample_props(notes)
    return [(sample, is_percussive(p)) for sample, p in props]

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
    rows = [[sample, '%.3f' % p.common_freq, p.n_unique_notes]
            for (sample, p) in props]
    header = ['Sample', 'Common freq', '# Unique']
    tt_print(rows,
             padding = (0, 0, 0, 0),
             alignment = 'rrr',
             style = ascii_booktabs,
             header = header)

if __name__ == '__main__':
    main()
