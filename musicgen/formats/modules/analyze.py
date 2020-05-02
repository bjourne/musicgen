# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, defaultdict, namedtuple
from itertools import groupby
from musicgen.utils import sort_groupby
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
    'Max ringout',
    'Percussive?'
]
ROW_FORMAT = ['%2d',
              '%.2f',
              '%3d', '%2d',
              '%3d', '%5d', '%2d',
              '%.2f',
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
    'max_ringout',
    'is_percussive'])

def relative_counter(seq):
    counter = Counter(seq)
    tot = len(seq)
    return {el : freq / tot for el, freq in counter.items()}

def bin_duration(dist):
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

def get_sample_props(mod, sample, notes):
    # Get all piches
    pitches = [n.pitch_idx for n in notes]

    # Homogenize durations
    durations = [bin_duration(n.row_duration) for n in notes]
    counter = relative_counter(durations)
    durations = [d for (d, freq) in counter.items() if freq >= 0.05]
    base_duration = max(durations)

    # Compute header size and repeat pct
    header = mod.sample_headers[sample - 1]
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

    # Compute average ringout
    max_ringout = max(n.ringout_duration for n in notes)

    # Guess whether the sample is for a percussive instrument.
    is_percussive = False
    if n_unique <= 2 and max_ringout <= 0.15:
        is_percussive = True
    if repeat_pct == 1.0:
        if most_common_freq > 0.9 and n_unique <= 2:
            if max_ringout < 0.6:
                is_percussive = True
        # If the same note is repeated more than 40 times, it must be
        # percussive. This is ofc completely arbitrary.
        if longest_rep >= 40:
            is_percussive = True

    return SampleProps(most_common_freq,
                       len(notes),
                       n_unique,
                       longest_rep,
                       size,
                       base_duration,
                       repeat_pct,
                       max_ringout,
                       is_percussive)

AnalyzeNote = namedtuple('AnalyzeNote', [
    'sample_idx', 'pitch_idx', 'row_duration', 'ringout_duration'])

def notes_to_analyze_notes(samples, notes):
    for col_idx, group in sort_groupby(notes, lambda n: n.col_idx):
        last_row = None
        for n in group:
            assert 0 <= n.note_idx < 60
            if last_row is None:
                row_duration = 0
            else:
                row_duration = n.row_idx - last_row
            assert row_duration >= 0

            # Compute ringout duration
            freq = FREQS[n.note_idx]

            n_orig = len(samples[n.sample_idx - 1].bytes)
            ringout_s = n_orig * BASE_FREQ / (freq * AMIGA_SAMPLE_RATE)
            yield AnalyzeNote(n.sample_idx, n.note_idx,
                              row_duration, ringout_s)
            last_row = n.row_idx

def sample_props(mod, notes):

    analyze_notes = notes_to_analyze_notes(mod.samples, notes)
    grouped = sort_groupby(analyze_notes, lambda n: n.sample_idx)
    grouped = [(sample, get_sample_props(mod, sample, list(group)))
               for (sample, group) in grouped]
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
             alignment = 'rrrrrrrrrc',
             style = ascii_booktabs,
             header = HEADER)

if __name__ == '__main__':
    main()
