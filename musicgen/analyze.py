# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, namedtuple
from itertools import groupby
from musicgen.defs import AMIGA_SAMPLE_RATE, BASE_FREQ, FREQS
from musicgen.utils import SP, sort_groupby

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
    thresholds = [64, 32, 16, 8, 4, 3, 2]
    for thr in thresholds:
        if dist >= thr:
            return thr
    return 1

def is_percussive(n_pitches, n_unique, n_pitch_classes,
                  max_ringout, repeat_pct,
                  longest_rep, most_common_freq):
    if n_unique <= 2 and max_ringout <= 0.15:
        return True

    # Sample is not repeating
    if repeat_pct == 1.0:
        # Always percussive if only one note is played.
        if n_unique == 1:
            return True

        if most_common_freq > 0.9 and n_unique <= 2 and max_ringout < 0.6:
            return True

        # If the same note is repeated more than 40 times, it must be
        # percussive. This is ofc completely arbitrary.
        if longest_rep >= 40:
            return True
        # Another arbitrary one.
        if n_unique == 3 and max_ringout <= 0.11 and longest_rep >= 23:
            return True

        # This heuristic is "unsafe" but removes a lot of noise.
        if n_unique == 2 and n_pitch_classes <= 1:
            SP.print('Only one pitch class (%d pitches)' % n_pitches)
            return True
    return False


def get_sample_props(mod, sample_idx, notes):
    # Get all piches
    pitches = [n.pitch_idx for n in notes]
    n_pitches = len(pitches)

    # Homogenize durations
    durations = [bin_duration(n.row_duration) for n in notes]
    counter = relative_counter(durations)
    durations = [d for (d, freq) in counter.items() if freq >= 0.05]
    base_duration = max(durations)

    # Compute header size and repeat pct
    header = mod.sample_headers[sample_idx - 1]
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

    # Pitch classes
    n_pitch_classes = len({p % 12 for p in counter})

    # Guess whether the sample is for a percussive instrument.
    is_perc = is_percussive(n_pitches, n_unique, n_pitch_classes,
                            max_ringout, repeat_pct,
                            longest_rep,
                            most_common_freq)

    return SampleProps(most_common_freq,
                       len(notes),
                       n_unique,
                       longest_rep,
                       size,
                       base_duration,
                       repeat_pct,
                       max_ringout,
                       is_perc)

AnalyzeNote = namedtuple('AnalyzeNote', [
    'sample_idx', 'pitch_idx', 'row_duration', 'ringout_duration'])

def notes_to_analyze_notes(samples, notes):
    for n in notes:
        assert 0 <= n.pitch_idx < 60

        # Compute ringout duration
        freq = FREQS[n.pitch_idx]
        n_orig = len(samples[n.sample_idx - 1].bytes)

        # Should be in ms?
        ringout_s = n_orig * BASE_FREQ / (freq * AMIGA_SAMPLE_RATE)
        yield AnalyzeNote(n.sample_idx, n.pitch_idx, n.duration,
                          ringout_s)

def sample_props(mod, notes):
    analyze_notes = notes_to_analyze_notes(mod.samples, notes)
    grouped = sort_groupby(analyze_notes, lambda n: n.sample_idx)
    return {sample : get_sample_props(mod, sample, list(group))
            for (sample, group) in grouped}
