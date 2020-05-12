# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Random utils.
from itertools import groupby
from keras.utils import Sequence, to_categorical
import numpy as np


class StructuredPrinter:
    def __init__(self, enabled):
        self.indent = 0
        self.enabled = enabled

    def print_indented(self, text):
        if self.enabled:
            print(' ' * self.indent + text)

    def header(self, name, fmt = None, args = None):
        if fmt is not None:
            self.print_indented('* %s %s' % (name, fmt % args))
        else:
            self.print_indented('* %s' % name)
        self.indent += 2

    def print(self, fmt, args = None):
        if args is not None:
            str = fmt % args
        else:
            str = fmt
        self.print_indented(str)

    def leave(self):
        self.indent -= 2

SP = StructuredPrinter(False)

def sort_groupby(seq, keyfun):
    return groupby(sorted(seq, key = keyfun), keyfun)

def parse_comma_list(seq):
    return [int(e) for e in seq.split(',')]

class OneHotGenerator(Sequence):
    def __init__(self, seq, batch_size, win_size, vocab_size):
        self.seq = seq
        self.batch_size = batch_size
        self.win_size = win_size
        self.vocab_size = vocab_size

    def __len__(self):
        n_windows = len(self.seq) - self.win_size
        return int(np.ceil(n_windows / self.batch_size))

    def __getitem__(self, i):
        base = i * self.batch_size

        # Fix running over the dge.
        n_windows = len(self.seq) - self.win_size
        batch_size = min(n_windows - base, self.batch_size)

        X = np.zeros((batch_size, self.win_size, self.vocab_size),
                     dtype = np.bool)
        Y = np.zeros((batch_size, self.vocab_size),
                     dtype = np.bool)
        for i in range(batch_size):
            for j in range(self.win_size):
                X[i, j, self.seq[base + i + j]] = 1
            Y[i, self.seq[base + i + self.win_size]] = 1
        return X, Y

# This algorithm is to slow to be practical.
def find_longest_repeating_non_overlapping_subseq(seq):
    candidates = []
    for i in range(len(seq)):
        candidate_max = len(seq[i + 1:]) // 2
        for j in range(1, candidate_max + 1):
            candidate, remaining = seq[i:i + j], seq[i + j:]
            n_reps = 1
            len_candidate = len(candidate)
            while remaining[:len_candidate] == candidate:
                n_reps += 1
                remaining = remaining[len_candidate:]
            if n_reps > 1:
                candidates.append((seq[:i], n_reps, candidate, remaining))
    if not candidates:
        return (type(seq)(), 1, seq, type(seq)())

    def score_candidate(candidate):
        intro, reps, loop, outro = candidate
        return reps - len(intro) - len(outro)
    return sorted(candidates, key = score_candidate)[-1]


if __name__ == '__main__':
    seq = 'EEEFGAFFGAFFGAFCD'
    print(find_longest_repeating_non_overlapping_substring(seq))
    seq = 'ACCCCCCCCCA'
    print(find_longest_repeating_non_overlapping_substring(seq))
    seq = 'ABCD'
    print(find_longest_repeating_non_overlapping_substring(seq))
