# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Random utils.
from functools import reduce
from itertools import groupby
from operator import iconcat
from pickle import dump, load
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

def flatten(seq):
    return reduce(iconcat, seq, [])

def parse_comma_list(seq):
    return [int(e) for e in seq.split(',')]

def load_pickle(pickle_path):
    assert pickle_path.exists()
    with open(pickle_path, 'rb') as f:
        return load(f)

def save_pickle(pickle_path, obj):
    with open(pickle_path, 'wb') as f:
        dump(obj, f)

def file_name_for_params(base, ext, params):
    def param_to_fmt(p):
        if type(p) == int:
            n = len(str(p))
            if n > 5:
                n = 10
            elif n > 3:
                n = 5
            else:
                n = 3
        else:
            return '%.2f'
        return '%%0%dd' % n
    strs = [param_to_fmt(p) % p for p in params]
    return '%s_%s.%s' % (base, '-'.join(strs), ext)


# This algorithm is to slow to be practical.
def find_best_split(seq):
    '''
    Real name: find_longest_repeating_non_overlapping_subseq
    '''
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

# https://stackoverflow.com/questions/61758735/find-longest-adjacent-repeating-non-overlapping-substring
def find_best_split2(s):
    from collections import deque

    # There are zcount equal characters starting
    # at index starti.
    def update(starti, zcount):
        nonlocal bestlen
        zcount += width
        while zcount >= bestlen:
            count = zcount - zcount % width
            numreps = count // width
            if numreps > 1 and count >= bestlen:
                if count > bestlen:
                    results.clear()
                results.append((starti, width, numreps))
                bestlen = count
            zcount -= 1
            starti += 1

    bestlen, results = 0, []
    if not s:
        return 0, len(s), 1
    t = deque(s)
    for width in range(1, len(s) // 2 + 1):
        t.popleft()
        zcount = 0
        for i, (a, b) in enumerate(zip(s, t)):
            if a == b:
                if not zcount: # new run starts here
                    starti = i
                zcount += 1
            # else a != b, so equal run (if any) ended
            elif zcount:
                update(starti, zcount)
                zcount = 0
        if zcount:
            update(starti, zcount)
    if not results:
        return 0, len(s), 1
    return sorted(results, key = lambda x: x[2])[-1]

def test_find_longest():
    seq = 'EEEFGAFFGAFFGAFCD'
    assert find_best_split(seq) == ('EEE', 3, 'FGAF', 'CD')
    seq = 'ACCCCCCCCCA'
    assert find_best_split(seq) == ('A', 9, 'C', 'A')
    seq = 'ABCD'
    assert find_best_split(seq) == ('', 1, 'ABCD', '')
    seq = 'BAMBAMBAMBAM'
    assert find_best_split(seq) == ('', 4, 'BAM', '')
    res = find_best_split2([
        'P2', 'P2', 'P2',
        'P0', 'P0', 'P0',
        'P-2', 'P-2', 'P-2',
        'P0', 'P0', 'P0'])
    assert res == (9, 1, 3)

if __name__ == '__main__':
    test_find_longest()
