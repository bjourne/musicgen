# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Random utils.
from collections import Counter
from functools import reduce
from itertools import groupby
from operator import iconcat
from pickle import dump, load
from time import time
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
            s = fmt % args
        else:
            s = str(fmt)
        self.print_indented(s)

    def leave(self):
        self.indent -= 2

SP = StructuredPrinter(False)

def sort_groupby(seq, keyfun):
    return groupby(sorted(seq, key = keyfun), keyfun)

def flatten(seq):
    return reduce(iconcat, seq, [])

def parse_comma_list(seq):
    return [int(e) for e in seq.split(',')]

def find_subseq(seq, subseq):
    '''Find indices to occurrences of subseq in seq. Oddly enough this
    function doesn't exist in Python's standard library.
    '''
    l = len(subseq)
    for i in range(len(seq) - l + 1):
        if seq[i:i+l] == subseq:
            yield i

########################################################################
# Pickle caching
########################################################################
def encode_training_sequence(seq):
    ix2ch = sorted(set(seq))
    assert len(ix2ch) < 256
    ch2ix = {c : i for i, c in enumerate(ix2ch)}
    seq = np.array([ch2ix[ch] for ch in seq], dtype = np.uint8)
    return ix2ch, ch2ix, seq

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
        elif type(p) == bool:
            return '%s'
        else:
            return '%.2f'
        return '%%0%dd' % n
    strs = [param_to_fmt(p) % p for p in params]
    return '%s-%s.%s' % (base, '-'.join(strs), ext)

def load_pickle(pickle_path):
    assert pickle_path.exists()
    with open(pickle_path, 'rb') as f:
        return load(f)

def save_pickle(pickle_path, obj):
    with open(pickle_path, 'wb') as f:
        dump(obj, f)

def load_pickle_cache(cache_path, rebuild_fun):
    if not cache_path.exists():
        start = time()
        SP.print('Building cache at %s.' % cache_path)
        save_pickle(cache_path, rebuild_fun())
        delta = time() - start
        SP.print('Cache built in %.2f seconds.' % delta)
    else:
        SP.print('Loading cache from %s.' % cache_path)
    return load_pickle(cache_path)

def analyze_code(ix2ch, seq):
    counts = Counter(seq)
    total = sum(counts.values())
    SP.header('%d TOKENS %d TYPES' % (total, len(counts)))
    for ix, cnt in sorted(counts.items()):
        cmd, arg = ix2ch[ix]
        SP.print('%s %3d %10d' % (cmd, arg, cnt))
    SP.leave()
