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

def split_train_validate_test(seq, train_frac, validate_frac):
    n_seq = len(seq)
    n_train = int(n_seq * train_frac)
    n_validate = int(n_seq * validate_frac)
    n_test = n_seq - n_train - n_validate
    train = seq[:n_train]
    validate = seq[n_train:n_train + n_validate]
    test = seq[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))
    return train, validate, seq

class CharEncoder:
    def __init__(self):
        self.ch2ix = {}
        self.ix2ch = {}
        self.next_idx = 0

    def decode_char(self, ix):
        return self.ix2ch[ix]

    def decode_chars(self, seq):
        return [self.decode_char(ix) for ix in seq]

    def encode_char(self, ch, add_missing):
        if ch not in self.ch2ix:
            if not add_missing:
                raise ValueError('%s missing!' % ch)
            self.ch2ix[ch] = self.next_idx
            self.ix2ch[self.next_idx] = ch
            self.next_idx += 1
        return self.ch2ix[ch]

    def encode_chars(self, chars, add_missing):
        return [self.encode_char(ch, add_missing) for ch in chars]
