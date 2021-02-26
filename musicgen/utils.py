# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Random utils.
from collections import Counter
from functools import reduce
from gzip import compress, decompress
from itertools import groupby
from operator import iconcat
from pickle import dumps, loads
from termtables import to_string
from time import time

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
        assert self.indent >= 0

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
    with open(pickle_path, 'rb') as f:
        return loads(decompress(f.read()))

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        f.write(compress(dumps(obj)))

def load_pickle_cache(cache_path, rebuild_fun):
    if not cache_path.exists():
        start = time()
        SP.header('BUILDING CACHE %s' % cache_path)
        save_pickle(cache_path, rebuild_fun())
        delta = time() - start
        SP.print('Cache built in %.2f seconds.' % delta)
        SP.leave()
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

def print_term_table(row_fmt, rows, header, alignment):
    def format_col(fmt, col):
        if callable(fmt):
            return fmt(col)
        return fmt % col
    rows = [[format_col(*e) for e in zip(row_fmt, row)] for row in rows]
    s = to_string(rows,
                  header = header,
                  padding = (0, 0, 0, 0),
                  alignment = alignment,
                  style = "            -- ")
    m = len(s.splitlines()[1]) - 2
    print(' ' + '=' * m)
    print(s)
    print(' ' + '=' * m)
