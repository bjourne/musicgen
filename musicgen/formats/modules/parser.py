# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from construct import *

SampleHeader = Struct(
    'name' / Bytes(22),
    'size' / Int16ub,
    'fine_tune' / Byte,
    'volume' / Byte,
    'repeat_from' / Int16ub,
    'repeat_len' / Int16ub,
    'integrity' / Check(0 <= this.volume <= 64)
    )

Cell = BitStruct(
    'sample_hi' / Nibble,
    'period' / BitsInteger(12),
    'sample_lo' / Nibble,
    'effect_cmd' / Nibble,
    'effect_arg1' / Nibble,
    'effect_arg2' / Nibble,
    'sample_idx' / Computed(this.sample_hi * 16 + this.sample_lo)
    )

Pattern = Struct(
    'rows' / Array(64, Array(4, Cell))
    )

def sample_length(x):
    size = x._.sample_headers[x._index].size * 2
    return size - 2 if size else 0

def padding_length(x):
    size = x._.sample_headers[x._index].size
    return 2 if size else 0

SampleData = Struct(
    Padding(padding_length),
    'bytes' / Bytes(sample_length)
)

Module = Struct(
    'title' / PaddedString(20, 'utf-8'),
    'sample_headers' / Array(31, SampleHeader),
    'n_played_patterns' / Byte,
    Padding(1),
    'pattern_table' / Bytes(128),
    'initials' / Const(b'M.K.'),
    'patterns' / Array(max_(this.pattern_table) + 1, Pattern),
    'samples' / Array(31, SampleData))

def load(fname):
    with open(fname, 'rb') as f:
        arr = bytearray(f.read())
    return Module.parse(arr)
