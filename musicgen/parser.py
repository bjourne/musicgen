# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from construct import *
from construct.core import encodingunit

# To improve:
#
#  * Handle incorrect loop starts.
#  * Handle ADPCM decompression.
class MyStringEncoded(StringEncoded):
    def __init__(self, subcon, encoding, errors):
        super().__init__(subcon, encoding)
        self.errors = errors

    def _decode(self, obj, context, path):
        return obj.decode(self.encoding, self.errors)

def MyPaddedString(length, encoding, errors):
    '''Like PaddedString but with the ability to ignore decoding
    errors.'''
    bytes_per_char = encodingunit(encoding)
    null_stripped = NullStripped(GreedyBytes, pad = bytes_per_char)
    fixed_sized = FixedSized(length, null_stripped)
    macro = MyStringEncoded(fixed_sized, encoding, errors)
    def _emitfulltype(ksy, bitwise):
        return dict(size=length, type="strz", encoding=encoding)
    macro._emitfulltype = _emitfulltype
    return macro

SampleHeader = Struct(
    'name' / MyPaddedString(22, 'utf-8', 'replace'),
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

# If the size is zero the sample is empty. It is also empty if the
# size is one because that is a trick players use to allow
# idle-looping. Same logic in libmodplug.
def empty_sample(header):
    return header.size < 2

def sample_length(x):
    '''Length of the sample in bytes.'''
    header = x._.sample_headers[x._index]
    return header.size * 2 if not empty_sample(header) else 0

class EOFPaddedBytes(Bytes):
    def _parse(self, stream, context, path):
        length = self.length(context) if callable(self.length) \
            else self.length
        data = stream.read(length)
        n_padding = length - len(data)
        assert n_padding >= 0
        data += bytes(n_padding)
        return data

SampleData = Struct(
    # Two first bytes are usually silence.
    'bytes' / EOFPaddedBytes(sample_length)
)

Module = Struct(
    'title' / MyPaddedString(20, 'utf-8', 'replace'),
    'sample_headers' / Array(31, SampleHeader),
    'n_orders' / Byte,
    'restart_pos' / Byte,
    'pattern_table' / Bytes(128),
    'initials' / Bytes(4),
    'patterns' / Array(max_(this.pattern_table) + 1, Pattern),
    'samples' / Array(31, SampleData),
    'integrity' / Check(0 <= this.n_orders <= 128))

ModuleSTK = Struct(
    'title' / MyPaddedString(20, 'utf-8', 'replace'),
    'sample_headers' / Array(15, SampleHeader),
    'n_orders' / Byte,
    'restart_pos' / Byte,
    'pattern_table' / Array(128, Byte),
    'patterns' / Array(max_(this.pattern_table) + 1, Pattern),
    'samples' / Array(15, SampleData),
    'integrity' / Check(0 <= this.n_orders <= 128)
    )

class PowerPackerModule(ValueError):
    pass

def load_file(fname):
    with open(fname, 'rb') as f:
        arr = bytearray(f.read())
    if arr[:4].decode('utf-8', 'ignore') == 'PP20':
        raise PowerPackerModule()
    # The magic at offset 1080 determines type of module. If the magic
    # is not a printable ascii string, the module is a Sound Tracker
    # module containing only 15 samples. Otherwise, if the magic is
    # the string "M.K." it is a ProTracker module containing 31
    # samples.
    magic = arr[1080:1084].decode('utf-8', 'ignore')
    signatures_4chan = ['4CHN', 'M.K.', 'FLT4', 'M!K!', 'M&K!']
    if not magic.isprintable():
        return ModuleSTK.parse(arr)
    elif magic in signatures_4chan:
        return Module.parse(arr)
    raise ValueError(f'Unknown magic "{magic}"!')

def save_file(fname, mod):
    if type(mod) == dict:
        samples = mod['samples']
    else:
        samples = mod.samples
    cls = Module if len(samples) == 31 else ModuleSTK
    with open(fname, 'wb') as f:
        f.write(cls.build(mod))
