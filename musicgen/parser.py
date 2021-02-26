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
    'integrity' / Check(0 <= this.n_orders <= 128)
)

def pattern_count_stk(this):
    '''This is correct unless the MOD uses hidden patterns.'''
    return max(this.pattern_table[:this.n_orders]) + 1

ModuleSTK = Struct(
    'title' / MyPaddedString(20, 'utf-8', 'replace'),
    'sample_headers' / Array(15, SampleHeader),
    'n_orders' / Byte,
    'restart_pos' / Byte,
    'pattern_table' / Array(128, Byte),
    'patterns' / Array(pattern_count_stk, Pattern),
    'samples' / Array(15, SampleData),
    'integrity' / Check(0 <= this.n_orders <= 128)
)

class UnsupportedModule(ValueError):
    pass

class PowerPackerModule(UnsupportedModule):
    pass

class XPKModule(UnsupportedModule):
    pass

class IceTrackerModule(UnsupportedModule):
    pass

def load_file(fname):
    with open(fname, 'rb') as f:
        arr = bytearray(f.read())

    # Read the four byte file signature. If it is PP20 or XPKF, the
    # MOD is compressed and we throw a suitable exception.
    first4 = arr[:4].decode('utf-8', 'ignore')
    if first4 == 'PP20':
        raise PowerPackerModule()
    elif first4 == 'XPKF':
        raise XPKModule()

    # The magic string at offset 1080 determines type of module. If it
    # one of the listed signatures it is a standard ProTracker
    # compatible module with 31 samples.
    signatures_4chan = {
        # Standard ones
        '4CHN', 'M.K.', 'FLT4', 'M!K!', 'M&K!',
        # Found in flight_of_grud.mod
        'FEST',
        # judgement_day_gvine.mod
        'LARD',
        # kingdomofpleasure.mod
        'NSMS'
    }
    magic = arr[1080:1084].decode('utf-8', 'ignore')
    if magic in signatures_4chan:
        return Module.parse(arr)

    # If the second magic string at offset 1464 is MTN\0 or IT10 it is
    # a SoundTracker 2.6 or Ice Tracker module which is incompatible
    # with ProTracker. Handling the format is not too hard but since
    # it is very uncommon it is not worth the bother.
    magic2 = arr[1464:1468].decode('utf-8', 'ignore')
    if magic2 in {'MTN\0', 'IT10'}:
        raise IceTrackerModule()

    # Otherwise if the magic is not printable we assume that it is a
    # 15 sample original Sound Tracker module.
    if not magic.isprintable():
        return ModuleSTK.parse(arr)

    # Unknown module type. Bail out.
    raise UnsupportedModule(f'Unknown magic "{magic}"!')

def save_file(fname, mod):
    if type(mod) == dict:
        samples = mod['samples']
    else:
        samples = mod.samples
    cls = Module if len(samples) == 31 else ModuleSTK
    with open(fname, 'wb') as f:
        f.write(cls.build(mod))
