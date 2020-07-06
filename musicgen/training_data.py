from collections import namedtuple
from musicgen.code_utils import CODE_MIDI_MAPPING
from musicgen.corpus import load_index
from musicgen.generation import notes_to_audio_file
from musicgen.pcode import (mod_to_pcode,
                            pcode_long_pause,
                            pcode_short_pause,
                            pcode_to_notes,
                            is_pcode_learnable)
from musicgen.scode import (mod_file_to_scode,
                            scode_to_midi_file,
                            scode_short_pause,
                            scode_long_pause)
from musicgen.tensorflow import sequence_to_samples
from musicgen.utils import (SP, CharEncoder, file_name_for_params,
                            load_pickle_cache, save_pickle)
import numpy as np

CodeInfo = namedtuple('CodeInfo', ['to_code_fn', 'to_notes_fn',
                                   'short_pause', 'long_pause',
                                   'is_learnable_fn'])

# TODO: Fix scode
CODE_TYPES = {
    'pcode_abs' : CodeInfo(lambda m: mod_to_pcode(m, False),
                           lambda c: pcode_to_notes(c, False),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'pcode_rel' : CodeInfo(lambda m: mod_to_pcode(m, True),
                           lambda c: pcode_to_notes(c, True),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'scode_abs' : CodeInfo(lambda m: mod_file_to_scode(m, False),
                           None,
                           scode_short_pause(),
                           scode_long_pause(),
                           None),
    'scode_rel' : CodeInfo(lambda m: mod_file_to_scode(m, True),
                           None,
                           scode_short_pause(),
                           scode_long_pause(),
                           None)
}

def mod_file_to_code_w_progress(i, n, file_path, info):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, file_path))
    try:
        mod = load_file(file_path)
    except UnsupportedModule:
        SP.print('Unsupported module format.')
        SP.leave()
        return None
    code = list(info.to_code_fn(mod))
    if not info.is_learnable_fn(code):
        code = None
    SP.leave()
    return code

def build_cache(path, shuffle_file, mods, info):
    mod_files = [path / mod.genre / mod.fname for mod in mods]
    n = len(mod_files)

    # Cache the shuffle to make trained models more comparable.
    shuffle_path = path / shuffle_file
    def rebuild_fn():
        indices = list(range(n))
        shuffle(indices)
        return indices
    indices = load_pickle_cache(shuffle_path, rebuild_fn)

    encoder = CharEncoder()
    arrs = []
    for i, p, in enumerate(sorted(mod_files)):
        code = mod_file_to_code_w_progress(i + 1, n, p, info)
        if not code:
            continue
        arrs.append((p.name, encoder.encode_chars(code, True)))

    # Shuffle according to indices.
    tmp = [(i, e) for (i, e) in zip(indices, arrs)]
    arrs = [e for (_, e) in sorted(tmp)]

    return encoder, arrs

class TrainingData:
    def __init__(self, code_type):
        if code_type not in CODE_TYPES:
            s = ', '.join(CODE_TYPES)
            raise ValueError('<code-type> must be one of %s' % s)
        self.code_type = code_type
        self.info = CODE_TYPES[code_type]

    def load_disk_cache(self, path, kb_limit):
        index = load_index(path)
        mods = [mod for mod in index.values()
                if (mod.n_channels == 4
                    and mod.format == 'MOD'
                    and mod.kb_size <= kb_limit)]
        size_sum = sum(mod.kb_size for mod in mods)
        real_prefix = 'cache_%s' % self.code_type
        params = size_sum, kb_limit
        cache_file = file_name_for_params(real_prefix, 'pickle', params)
        cache_path = path / cache_file
        shuffle_file = file_name_for_params('shuffle', 'pickle', params)
        def rebuild_fn():
            return build_cache(path, shuffle_file, mods, self.info)
        data = load_pickle_cache(cache_path, rebuild_fn)
        self.encoder, self.arrs = data

    def load_mod_file(self, p):
        code = mod_file_to_code_w_progress(1, 1, p, self.info.to_code_fn)
        self.encoder = CharEncoder()
        self.arrs = [(p.name, self.encoder.encode_chars(code, True))]

    def print_historgram(self):
        codes = [arr for (name, arr) in self.arrs]
        seq = np.concatenate(codes)
        ix_counts = Counter(seq)
        ch_counts = {self.encoder.decode_char(ix) : cnt
                     for (ix, cnt) in ix_counts.items()}
        total = sum(ch_counts.values())
        SP.header('%d TOKENS %d TYPES' % (total, len(ch_counts)))
        for (cmd, arg), cnt in sorted(ch_counts.items()):
            SP.print('%s %3d %10d' % (cmd, arg, cnt))
        SP.leave()

    def split_3way(self, train_frac, valid_frac):
        n_mods = len(self.arrs)
        n_train = int(n_mods * train_frac)
        n_valid = int(n_mods * valid_frac)
        n_test = n_mods - n_train - n_valid

        parts = (self.arrs[:n_train],
                 self.arrs[n_train:n_train + n_valid],
                 self.arrs[n_train + n_valid:])
        tds = [TrainingData(self.code_type) for _ in range(3)]
        for td, part in zip(tds, parts):
            td.arrs = part
            td.encoder = self.encoder
        return tds

    def save_code(self, seq, file_path):
        code = self.encoder.decode_chars(seq)
        if file_path.suffix == '.pickle':
            save_pickle(file_path, code)
        else:
            notes = self.info.to_notes_fn(code)
            notes_to_audio_file(notes, file_path, CODE_MIDI_MAPPING, True)

    def flatten(self):
        pause = self.encoder.encode_chars(self.info.long_pause, False)
        padded = []
        for name, arr in self.arrs:
            if len(arr) > 0:
                padded.append(arr)
                padded.append(pause)
        return np.concatenate(padded)

    def to_samples(self, length):
        seq = self.flatten()
        return sequence_to_samples(seq, length)

def load_training_data(code_type, path):
    td = TrainingData(code_type)
    if path.is_dir():
        td.load_disk_cache(path, 150)
        return td.split_3way(0.8, 0.1)
    td.load_mod_file(path)
    return td, td, td
