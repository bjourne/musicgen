# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, namedtuple
from musicgen.code_utils import CODE_MIDI_MAPPING, INSN_END
from musicgen.corpus import load_index
from musicgen.generation import notes_to_audio_file
from musicgen.parser import UnsupportedModule, load_file
from musicgen import pcode, scode
from musicgen.pcode import mod_to_pcode, pcode_to_notes
from musicgen.scode import mod_file_to_scode
from musicgen.utils import (SP, CharEncoder,
                            flatten, file_name_for_params,
                            load_pickle_cache, save_pickle)
from random import randrange, shuffle
import numpy as np

CodeInfo = namedtuple('CodeInfo', ['to_code_fn', 'to_notes_fn',
                                   'pause', 'metadata_fn'])

# TODO: Fix scode
CODE_TYPES = {
    'pcode_abs' : CodeInfo(lambda m: mod_to_pcode(m, False),
                           lambda c: pcode_to_notes(c, False),
                           pcode.pause(),
                           pcode.metadata),
    'pcode_rel' : CodeInfo(lambda m: mod_to_pcode(m, True),
                           lambda c: pcode_to_notes(c, True),
                           pcode.pause(),
                           pcode.metadata),
    'scode_abs' : CodeInfo(lambda m: mod_file_to_scode(m, False),
                           None,
                           scode.pause(),
                           None),
    'scode_rel' : CodeInfo(lambda m: mod_file_to_scode(m, True),
                           None,
                           scode.pause(),
                           None)
}

def is_learnable(meta):
    n_toks = meta['n_toks']
    pitch_range = meta['pitch_range']
    n_notes = meta['n_notes']
    n_unique_notes = meta['n_unique_notes']
    if n_toks < 64:
        SP.print('To few tokens, %d.' % n_toks)
        return False
    if n_notes < 16:
        SP.print('To few notes, %d.' % n_notes)
        return False
    if n_unique_notes < 4:
        SP.print('To few unique melodic notes, %d.' % n_unique_notes)
        return False
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        return False
    return True

def mod_file_to_code_w_progress(i, n, file_path, info):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, file_path))
    try:
        mod = load_file(file_path)
    except UnsupportedModule:
        SP.print('Unsupported module format.')
        SP.leave()
        return None
    code = list(info.to_code_fn(mod)) + [(INSN_END, 0)]
    meta = info.metadata_fn(code)
    if not is_learnable(meta):
        SP.leave()
        return None
    SP.leave()
    return code

def transpose_code(code):
    pitches = [p for (c, p) in code if c == 'P']

    n_versions = 36 - max(pitches)
    assert n_versions > 0
    SP.print('%d transpositions.' % n_versions)

    codes = [[(c, p) if c != 'P' else (c, p + i)
              for (c, p) in code]
             for i in range(n_versions)]
    for i, code in enumerate(codes):
        pitches = [p for (c, p) in code if c == 'P']
        assert min(pitches) == i
        assert max(pitches) <= 35
    return codes

def load_and_encode_mod_files(mod_files, info):
    encoder = CharEncoder()
    arrs = []
    n = len(mod_files)
    for i, p, in enumerate(sorted(mod_files)):
        code = mod_file_to_code_w_progress(i + 1, n, p, info)
        if not code:
            continue
        codes = transpose_code(code)
        codes = [encoder.encode_chars(c, True) for c in codes]
        arrs.append((p.name, codes))
    return encoder, arrs

def build_cache(path, shuffle_file, mods, info):
    mod_files = [path / mod.genre / mod.fname for mod in mods]

    # Cache the shuffle to make trained models more comparable.
    shuffle_path = path / shuffle_file
    def rebuild_fn():
        indices = list(range(n))
        shuffle(indices)
        return indices
    indices = load_pickle_cache(shuffle_path, rebuild_fn)
    encoder, arrs = load_and_encode_mod_files(mod_files, info)

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
        self.encoder, self.arrs = \
            load_and_encode_mod_files([p], self.info)

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

def flatten_training_data(td):
    arrs = flatten([arrs for (_, arrs) in td.arrs])

    # Temporary hack
    end_idx = td.encoder.encode_char((INSN_END, 0), True)
    end_arr = np.array([end_idx])
    padded = []
    for arr in arrs:
        padded.append(arr)
        padded.append(end_arr)
    return np.concatenate(padded)

def print_histogram(td):
    seq = flatten_training_data(td)
    unique, counts = np.unique(seq, return_counts = True)
    ix_counts = dict(zip(unique, counts))
    ch_counts = {td.encoder.decode_char(ix) : cnt
                 for (ix, cnt) in ix_counts.items()}
    total = sum(ch_counts.values())
    SP.header('%d TOKENS %d TYPES' % (total, len(ch_counts)))
    for (cmd, arg), cnt in sorted(ch_counts.items()):
        SP.print('%s %3d %10d' % (cmd, arg, cnt))
    SP.leave()

def load_training_data(code_type, path):
    td = TrainingData(code_type)
    if path.is_dir():
        td.load_disk_cache(path, 150)
        train, valid, test = td.split_3way(0.8, 0.1)
    else:
        td.load_mod_file(path)
        train = valid = test = td
    print_histogram(td)
    return train, valid, test

def pick_song_fragment(seq, i, n, end_ix):
    if i != 'random':
        i = int(i)
        return i, seq[i:i + n]
    while True:
        i = randrange(len(seq) - n)
        fragment = seq[i:i + n]
        if end_ix in fragment:
            SP.print('EOS in fragment, regenerating.')
            continue
        return i, fragment

if __name__ == '__main__':
    from pathlib import Path
    from sys import argv
    SP.enabled = True
    load_training_data('pcode_abs', Path(argv[1]))
