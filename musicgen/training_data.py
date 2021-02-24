# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from collections import Counter, namedtuple
from musicgen import dcode, pcode_abs, pcode_rel, scode_abs, scode_rel
from musicgen.code_utils import (CODE_MIDI_MAPPING, INSN_END, INSN_PITCH,
                                 guess_percussive_instruments)
from musicgen.corpus import load_index
from musicgen.generation import notes_to_audio_file
from musicgen.parser import UnsupportedModule, load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.utils import (SP, CharEncoder,
                            flatten, file_name_for_params,
                            load_pickle_cache, save_pickle)
from random import randrange, shuffle
import numpy as np

CODE_MODULES = {
    'pcode_abs' : pcode_abs,
    'pcode_rel' : pcode_rel,
    'scode_abs' : scode_abs,
    'scode_rel' : scode_rel,
    'dcode' : dcode
}

def is_learnable(meta):
    n_toks = meta['n_toks']
    pitch_range = meta['pitch_range']
    n_notes = meta['n_notes']
    n_unique_notes = meta['n_unique_notes']
    if n_toks < 128:
        SP.print('To few tokens, %d.' % n_toks)
        return False
    if n_notes < 64:
        SP.print('To few notes, %d.' % n_notes)
        return False
    if n_unique_notes < 4:
        SP.print('To few unique melodic notes, %d.' % n_unique_notes)
        return False
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        return False
    return True

def mod_file_to_codes_w_progress(i, n, file_path, code_type):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, file_path))
    try:
        mod = load_file(file_path)
    except UnsupportedModule:
        SP.print('Unsupported module format.')
        SP.leave()
        return
    code_mod = CODE_MODULES[code_type]

    subsongs = list(linearize_subsongs(mod, 1))
    volumes = [header.volume for header in mod.sample_headers]
    for idx, (order, rows) in enumerate(subsongs):
        SP.header('SUBSONG %d' % idx)
        SP.print('Table order: %s' % order)

        notes = rows_to_mod_notes(rows, volumes)
        percussion = guess_percussive_instruments(mod, notes)
        if notes:
            fmt = '%d rows, %d ms/row, percussion %s, %d notes'
            args = len(rows), notes[0].time_ms, percussion, len(notes)
            SP.print(fmt % args)
        pitches = {n.pitch_idx for n in notes
                   if n.sample_idx not in percussion}
        min_pitch = min(pitches, default = 0)
        code = list(code_mod.to_code(notes, percussion, min_pitch))
        if not is_learnable(code_mod.metadata(code)):
            SP.leave()
            continue
        if code_mod.is_transposable():
            codes = code_mod.transpose_code(code)
        else:
            codes = [code]
        SP.print('%d transpositions' % len(codes))
        for code in codes:
            yield code
        SP.leave()
    SP.leave()

def load_and_encode_mod_files(mod_files, code_type):
    encoder = CharEncoder()
    n = len(mod_files)
    end_tok = (INSN_END, 0)
    meta = []
    data = []
    offset = 0
    for i, p, in enumerate(mod_files):
        codes = list(mod_file_to_codes_w_progress(i + 1, n, p, code_type))
        if not codes:
            continue
        code = flatten([c + [end_tok] for c in codes])
        code = encoder.encode_chars(code, True)
        code = np.array(code, dtype = np.uint16)
        data.append(code)
        meta.append((offset, p.name))
        offset += len(code)
    return encoder, meta, np.concatenate(data)

def build_cache(path, shuffle_path, mods, code_type):
    mod_files = [path / mod.genre / mod.fname for mod in mods]

    # Cache the shuffle so that the train-validation split is the same
    # no matter the code type.
    def rebuild_fn():
        n = len(mods)
        SP.print('Shuffling %d mods.' % n)
        indices = list(range(n))
        shuffle(indices)
        return indices

    # Shuffle according to cached indexes.
    indices = load_pickle_cache(shuffle_path, rebuild_fn)
    tmp = [(i, e) for (i, e) in zip(indices, mod_files)]
    mod_files = [e for (_, e) in sorted(tmp)]

    return load_and_encode_mod_files(mod_files, code_type)

class TrainingData:
    def __init__(self, code_type):
        if code_type not in CODE_MODULES:
            s = ', '.join(CODE_MODULES)
            raise ValueError('<code-type> must be one of %s' % s)
        self.code_type = code_type

    def pause_code(self):
        return CODE_MODULES[self.code_type].pause()

    def load_disk_cache(self, path, kb_limit):
        index = load_index(path)
        mods = [mod for mod in index.values()
                if (mod.n_channels == 4
                    and mod.format == 'MOD'
                    and mod.kb_size <= kb_limit)]
        size_sum = sum(mod.kb_size for mod in mods)
        prefix = 'cache_%s' % self.code_type
        params = size_sum, kb_limit

        cache = file_name_for_params(prefix, 'pickle.gz', params)
        shuffle = file_name_for_params('shuffle', 'pickle.gz', params)

        cache_dir = path / 'caches'
        cache_dir.mkdir(exist_ok = True)
        cache_path = cache_dir / cache
        shuffle_path = cache_dir / shuffle

        def rebuild_fn():
            return build_cache(path, shuffle_path, mods, self.code_type)
        o = load_pickle_cache(cache_path, rebuild_fn)
        self.encoder, self.meta, self.data = o

    def load_mod_file(self, p):
        o = load_and_encode_mod_files([p], self.code_type)
        self.encoder, self.meta, self.data = o

    def split_3way(self, train_frac, valid_frac):
        n_mods = len(self.meta)
        n_train = int(n_mods * train_frac)
        n_valid = int(n_mods * valid_frac)
        n_test = n_mods - n_train - n_valid

        valid_offset = self.meta[n_train][0]
        test_offset = self.meta[n_train + n_valid][0]

        tds = [TrainingData(self.code_type) for _ in range(3)]
        tds[0].data = self.data[:valid_offset]
        tds[0].meta = self.meta[:n_train]

        tds[1].data = self.data[valid_offset:test_offset]
        tds[1].meta = self.meta[n_train:n_train + n_valid]

        tds[2].data = self.data[test_offset:]
        tds[2].meta = self.meta[n_train + n_valid:]

        for td in tds:
            base_ofs = td.meta[0][0]
            td.encoder = self.encoder
            td.meta = [(o - base_ofs, n) for (o, n) in td.meta]
        return tds

    def save_code(self, seq, file_path):
        code = self.encoder.decode_chars(seq)
        if file_path.suffix == '.pickle':
            save_pickle(file_path, code)
        else:
            notes = CODE_MODULES[self.code_type].to_notes(code)
            notes_to_audio_file(notes, file_path, CODE_MIDI_MAPPING, True)

def tally_tokens(encoder, data):
    unique, counts = np.unique(data, return_counts = True)
    ch_counts = [(encoder.decode_char(ix), cnt) for (ix, cnt) in
                 zip(unique, counts)]
    return sorted(ch_counts)

def print_histogram(td):
    counts = tally_tokens(td.encoder, td.data)
    total = sum(v for (_, v) in counts)
    SP.header('%d TOKENS %d TYPES' % (total, len(counts)))
    for (cmd, arg), cnt in counts:
        SP.print('%3s %10s %10d' % (cmd, arg, cnt))
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

def find_name_by_offset(meta, seek):
    at = meta[0][0]
    for ofs, name in meta[1:]:
        if ofs > seek:
            return at
        at = name
    return meta[-1][1]

def pick_song_fragment(td, i, n):
    if i != 'random':
        i = int(i)
        frag = td.data[i:i + n]
    else:
        end_idx = td.encoder.encode_char((INSN_END, 0), False)
        while True:
            i = randrange(len(td.data) - n)
            frag = td.data[i:i + n]
            if end_idx in frag:
                SP.print('EOS in fragment, regenerating.')
            else:
                break
        assert not end_idx in frag
    name = find_name_by_offset(td.meta, i)
    fmt = 'Picked fragment at %d+%d of song %s.'
    SP.print(fmt % (i, len(frag), name))
    return i, frag

if __name__ == '__main__':
    from random import choice
    from pathlib import Path
    from sys import argv
    SP.enabled = True
    _, _, td = load_training_data('pcode_abs', Path(argv[1]))
    for i in range(3):
        ofs, frag = pick_song_fragment(td, 'random', 1000)
        td.save_code(frag, Path('test-%02d.mid' % i))
