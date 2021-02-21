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
    arrs = []
    n = len(mod_files)
    end_tok = (INSN_END, 0)
    for i, p, in enumerate(sorted(mod_files)):
        codes = list(mod_file_to_codes_w_progress(i + 1, n, p, code_type))
        if not codes:
            continue
        # Encode and add ending
        codes = [c + [end_tok] for c in codes]
        codes = [encoder.encode_chars(c, True) for c in codes]
        arrs.append((p.name, codes))
    return encoder, arrs

def build_cache(path, shuffle_file, mods, code_type):
    mod_files = [path / mod.genre / mod.fname for mod in mods]
    encoder, arrs = load_and_encode_mod_files(mod_files, code_type)

    # Cache the shuffle so that the train-validation split is the same
    # no matter the code type.
    shuffle_path = path / shuffle_file
    def rebuild_fn():
        n = len(arrs)
        SP.print('Shuffling %d mods.' % n)
        indices = list(range(n))
        shuffle(indices)
        return indices
    indices = load_pickle_cache(shuffle_path, rebuild_fn)

    # Shuffle according to indices.
    tmp = [(i, e) for (i, e) in zip(indices, arrs)]
    arrs = [e for (_, e) in sorted(tmp)]

    return encoder, arrs

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
        real_prefix = 'cache_%s' % self.code_type
        params = size_sum, kb_limit
        cache_file = file_name_for_params(real_prefix, 'pickle', params)
        cache_path = path / cache_file
        shuffle_file = file_name_for_params('shuffle', 'pickle', params)
        def rebuild_fn():
            return build_cache(path, shuffle_file, mods, self.code_type)
        data = load_pickle_cache(cache_path, rebuild_fn)
        self.encoder, self.arrs = data

    def load_mod_file(self, p):
        self.encoder, self.arrs = \
            load_and_encode_mod_files([p], self.code_type)

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
            notes = CODE_MODULES[self.code_type].to_notes(code)
            notes_to_audio_file(notes, file_path, CODE_MIDI_MAPPING, True)

def flatten_training_data(td):
    arrs = flatten([arrs for (_, arrs) in td.arrs])
    return np.concatenate(arrs)

def tally_tokens(td):
    seq = flatten_training_data(td)
    unique, counts = np.unique(seq, return_counts = True)
    ch_counts = [(td.encoder.decode_char(ix), cnt) for (ix, cnt) in
                 zip(unique, counts)]
    return sorted(ch_counts)

def print_histogram(td):
    counts = tally_tokens(td)
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
    _, _, td = load_training_data('pcode_abs', Path(argv[1]))
    end_idx = td.encoder.encode_char((INSN_END, 0), True)
    seq = flatten_training_data(td)
    for i in range(10):
        ofs, frag = pick_song_fragment(seq, 'random', 1500, end_idx)
        SP.print('Picked fragment %d+%d.' % (ofs, len(frag)))
        td.save_code(frag, Path('test-%02d.mid' % i))
