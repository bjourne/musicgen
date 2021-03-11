# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# There's lots of random stuff in this module.
from collections import namedtuple
from musicgen import dcode, pcode_abs, pcode_rel
from musicgen.code_utils import (CODE_MIDI_MAPPING, INSN_END, INSN_PITCH,
                                 guess_percussive_instruments)
from musicgen.corpus import load_index
from musicgen.analyze import dissonant_chords
from musicgen.generation import notes_to_audio_file
from musicgen.parser import UnsupportedModule, load_file
from musicgen.rows import linearize_subsongs, rows_to_mod_notes
from musicgen.utils import (SP, CharEncoder,
                            flatten, file_name_for_params,
                            load_pickle_cache, save_pickle,
                            sort_groupby)
from pathlib import Path
from random import randrange, shuffle
import numpy as np

CODE_MODULES = {
    'pcode_abs' : pcode_abs,
    'pcode_rel' : pcode_rel,
    'dcode' : dcode
}

ERR_DISSONANCE = 0
ERR_FEW_NOTES = 1
ERR_FEW_MEL_NOTES = 2
ERR_FEW_UNIQUE_PITCHES = 3
ERR_PARSE_ERROR = 4
ERR_PITCH_RANGE = 5
ERR_EXCESSIVE_PERCUSSION = 6

def training_error(notes, percussion):
    n_notes = len(notes)
    if n_notes < 64:
        return ERR_FEW_NOTES, n_notes
    mel_notes = {n for n in notes if not n.sample_idx in percussion}
    perc_notes = {n for n in notes if n.sample_idx in percussion}
    n_mel_notes = len(mel_notes)
    n_perc_notes = len(perc_notes)
    assert n_mel_notes + n_perc_notes == n_notes
    if n_mel_notes < 32:
        return ERR_FEW_MEL_NOTES, n_mel_notes

    pitches = {n.pitch_idx for n in mel_notes}
    n_unique_pitches = len(pitches)
    if n_unique_pitches < 4:
        return ERR_FEW_UNIQUE_PITCHES, n_unique_pitches

    pitch_range = max(pitches) - min(pitches)
    if pitch_range >= 36:
        return ERR_PITCH_RANGE, pitch_range

    if n_perc_notes > 2 * n_mel_notes:
        return ERR_EXCESSIVE_PERCUSSION, n_perc_notes, n_mel_notes

    # 40 is an arbitrary cutoff...
    n_chords, n_diss_chords = dissonant_chords(mel_notes)
    diss_frac = n_diss_chords / n_chords if n_chords else 0.0
    if diss_frac >= 0.25 and n_chords > 40:
        return ERR_DISSONANCE, diss_frac, n_chords
    return None

def print_encoding_errors(errors):
    errors_per_type = sort_groupby(errors, lambda x: x[2][0])
    for error_type, subsongs in errors_per_type:
        subsongs = list(subsongs)
        n_subsongs = len(subsongs)
        if error_type == ERR_DISSONANCE:
            header_part = 'WITH DISSONANCE'
        elif error_type == ERR_FEW_MEL_NOTES:
            header_part = 'WITH TO FEW MELODIC NOTES'
        elif error_type == ERR_PARSE_ERROR:
            header_part = 'WITH PARSE ERRORS'
        elif error_type == ERR_PITCH_RANGE:
            header_part = 'WITH TOO WIDE PITCH RANGES'
        elif error_type == ERR_FEW_NOTES:
            header_part = 'WITH TO FEW NOTES'
        elif error_type == ERR_FEW_UNIQUE_PITCHES:
            header_part = 'WITH TO FEW UNIQUE PITCHES'
        elif error_type == ERR_EXCESSIVE_PERCUSSION:
            header_part = 'WITH EXCESSIVE PERCUSSION'
        else:
            assert False
        SP.header('%d SUBSONGS %s' % (n_subsongs, header_part))
        for name, idx, err in subsongs:
            if error_type == ERR_DISSONANCE:
                args = name, idx, err[1], err[2]
                fmt = '%-40s %3d %.2f %4d'
            elif error_type == ERR_FEW_MEL_NOTES:
                args = name, idx, err[1]
                fmt = '%-40s %3d %4d'
            elif error_type == ERR_PARSE_ERROR:
                args = name, idx, err[1]
                fmt = '%-40s %3d %s'
            elif error_type == ERR_PITCH_RANGE:
                args = name, idx, err[1]
                fmt = '%-40s %3d %2d'
            elif error_type == ERR_FEW_NOTES:
                args = name, idx, err[1]
                fmt = '%-40s %3d %4d'
            elif error_type == ERR_FEW_UNIQUE_PITCHES:
                args = name, idx, err[1]
                fmt = '%-40s %3d %4d'
            elif error_type == ERR_EXCESSIVE_PERCUSSION:
                args = name, idx, err[1], err[2]
                fmt = '%-40s %3d %4d %4d'
            else:
                assert False
            SP.print(fmt % args)
        SP.leave()

def mod_file_to_codes_w_progress(i, n, file_path, code_type):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, file_path))
    try:
        mod = load_file(file_path)
    except UnsupportedModule as e:
        SP.print('Unsupported module format.')
        SP.leave()
        err_arg = e.args[0] if e.args else e.__class__.__name__
        return [(False, 0, (ERR_PARSE_ERROR, err_arg))]

    code_mod = CODE_MODULES[code_type]
    subsongs = list(linearize_subsongs(mod, 1))
    volumes = [header.volume for header in mod.sample_headers]
    parsed_subsongs = []
    for idx, (order, rows) in enumerate(subsongs):
        SP.header('SUBSONG %d' % idx)
        notes = rows_to_mod_notes(rows, volumes)
        percussion = guess_percussive_instruments(mod, notes)
        if notes:
            fmt = '%d rows, %d ms/row, percussion %s, %d notes'
            args = (len(rows), notes[0].time_ms,
                    set(percussion), len(notes))
            SP.print(fmt % args)

        err = training_error(notes, percussion)
        if err:
            parsed_subsongs.append((False, idx, err))
        else:
            pitches = {n.pitch_idx for n in notes
                       if n.sample_idx not in percussion}
            min_pitch = min(pitches, default = 0)

            # Subtract min pitch
            for n in notes:
                n.pitch_idx -= min_pitch
            code = list(code_mod.to_code(notes, percussion))
            if code_mod.is_transposable():
                codes = code_mod.code_transpositions(code)
            else:
                codes = [code]
            fmt = '%d transpositions of length %d'
            SP.print(fmt % (len(codes), len(code)))
            parsed_subsongs.append((True, idx, codes))
        SP.leave()
    SP.leave()
    return parsed_subsongs

def load_and_encode_mod_files(mod_files, code_type):
    encoder = CharEncoder()
    n = len(mod_files)
    end_tok = (INSN_END, 0)
    meta = []
    data = []
    errors = []
    offset = 0
    for i, p, in enumerate(mod_files):
        parsed_subsongs = mod_file_to_codes_w_progress(i + 1, n,
                                                       p, code_type)
        # Handle errors
        errors.extend([(p.name, idx, err)
                       for (trainable, idx, err) in parsed_subsongs
                       if not trainable])

        # Filter those that parsed
        codes = [codes for (trainable, _, codes) in parsed_subsongs
                 if trainable]
        # Flatten subsongs
        codes = flatten(codes)
        # Add end token
        code = flatten([c + [end_tok] for c in codes])

        # Encode
        code = encoder.encode_chars(code, True)
        code = np.array(code, dtype = np.uint16)
        n_code = len(code)

        # Skip if empty
        if n_code == 0:
            continue

        data.append(code)
        meta.append((offset, p.name))
        offset += n_code

    print_encoding_errors(errors)
    return encoder, meta, np.concatenate(data) if data else []

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

# The point of this hairy code is to be able to convert offsets in the
# pcode cache file to corresponding offsets in the dcode cache file.
def abs_ofs_to_rel(td, abs_ofs):
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    endings = np.where(td.data[:abs_ofs] == end_idx)[0]
    base = len(endings)
    if base > 0:
        rel = abs_ofs - endings[-1] - 1
    else:
        rel = abs_ofs

    # Ensure that the relative offset is in even pcode units.
    if td.code_type == 'dcode':
        rel *= 2
    if rel % 2 == 1:
        rel -= 1
    return base, rel

def rel_ofs_to_abs(td, rel_ofs):
    base, rel = rel_ofs
    # Cause rel is in even pcode units
    assert rel % 2 == 0
    if td.code_type == 'dcode':
        rel //= 2
    if base == 0:
        return rel
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    endings = np.where(td.data == end_idx)[0]
    ending_ofs = endings[base - 1]
    return ending_ofs + 1 + rel

def random_song_offset(td, n):
    end_idx = td.encoder.encode_char((INSN_END, 0), False)
    while True:
        abs_ofs = randrange(len(td.data) - n)
        frag = td.data[abs_ofs:abs_ofs + n]
        if not end_idx in frag:
            return abs_ofs
        else:
            SP.print('INSN_END in sampled sequence, retrying.')

def song_fragment(td, abs_ofs, n_frag):
    frag = td.data[abs_ofs:abs_ofs + n_frag]
    code = td.encoder.decode_chars(frag)
    code = CODE_MODULES[td.code_type].normalize_pitches(code)
    return td.encoder.encode_chars(code, False)

# Remove these
def find_name_by_offset(meta, seek):
    at = meta[0][0]
    for ofs, name in meta[1:]:
        if ofs > seek:
            return at
        at = name
    return meta[-1][1]

def pick_song_fragment(td, i, n, normalize_pitches):
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

    if normalize_pitches:
        code = td.encoder.decode_chars(frag)
        code = CODE_MODULES[td.code_type].normalize_pitches(code)
        frag = td.encoder.encode_chars(code, False)
    return i, frag

def save_generated_sequences(g, output_path, td,
                             seqs, rel_offsets, log_probs, skews):
    fmt = '%06d-%06d-%s-%s-%s%.3f-%04d.pickle.gz'
    skew_type_to_char = {
        'top-p' : 'p',
        'temperature' : 't',
        'original' : 'o',
        'random' : 'r'
    }
    for seq, rel_offset, log_prob, skew in zip(seqs, rel_offsets,
                                               log_probs, skews):
        skew_ch = skew_type_to_char[skew[0]]
        args = (rel_offset[0], rel_offset[1],
                g['code-type'], g['network-type'],
                skew_ch, skew[1], -log_prob)
        filename = fmt % args
        file_path = output_path / filename
        code = td.encoder.decode_chars(seq)
        save_pickle(file_path, code)

def convert_to_midi(code_type, mod_file):
    code_mod = CODE_MODULES[code_type]
    mod = load_file(mod_file)
    subsongs = linearize_subsongs(mod, 1)
    volumes = [header.volume for header in mod.sample_headers]
    for idx, (_, rows) in enumerate(subsongs):
        notes = rows_to_mod_notes(rows, volumes)
        percussion = guess_percussive_instruments(mod, notes)
        if notes:
            fmt = '%d rows, %d ms/row, percussion %s, %d notes'
            args = len(rows), notes[0].time_ms, percussion, len(notes)
            SP.print(fmt % args)
        pitches = {n.pitch_idx for n in notes
                   if n.sample_idx not in percussion}
        min_pitch = min(pitches, default = 0)
        for n in notes:
            n.pitch_idx -= min_pitch
        code = list(code_mod.to_code(notes, percussion))
        row_time = code_mod.estimate_row_time(code)
        notes = code_mod.to_notes(code, row_time)
        fname = Path('test-%02d.mid' % idx)
        notes_to_audio_file(notes, fname, CODE_MIDI_MAPPING, False)

if __name__ == '__main__':
    from sys import argv
    SP.enabled = True
    convert_to_midi(argv[1], argv[2])
