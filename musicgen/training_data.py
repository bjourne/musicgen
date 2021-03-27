# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# There's lots of random stuff in this module.
from collections import Counter
from musicgen import dcode, pcode_abs, pcode_rel, rcode, rcode2
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
    'dcode' : dcode,
    'rcode' : rcode,
    'rcode2' : rcode2
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
        yield False, 0, (ERR_PARSE_ERROR, err_arg)
        return

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
            yield False, idx, err
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
            yield True, idx, codes
        SP.leave()
    SP.leave()

def load_and_encode_mod_files(mod_files, code_type):
    encoder = CharEncoder()
    n = len(mod_files)
    errors = []
    songs = []
    for i, p, in enumerate(mod_files):
        subsongs = list(mod_file_to_codes_w_progress(i + 1, n,
                                                     p, code_type))

        # Handle errors
        errors.extend([(p.name, idx, err)
                       for (trainable, idx, err) in subsongs
                       if not trainable])

        # Filter those that parsed
        codes = [codes for (trainable, _, codes) in subsongs
                 if trainable]

        # Skip if none did
        if len(codes) == 0:
            continue

        # Encode
        codes = [[np.array(encoder.encode_chars(trans, True),
                           dtype = np.uint16)
                  for trans in subsong]
                 for subsong in codes]
        songs.append((p.name, codes))
    print_encoding_errors(errors)
    return encoder, songs

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
        self.encoder, self.songs = o

    def load_mod_file(self, p):
        o = load_and_encode_mod_files([p], self.code_type)
        self.encoder, self.songs = o

    def split_3way(self, train_frac, valid_frac):
        n_songs = len(self.songs)
        n_train = int(n_songs * train_frac)
        n_valid = int(n_songs * valid_frac)
        n_test = n_songs - n_train - n_valid

        tds = [TrainingData(self.code_type) for _ in range(3)]
        tds[0].songs = self.songs[:n_train]
        tds[1].songs = self.songs[n_train:n_train + n_valid]
        tds[2].songs = self.songs[n_train + n_valid:]
        for td in tds:
            td.encoder = self.encoder
        return tds

def tally_tokens(encoder, songs):
    counter = Counter()
    for name, song in songs:
        for subsong in song:
            for transp in subsong:
                els, counts = np.unique(transp, return_counts = True)
                for el, count in zip(els, counts):
                    counter[el] += count
    ch_counts = [(encoder.decode_char(ix), cnt) for (ix, cnt) in
                 counter.items()]
    return sorted(ch_counts)

def print_histogram(td):
    counts = tally_tokens(td.encoder, td.songs)
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
    return train, valid, test

def normalize_pitches(td, frag):
    code = td.encoder.decode_chars(frag)
    code = CODE_MODULES[td.code_type].normalize_pitches(code)
    frag = td.encoder.encode_chars(code, False)
    return frag

def abs_ofs_to_rel_ofs(td, abs_ofs):
    at = 0
    for s_i, (name, s) in enumerate(td.songs):
        for ss_i, ss in enumerate(s):
            for t_i, t in enumerate(ss):
                t_len = len(t)
                assert t_len < 100_000
                if at + t_len > abs_ofs:
                    o = abs_ofs - at
                    # Ensure that the relative offset is in even pcode
                    # units.
                    if td.code_type == 'dcode':
                        o *= 2
                    if o % 2 == 1:
                        o -= 1
                    return s_i, ss_i, t_i, o
                at += t_len
    return None

def random_rel_ofs(td, n):
    tot = sum(sum(sum(len(transp) for transp in subsong)
                  for subsong in song) for name, song in td.songs)
    while True:
        abs_ofs = randrange(tot - n)
        s_i, ss_i, t_i, o = abs_ofs_to_rel_ofs(td, abs_ofs)
        transp = td.songs[s_i][1][ss_i][t_i]
        if o + n > len(transp):
            continue
        return s_i, ss_i, t_i, o

def save_generated_sequences(g, output_path, td,
                             seqs, rel_offsets, log_probs, skews):
    skew_type_to_char = {
        'top-p' : 'p',
        'temperature' : 't',
        'original' : 'o',
        'random' : 'r'
    }

    song_fmt = '%s-%s-%s-%s%.3f-%04d.pickle.gz'
    song_id_fmt = '%04d-%02d-%02d-%05d'

    for seq, rel_offset, log_prob, skew in zip(seqs, rel_offsets,
                                               log_probs, skews):

        song_id = song_id_fmt % rel_offset
        skew_ch = skew_type_to_char[skew[0]]
        filename = song_fmt % (song_id,
                               g['code-type'], g['network-type'],
                               skew_ch, skew[1], -log_prob)
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
        pitches = {n.pitch_idx for n in notes
                   if n.sample_idx not in percussion}
        min_pitch = min(pitches, default = 0)
        for n in notes:
            n.pitch_idx -= min_pitch
        code = list(code_mod.to_code(notes, percussion))

        fmt = '%d notes, %d rows, %d tokens, %d ms/row, percussion %s'
        args = (len(notes), len(rows), len(code),
                notes[0].time_ms if notes else - 1, set(percussion))
        SP.print(fmt % args)

        row_time = code_mod.estimate_row_time(code)
        notes = code_mod.to_notes(code, row_time)
        fname = Path('test-%02d.mid' % idx)
        notes_to_audio_file(notes, fname, CODE_MIDI_MAPPING, False)

if __name__ == '__main__':
    from sys import argv
    SP.enabled = True
    convert_to_midi(argv[1], argv[2])
