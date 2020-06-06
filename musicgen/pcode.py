# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# PCode stands for parallel or polyphonic code.
from collections import Counter
from musicgen.code_utils import (CODE_MIDI_MAPPING,
                                 INSN_PITCH,
                                 INSN_REL_PITCH,
                                 INSN_SILENCE,
                                 INSN_DRUM,
                                 fix_durations,
                                 guess_initial_pitch,
                                 guess_percussive_instruments)
from musicgen.corpus import load_index
from musicgen.generation import notes_to_midi_file
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import ModNote, linearize_rows, rows_to_mod_notes
from musicgen.utils import (SP,
                            encode_training_sequence,
                            file_name_for_params, flatten,
                            load_pickle_cache, sort_groupby)
from random import shuffle
import numpy as np

# All songs end with four bars of silence
EOS_SILENCE = [(INSN_SILENCE, 16)] * 4

########################################################################
# Encode/Decode
########################################################################
def pcode_to_midi_file(pcode, file_path, relative_pitches):
    SP.header('WRITING %s' % file_path)
    if relative_pitches:
        at_pitch = guess_initial_pitch(pcode)

    notes = []
    at = 0
    for cmd, arg in pcode:
        ri = at // 4
        ci = at % 4
        if cmd in (INSN_PITCH, INSN_REL_PITCH, INSN_DRUM):
            if cmd == INSN_DRUM:
                sample_idx = arg + 1
                pitch_idx = 36
            else:
                sample_idx = 4
                if cmd == INSN_PITCH:
                    pitch_idx = arg
                else:
                    at_pitch += arg
                    pitch_idx = at_pitch
            note = ModNote(ri, ci, sample_idx, pitch_idx, 48, -1)
            notes.append(note)
            at += 1
        elif cmd == INSN_SILENCE:
            at += arg
        else:
            assert False

    # Guess and set row time
    row_indices = {n.row_idx for n in notes}
    max_row = max(row_indices)
    row_time_ms = int(160 * len(row_indices) / max_row)
    for n in notes:
        n.time_ms = row_time_ms

    fmt = 'Rel pitches: %s, guessed row time: %s.'
    SP.print(fmt % (rel_pitches, row_time_ms))

    # Fix durations
    cols = sort_groupby(notes, lambda n: n.col_idx)
    for _, col in cols:
        fix_durations(list(col))
    notes_to_midi_file(notes, file_path, CODE_MIDI_MAPPING)
    SP.leave()

def mod_file_to_pcode(file_path, rel_pitches):
    SP.header('READING %s' % file_path)
    try:
        mod = load_file(file_path)
    except PowerPackerModule:
        SP.print('PowerPacker module.')
        SP.leave()
        return

    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    if not notes:
        SP.print('Empty module.')
        SP.leave()
        return

    percussion = guess_percussive_instruments(mod, notes)
    fmt = 'Row time %d ms, guessed percussion: %s.'
    SP.print(fmt % (notes[0].time_ms, percussion))

    pitches = {n.pitch_idx for n in notes
               if n.sample_idx not in percussion}
    if not pitches:
        SP.print('No melody.')
        SP.leave()
        return
    min_pitch = min(pitch for pitch in pitches)
    max_pitch = max(pitch for pitch in pitches)
    pitch_range = max_pitch - min_pitch
    if pitch_range >= 36:
        SP.print('Pitch range %d too large.' % pitch_range)
        SP.leave()
        return

    def note_to_event(n):
        si = n.sample_idx
        at = 4 * n.row_idx + n.col_idx
        if si in percussion:
            return at, True, percussion[si]
        return at, False, n.pitch_idx - min_pitch,
    notes = sorted([note_to_event(n) for n in notes])

    if rel_pitches:
        # Make pitches relative
        current_pitch = None
        notes2 = []
        for at, is_drum, pitch in notes:
            if is_drum:
                notes2.append((at, True, pitch))
            else:
                if current_pitch is None:
                    notes2.append((at, False, 0))
                else:
                    notes2.append((at, False, pitch - current_pitch))
                current_pitch = pitch
        notes = notes2

    def produce_silence(delta):
        thresholds = [16, 8, 4, 3, 2, 1]
        for threshold in thresholds:
            while delta >= threshold:
                yield threshold
                delta -= threshold
        assert delta >= -1

    at = 0
    last_pitch = None
    for ofs, is_drum, arg in notes:
        delta = ofs - at
        for sil in produce_silence(delta - 1):
            yield INSN_SILENCE, sil
        if is_drum:
            yield INSN_DRUM, arg
        elif rel_pitches:
            yield INSN_REL_PITCH, arg
        else:
            yield INSN_PITCH, arg
        at = ofs
    SP.leave()

    # We end every mod with four bars of silence
    for insn in EOS_SILENCE:
        yield insn

########################################################################
# Test encode and decode
########################################################################
def test_encode_decode(mod_file, rel_pitches):
    pcode = list(mod_file_to_pcode(mod_file, rel_pitches))
    pcode_to_midi_file(pcode, 'test.mid', rel_pitches)

########################################################################
# Cache loading
########################################################################
def build_corpus(corpus_path, mods, rel_pitches):
    file_paths = [corpus_path / mod.genre / mod.fname for mod in mods]
    pcode_per_mod = [mod_file_to_pcode(fp, rel_pitches)
                     for fp in file_paths]

    shuffle(pcode_per_mod)
    pcode = flatten(pcode_per_mod)
    return encode_training_sequence(pcode)

def load_corpus(corpus_path, kb_limit, rel_pitches):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
                and mod.kb_size <= kb_limit)]
    size_sum = sum(mod.kb_size for mod in mods)
    params = (size_sum, kb_limit, rel_pitches)
    cache_file = file_name_for_params('cached_pcode', 'pickle', params)
    cache_path = corpus_path / cache_file
    def rebuild_fun():
        return build_corpus(corpus_path, mods, rel_pitches)
    return load_pickle_cache(cache_path, rebuild_fun)

def load_mod_file(mod_file, rel_pitches):
    pcode = list(mod_file_to_pcode(mod_file, rel_pitches))
    return encode_training_sequence(pcode)
