# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Model trainer

Usage:
    train-model.py [-v --win-size=<int> --kb-limit=<int>] <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --win-size=<int>       window size [default: 64]
    --kb-limit=<int>       kb limit [default: 150]
"""
from docopt import docopt
from musicgen.generation import mycode_to_midi_file
from musicgen.ml import train_model
from musicgen.mycode import INSN_PROGRAM, corpus_to_mycode_mods
from musicgen.utils import (SP, file_name_for_params,
                            flatten, load_pickle, save_pickle)
from os import environ
from pathlib import Path
from pickle import dump, load
from random import shuffle
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def flatten_corpus(corpus_path, win_size, kb_limit, pad_token):
    mycode_mods = corpus_to_mycode_mods(corpus_path, kb_limit)
    n_mods = len(mycode_mods)

    params = [n_mods, win_size, kb_limit]
    cache_file = file_name_for_params('flat_cache', 'pickle', params)
    cache_path = corpus_path / cache_file
    if not cache_path.exists():
        seqs = [[c[1] for c in mycode_mod.cols]
                for mycode_mod in mycode_mods]
        seqs = flatten(seqs)
        shuffle(seqs)

        padding = [pad_token] * win_size
        for seq in seqs:
            seq.extend(padding)
        seq = flatten(seqs)
        SP.print('Saving sequence to %s.' % cache_path)
        save_pickle(cache_path, seq)
    else:
        SP.print('Loading sequence from %s.' % cache_path)
    seq = load_pickle(cache_path)
    seq = seq[:len(seq) // 10]
    return seq

def save_sample_sequences(corpus_path, epoch, samples, idx2char):
    SP.header('EPOCH', '%d', epoch)
    for temp, seq in samples:
        fmt = '%s' if temp is None else '%.2f'
        temp_str = fmt % temp
        SP.header('TEMPERATURE %s' % temp_str)
        seq = [idx2char[i] for i in seq]
        SP.print('%s', seq)
        file_name = 'gen-%03d-%s.mid' % (epoch, temp_str)
        file_path = corpus_path / file_name
        mycode_to_midi_file(seq, file_path, 120, None)
        SP.leave()
    SP.leave()

def main():
    args = docopt(__doc__, version = 'MOD model builder 1.0')
    SP.enabled = args['--verbose']

    corpus_path = Path(args['<corpus-path>'])
    win_size = int(args['--win-size'])
    kb_limit = int(args['--kb-limit'])

    pad_token = (INSN_PROGRAM, 0)
    seq = flatten_corpus(corpus_path, win_size, kb_limit, pad_token)
    n_seq = len(seq)

    # Convert to integer sequence

    chars = sorted(set(seq))
    vocab_size = len(chars)
    SP.print('%d tokens and %d token types.', (n_seq, vocab_size))
    char2idx = {c : i for i, c in enumerate(chars)}
    idx2char = {i : c for i, c in enumerate(chars)}
    seq = np.array([char2idx[c] for c in seq])
    pad_int = char2idx[pad_token]

    # Split data
    n_train = int(n_seq * 0.8)
    n_validate = int(n_seq * 0.1)
    n_test = n_seq - n_train - n_validate
    train = seq[:n_train]
    validate = seq[n_train:n_train + n_validate]
    test = seq[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))

    def on_epoch_begin(epoch, seqs):
        save_sample_sequences(corpus_path, epoch, seqs, idx2char)

    train_model(train, validate, test,
                corpus_path, vocab_size, win_size, 128,
                on_epoch_begin, pad_int)

if __name__ == '__main__':
    main()
