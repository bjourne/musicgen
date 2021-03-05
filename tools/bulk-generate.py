# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
"""
Bulk generation
===============
Tool for bulk generation of songs.

Usage:
    bulk-generate.py [options] <root-path> <generator>

Options:
    -h --help                show this screen
    -v --verbose             print more output
    --n-prompt=<i>           number of tokens in the prompt
    --n-generate=<i>         number of tokens to generate per clip
    --n-clips=<i>            number of clips
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen.code_generators import get_code_generator
from musicgen.code_utils import INSN_END
from musicgen.tensorflow import load_generating_model, generate_sequences
from musicgen.training_data import load_training_data, pick_song_fragment
from musicgen.utils import SP, load_pickle_cache, save_pickle
from pathlib import Path
from random import choices
from tensorflow.nn import softmax

import numpy as np
import tensorflow as tf

def main():
    # Prologue
    args = docopt(__doc__, version = 'Bulk generator 1.0')
    SP.enabled = args['--verbose']

    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<generator>'])

    # Load training data
    _, td, _ = load_training_data(g['code-type'], root_path)
    vocab_size = len(td.encoder.ix2ch)

    # Parse generation schedule
    n_prompt = int(args['--n-prompt'])
    n_generate = int(args['--n-generate'])
    n_clips = int(args['--n-clips'])
    n_frag = n_prompt + n_generate

    if n_generate % 2 == 1 or n_prompt % 2 == 1:
        raise ValueError('The number of tokens in the prompt and '
                         'the number of tokens to generate must '
                         'be divisible by two.')

    # Load schedule and create the prompt array
    output_path = root_path / 'bulk-generated'
    output_path.mkdir(exist_ok = True)

    schedule_name = 'schedule-%03d.pickle.gz' % n_clips
    schedule_path = output_path / schedule_name
    def build_schedule():
        return [pick_song_fragment(td, 'random', n_frag, False)[0]
                for _ in range(n_clips)]
    offsets = load_pickle_cache(schedule_path, build_schedule)

    # Pick fragments from the training data
    frags = np.array([pick_song_fragment(td, o, n_frag, True)[1]
                      for o in offsets])
    prompt, orig = frags[:,:n_prompt], frags[:,n_prompt:]

    # Token(s) to avoid
    end_ix = td.encoder.encode_char((INSN_END, 0), False)

    # Generate tokens
    skews = [g['sampling-method']] * n_clips
    if g['network-type'] in ('lstm', 'gpt2', 'transformer'):
        model = load_generating_model(g, root_path, vocab_size, n_clips)
        seqs, log_probs = generate_sequences(g, model, prompt, n_generate,
                                             [end_ix], skews)
    elif g['network-type'] == 'original':
        seqs = orig
        log_probs = [0] * n_clips
    elif g['network-type'] == 'random':
        ixs = [ix for ix in td.encoder.ix2ch if ix != end_ix]
        seqs = [choices(ixs, k = n_generate) for _ in offsets]
        seqs = np.array(generated)
        log_probs = [0] * n_clips
    else:
        assert False

    # Concatenate
    seqs = np.hstack((prompt, seqs))

    # Save generated code
    fmt = '%010d-%s-%s-%s%.3f-%04d.pickle.gz'
    for seq, offset, log_prob, skew in zip(seqs, offsets,
                                           log_probs, skews):
        args = (offset, g['code-type'], g['network-type'],
                skew[0][0], skew[1], -log_prob)
        filename = fmt % args
        file_path = output_path / filename
        code = td.encoder.decode_chars(seq)
        save_pickle(file_path, code)

if __name__ == '__main__':
    main()
