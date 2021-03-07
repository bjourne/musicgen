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
from musicgen.training_data import (load_training_data,
                                    abs_ofs_to_rel,
                                    random_song_offset,
                                    rel_ofs_to_abs,
                                    song_fragment,
                                    save_generated_sequences)
from musicgen.utils import SP, load_pickle_cache, save_pickle
from pathlib import Path
from random import choices, randrange
from tensorflow.nn import softmax

import numpy as np
import tensorflow as tf

def bulk_generate(g, root_path, rel_offsets, td, n_prompt, n_generate):

    # Pick the prompts
    n_clips = len(rel_offsets)
    n_frag = n_prompt + n_generate

    # Make offsets absolute
    abs_offsets = [rel_ofs_to_abs(td, rel_ofs)
                   for rel_ofs in rel_offsets]

    frags = np.array([song_fragment(td, o, n_frag) for o in abs_offsets])

    prompt, orig = frags[:,:n_prompt], frags[:,n_prompt:]
    SP.print('%s, %s' % (prompt.shape, orig.shape))

    # Token(s) to avoid
    end_ix = td.encoder.encode_char((INSN_END, 0), False)

    # Generate tokens
    skews = [g['sampling-method']] * len(rel_offsets)
    if g['network-type'] in ('lstm', 'gpt2', 'transformer'):
        vocab_size = len(td.encoder.ix2ch)
        model = load_generating_model(g, root_path, vocab_size, n_clips)
        seqs, log_probs = generate_sequences(g, model, prompt, n_generate,
                                             [end_ix], skews)
    elif g['network-type'] == 'original':
        seqs = orig
        log_probs = [0] * n_clips
    elif g['network-type'] == 'random':
        ixs = [ix for ix in td.encoder.ix2ch if ix != end_ix]
        seqs = np.array([choices(ixs, k = n_generate)
                         for _ in rel_offsets])
        log_probs = [0] * n_clips
    else:
        assert False

    # Concatenate
    seqs = np.hstack((prompt, seqs))

    # Save generated code
    output_path = root_path / 'bulk-generated'
    output_path.mkdir(exist_ok = True)

    save_generated_sequences(g, output_path, td,
                             seqs, rel_offsets, log_probs, skews)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Bulk generator 1.0')
    SP.enabled = args['--verbose']

    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<generator>'])

    _, td, _ = load_training_data(g['code-type'], root_path)
    vocab_size = len(td.encoder.ix2ch)

    # Parse generation schedule
    n_prompt = int(args['--n-prompt'])
    n_generate = int(args['--n-generate'])
    n_clips = int(args['--n-clips'])
    if n_generate % 2 == 1 or n_prompt % 2 == 1:
        raise ValueError('The number of tokens in the prompt and '
                         'the number of tokens to generate must '
                         'be divisible by two.')
    if td.code_type == 'dcode':
        SP.print('Code type is dcode so halving generation sizes.')
        n_generate //= 2
        n_prompt //= 2
    n_frag = n_prompt + n_generate

    # Load schedule and create the prompt array
    output_path = root_path / 'bulk-generated'
    output_path.mkdir(exist_ok = True)

    # We save the random indexes in a file so that the same bulk job
    # can be repeated using other code generators.
    schedule_name = 'schedule-%04d.pickle.gz' % n_clips
    schedule_path = output_path / schedule_name
    def pickle_cache_fun():
        offsets = [random_song_offset(td, n_frag)
                   for _ in range(n_clips)]
        return [abs_ofs_to_rel(td, o) for o in offsets]
    rel_offsets = load_pickle_cache(schedule_path, pickle_cache_fun)

    # Filter out those that already exist
    output_path = root_path / 'bulk-generated'
    existing = [e.stem.split('-')[:4]
                for e in output_path.glob('*.pickle.gz')]
    existing = [e for e in existing if len(e) == 4]
    existing = {(int(b), int(r), c, n) for (b, r, c, n) in existing}
    rel_offsets = [(b, r) for (b, r) in rel_offsets
                   if not (b, r, g['code-type'], g['network-type'])
                   in existing]

    n_rel_offsets = len(rel_offsets)
    SP.print('Generating %d clips.' % n_rel_offsets)

    # Splitting the load into chunks of 16. To many sequences at once
    # either exhausts the memory or times out Google Colab.
    job_size = 16
    for i in range(0, n_rel_offsets, job_size):
        job = rel_offsets[i:i+job_size]
        bulk_generate(g, root_path, job, td, n_prompt, n_generate)

if __name__ == '__main__':
    main()
