# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
"""
Music generation
================

Usage:
    model-generator.py [options] <root-path> <generator>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --seed-idx=<i>         seed index [default: random]
    --add-pause            insert a pause between the prompt and
                           generated code
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen.code_generators import get_code_generator
from musicgen.code_utils import INSN_END
from musicgen.tensorflow import load_generating_model, generate_sequences
from musicgen.training_data import load_training_data, pick_song_fragment
from musicgen.utils import SP, save_pickle
from pathlib import Path
from random import randrange
from tensorflow.nn import softmax
from tqdm import trange

import numpy as np

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<generator>'])

    # Load training data
    _, _, td = load_training_data(g['code-type'], root_path)
    vocab_size = len(td.encoder.ix2ch)

    # Generating settings
    temps = [0.90, 0.95, 1.0, 1.01, 1.02]
    # For GPT-2 0.87 and 0.90 is too low
    # top_ps = [0.87, 0.90, 0.94, 0.98, 0.99]
    top_ps = [0.92, 0.95, 0.98, 0.99, 0.999]

    skews = [('temperature', t) for t in temps] \
        + [('top-p', p) for p in top_ps]

    n_clips = len(skews)
    n_generate = 800
    n_prompt = 64

    # Load the model
    model = load_generating_model(g, root_path, vocab_size, n_clips)

    # Pick a song fragment
    n_frag = n_prompt + n_generate
    seed_ix = args['--seed-idx']
    seed_ix, frag = pick_song_fragment(td, seed_ix, n_frag, True)
    prompt, orig = frag[:n_prompt], frag[n_prompt:]
    prompt = np.repeat(np.expand_dims(prompt, 0), n_clips, axis = 0)

    # Token(s) to avoid
    end_ix = td.encoder.encode_char((INSN_END, 0), False)

    seqs, log_probs = generate_sequences(g, model, prompt, n_generate,
                                         [end_ix], skews)

    # Add the original
    seqs = np.vstack((seqs, orig))
    prompt = np.vstack((prompt, prompt[0]))
    log_probs = np.append(log_probs, 0)
    skews.append(('original', 0))

    # Maybe add a pause
    if args['--add-pause']:
        pause = td.encoder.encode_chars(td.pause_code(), False)
        join = np.repeat(np.expand_dims(pause, 0), len(seqs), axis = 0)
        seqs = np.hstack((prompt, join, seqs))
    else:
        seqs = np.hstack((prompt, seqs))

    # Save generated code
    output_path = root_path / 'generated'
    output_path.mkdir(exist_ok = True)

    fmt = '%010d-%s-%s-%s%.3f-%04d.pickle.gz'
    for seq, log_prob, skew in zip(seqs, log_probs, skews):
        args = (seed_ix, g['code-type'], g['network-type'],
                skew[0][0], skew[1], -log_prob)
        filename = fmt % args
        file_path = output_path / filename
        code = td.encoder.decode_chars(seq)
        save_pickle(file_path, code)

if __name__ == '__main__':
    main()
