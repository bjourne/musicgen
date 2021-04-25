# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
"""
Measure perplexity
==================
Tool for measuring a model's perplexity

Usage:
    measure-perplexity.py [options] <root-path> <generator>

Options:
    -h --help                show this screen
    -v --verbose             print more output
    --n-prompt=<i>           number of tokens in the prompt
    --n-generate=<i>         number of tokens to generate per clip
    --n-clips=<i>            number of clips
    --use-original           use original data for measurements
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen.code_generators import get_code_generator, weights_file
from musicgen.tensorflow import load_generating_model, measure_sequences
from musicgen.training_data import (CODE_MODULES,
                                    load_training_data,
                                    normalize_pitches,
                                    random_rel_ofs,
                                    save_generated_sequences)
from musicgen.utils import SP, load_pickle_cache, save_pickle
from pathlib import Path
from random import choices

import numpy as np
import tensorflow as tf

def bulk_measure(g, root_path, offsets, td, n_prompt, n_generate,
                 measure_original):
    n_clips = len(offsets)
    n_frag = n_prompt + n_generate

    transps = [td.songs[s_i][1][ss_i][t_i]
               for (s_i, ss_i, t_i, _) in offsets]
    rel_offsets = [o for (s_i, ss_i, t_i, o) in offsets]
    if td.code_type == 'dcode':
        rel_offsets = [o // 2 for o in rel_offsets]

    # Get fragments
    frags = [transp[o:o+n_frag]
             for (transp, o) in zip(transps, rel_offsets)]
    frags = [normalize_pitches(td, frag) for frag in frags]
    frags = np.array(frags)

    prompt, orig = frags[:,:n_prompt], frags[:,n_prompt:]

    # Generate tokens
    skews = [g['sampling-method']] * len(offsets)
    if g['network-type'] in ('lstm', 'gpt2', 'transformer'):
        vocab_size = len(td.encoder.ix2ch)
        model = load_generating_model(g, vocab_size, n_clips)
        weights_path = root_path / 'weights' / weights_file(g)
        assert weights_path.exists()
        model.load_weights(str(weights_path))
        model.reset_states()
        if not measure_original:
            orig = None
        log_probs = measure_sequences(g, model, prompt, n_generate,
                                      skews, orig)
    else:
        assert False
    return log_probs

def main():
    # Prologue
    args = docopt(__doc__, version = 'Bulk generator 1.0')
    SP.enabled = args['--verbose']

    root_path = Path(args['<root-path>'])

    # Kind of code
    gen_name = args['<generator>']
    g = get_code_generator(gen_name)

    # Parse generation schedule
    n_prompt = int(args['--n-prompt'])
    n_generate = int(args['--n-generate'])
    n_clips = int(args['--n-clips'])
    if n_generate % 2 == 1 or n_prompt % 2 == 1:
        raise ValueError('The number of tokens in the prompt and '
                         'the number of tokens to generate must '
                         'be divisible by two.')

    # Load the generation schedule
    perp_path = root_path / 'perplexity'
    perp_path.mkdir(exist_ok = True)

    # Load the cached code
    _, td, _ = load_training_data(g['code-type'], root_path)
    if td.code_type == 'dcode':
        SP.print('Code type is dcode so halving generation sizes.')
        n_generate //= 2
        n_prompt //= 2
    n_frag = n_prompt + n_generate

    # We save the random indexes in a file so that the same bulk job
    # can be repeated using other code generators.
    schedule_name = 'schedule-%04d.pickle.gz' % n_clips
    schedule_path = perp_path / schedule_name
    def pickle_cache_fun():
        return [random_rel_ofs(td, n_frag) for _ in range(n_clips)]
    offsets = load_pickle_cache(schedule_path, pickle_cache_fun)

    # Save all measurements in a datafile
    measurements_name = 'measurements-%04d.pickle.gz' % n_clips
    measurements_path = perp_path / measurements_name
    measurements = load_pickle_cache(measurements_path, lambda: {})
    if not gen_name in measurements:
        measurements[gen_name] = {}

    use_original = args['--use-original']
    if not use_original in measurements[gen_name]:
        measurements[gen_name][use_original] = {}
    offsets = [o for o in offsets
               if not o in measurements[gen_name][use_original]]

    SP.print('Measuring %d offsets...' % len(offsets))

    # Splitting the load into chunks of 16. To many sequences at once
    # either exhausts the memory or times out Google Colab.
    job_size = 16
    for i in range(0, len(offsets), job_size):
        job = offsets[i:i+job_size]
        log_probs = bulk_measure(g, root_path, job, td,
                                 n_prompt, n_generate, use_original)
        for offset, log_prob in zip(job, log_probs):
            logppl = -log_prob / n_generate
            measurements[gen_name][use_original][offset] = logppl
        save_pickle(measurements_path, measurements)
    for gen_name, cats in measurements.items():
        SP.header(gen_name)
        for use_original, offsets in cats.items():
            SP.header('%s' % use_original)
            for ofs, ppl in offsets.items():
                SP.print('%s %.3f' % (ofs, ppl))
            SP.leave()
        SP.leave()


if __name__ == '__main__':
    main()
