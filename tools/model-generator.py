# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
"""
Music generation
================

Usage:
    model-generator.py [options] <code-type> lstm <corpus-path>
        --emb-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-generator.py [options] <code-type> transformer <corpus-path>
    model-generator.py [options] <code-type> gpt2 <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --lr=<f>               learning rate
    --epochs=<i>           epochs to train for
    --seq-len=<i>          training sequence length
    --seed-idx=<i>         seed index [default: random]
    --batch-size=<i>       size of training batches
    --file-format=<s>      output format [default: pickle]
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen.params import ModelParams
from musicgen.tensorflow import (compiled_model_from_params,
                                 select_strategy)
from musicgen.training_data import load_training_data
from musicgen.utils import SP, find_subseq
from pathlib import Path
from random import randrange
from tensorflow.nn import softmax
from tqdm import trange

import numpy as np

def temperature_skew(P, temp):
    P = np.exp(np.log(P) / temp)
    return P / P.sum()

def top_p_skew(P, top_p):
    prob_ixs = np.argsort(-P)
    PC = np.cumsum(P[prob_ixs])
    top_n = len(PC[PC <= top_p]) + 1

    # Clear the prob of those who didn't make it.
    P[prob_ixs[top_n:]] = 0.0
    return P / P.sum()

def lstm_continuation(model, temps, top_ps, seed, n_samples, max_seq_len):

    SP.print('Priming the model with %d tokens.' % seed.shape[1])
    for i in trange(seed.shape[1] - 1):
        model.predict(seed[:, i])

    preds = np.expand_dims(seed[:, -1], 0)
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps
    log_probs = [0] * n_preds

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        last_word = preds[-1]
        logits = model.predict(last_word)[:, 0, :]
        Ps = softmax(logits).numpy()

        for i in range(n_temps):
            Ps[i] = temperature_skew(Ps[i], temps[i])

        for i in range(n_top_ps):
            Ps[i + n_temps] = top_p_skew(Ps[i + n_temps], top_ps[i])

        ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
        for i, ix in enumerate(ixs):
            log_probs[i] += np.log(Ps[i, ix])
        preds = np.vstack((preds, ixs))

    SP.leave()
    # Skip the first element which is not actually a prediction.
    return preds.T[:,1:], log_probs

def transformer_continuation(model, temps, top_ps, seed, n_samples,
                             max_seq_len):
    seed = np.array(seed, dtype = np.int32)
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps
    log_probs = [0] * n_preds
    preds = []
    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        y = model(seed)
        Ps = y[:, -1, :]
        Ps = softmax(Ps).numpy()
        for i in range(n_temps):
            Ps[i] = temperature_skew(Ps[i], temps[i])
        for i in range(n_top_ps):
            Ps[i + n_temps] = top_p_skew(Ps[i + n_temps], top_ps[i])
        ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
        preds.append(ixs)

        for i, ix in enumerate(ixs):
            log_probs[i] += np.log(Ps[i, ix])

        # Append column
        seed = np.append(seed, np.expand_dims(ixs, 1), axis = 1)
        if seed.shape[1] >= max_seq_len:
            # Delete first column
            seed = seed[:, 1:]
    return [[int(preds[j][i]) for j in range(n_samples)]
            for i in range(n_temps + n_top_ps)], log_probs

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])

    params = ModelParams.from_docopt_args(args)
    _, _, data = load_training_data(params.code_type, path)
    temps = [0.90, 0.95, 1.0, 1.01, 1.02]
    top_ps = [0.87, 0.90, 0.94, 0.98, 0.99]

    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps
    n_samples = 1200
    n_seed = 256
    encoder = data.encoder
    vocab_size = len(encoder.ix2ch)
    seed_idx = args['--seed-idx']
    max_seq_len = int(args['--seq-len'])
    file_format = args['--file-format']

    SP.header('%d PREDICTIONS' % n_preds)

    model = compiled_model_from_params(path, params, vocab_size,
                                       n_preds, True)
    seq = data.flatten(True)
    pause = encoder.encode_chars(data.info.long_pause, False).tolist()

    # Select the seed
    if seed_idx != 'random':
        seed_idx = int(seed_idx)
        seed = seq[seed_idx:seed_idx + n_seed]
    else:
        while True:
            seed_idx  = randrange(len(seq) - n_seed - n_samples)
            seed = seq[seed_idx:seed_idx + n_seed]
            seed_seq = seed.tolist()
            n_unique = len(set(seed_seq))
            if list(find_subseq(seed_seq, pause)):
                SP.print('Pause in seed, regenerating.')
                continue
            if n_unique < 5:
                SP.print('To few different tokens, regenerating.')
                continue
            break
    SP.print('Seed %d+%d.' % (seed_idx, n_seed))
    seed = np.repeat(np.expand_dims(seed, 0), n_preds, axis = 0)

    mtype = params.model_type
    if mtype in ('transformer', 'gpt2'):
        cont_fn = transformer_continuation
    else:
        cont_fn = lstm_continuation

    seqs, log_probs = cont_fn(model, temps, top_ps, seed, n_samples,
                              max_seq_len)

    # Add the original
    orig = seq[seed_idx + n_seed:seed_idx + n_seed + n_samples]
    seqs = np.vstack((seqs, orig))

    seed = np.vstack((seed, seed[0]))

    join = np.repeat(np.expand_dims(pause, 0), len(seqs), axis = 0)
    seqs = np.hstack((seed, join, seqs))

    prefix = '%s-%s-%09d' % (data.code_type, mtype, seed_idx)
    file_names = ['%s-t%.3f' % (prefix, t) for t in temps]
    file_names += ['%s-p%.3f' % (prefix, p) for p in top_ps]

    # Add orig
    file_names.append('%s-orig' % prefix)
    log_probs.append(0)

    base_path = path / 'generated'
    base_path.mkdir(exist_ok = True)

    for file_name, seq, log_prob in zip(file_names, seqs, log_probs):
        file_name = '%s-%04d.%s' % (file_name, -log_prob, file_format)
        file_path = base_path / file_name
        data.save_code(seq, file_path)

if __name__ == '__main__':
    main()
