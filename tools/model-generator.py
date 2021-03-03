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
    --add-pause            insert a pause between the prompt and
                           generated code
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen.code_utils import INSN_END
from musicgen.params import ModelParams
from musicgen.tensorflow import (compiled_model_from_params,
                                 select_strategy)
from musicgen.training_data import load_training_data, pick_song_fragment
from musicgen.utils import SP
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
    P[prob_ixs[top_n:]] = np.finfo('float').eps
    return P / P.sum()

def sample_logits(logits, end_ix, temps, top_ps):
    n_temps = len(temps)
    eps = np.finfo('float').eps
    Ps = softmax(logits).numpy()

    print('Maxes: %s' % np.around(np.max(Ps, axis = 1), 4))

    # Dont sample end tokens
    Ps[:, end_ix] = eps
    for i, temp in enumerate(temps):
        Ps[i] = temperature_skew(Ps[i], temp)
    for i, top_p in enumerate(top_ps):
        Ps[i + n_temps] = top_p_skew(Ps[i + n_temps], top_p)
    ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
    return ixs, [np.log(Ps[i, ix]) for i, ix in enumerate(ixs)]

def lstm_continuation(model, temps, top_ps, seed,
                      n_samples, max_seq_len, end_ix):

    SP.print('Priming the model with %d tokens.' % seed.shape[1])
    for i in trange(seed.shape[1] - 1):
        model.predict(seed[:, i])

    # The last item of the seed is saved so that it can be used to
    # generate the first prediction.
    preds = np.expand_dims(seed[:, -1], 0)
    log_prob_sums = np.zeros(len(temps) + len(top_ps))

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in range(n_samples):
        last_word = preds[-1]
        logits = model.predict(last_word)[:, -1, :]
        ixs, log_probs = sample_logits(logits, end_ix, temps, top_ps)
        log_prob_sums += log_probs
        preds = np.vstack((preds, ixs))
    # Skip the first element which is not actually a prediction.
    return preds.T[:,1:], log_prob_sums

def transformer_continuation(model, temps, top_ps, seed, n_samples,
                             max_seq_len, end_ix):
    # seed = np.array(seed, dtype = np.int32)
    log_prob_sums = np.zeros(len(temps) + len(top_ps))
    preds = np.empty((0, seed.shape[0]), int)

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in range(n_samples):
        logits = model(seed, training = False)[:, -1, :]
        ixs, log_probs = sample_logits(logits, end_ix, temps, top_ps)
        preds = np.vstack((preds, ixs))
        log_prob_sums += log_probs

        # Append column
        seed = np.append(seed, np.expand_dims(ixs, 1), axis = 1)
        if seed.shape[1] >= max_seq_len:
            # Delete first column
            seed = seed[:, 1:]
    return preds.T, log_prob_sums

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])

    params = ModelParams.from_docopt_args(args)
    _, _, td = load_training_data(params.code_type, path)

    temps = [0.90, 0.95, 1.0, 1.01, 1.02]

    # For GPT-2 0.87 and 0.90 is too low
    # top_ps = [0.87, 0.90, 0.94, 0.98, 0.99]
    top_ps = [0.92, 0.95, 0.98, 0.99, 0.999]

    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps

    n_samples = 800
    n_seed = 64

    enc = td.encoder
    end_ix = td.encoder.encode_char((INSN_END, 0), False)

    vocab_size = len(enc.ix2ch)
    seed_ix = args['--seed-idx']
    max_seq_len = int(args['--seq-len'])
    file_format = args['--file-format']

    SP.header('%d PREDICTIONS' % n_preds)

    weights_dir = path / 'weights'
    model = compiled_model_from_params(weights_dir, params, vocab_size,
                                       n_preds, False)

    n_frag = n_seed + n_samples
    seed_ix, frag = pick_song_fragment(td, seed_ix, n_frag, True)

    seed, orig = frag[:n_seed], frag[n_seed:]
    seed = np.repeat(np.expand_dims(seed, 0), n_preds, axis = 0)

    mtype = params.model_type
    if mtype in ('transformer', 'gpt2'):
        cont_fn = transformer_continuation
    else:
        cont_fn = lstm_continuation

    seqs, log_probs = cont_fn(model, temps, top_ps, seed, n_samples,
                              max_seq_len, end_ix)

    seqs = np.vstack((seqs, orig))
    seed = np.vstack((seed, seed[0]))

    # Maybe add a pause
    if args['--add-pause']:
        pause = td.encoder.encode_chars(td.pause_code(), False)
        join = np.repeat(np.expand_dims(pause, 0), len(seqs), axis = 0)
        seqs = np.hstack((seed, join, seqs))
    else:
        seqs = np.hstack((seed, seqs))

    prefix = '%s-%s-%09d' % (td.code_type, mtype, seed_ix)
    file_names = ['%s-t%.3f' % (prefix, t) for t in temps]
    file_names += ['%s-p%.3f' % (prefix, p) for p in top_ps]

    # Add orig
    file_names.append('%s-orig' % prefix)
    log_probs = np.append(log_probs, 0)

    base_path = path / 'generated'
    base_path.mkdir(exist_ok = True)

    for file_name, seq, log_prob in zip(file_names, seqs, log_probs):
        file_name = '%s-%04d.%s' % (file_name, -log_prob, file_format)
        file_path = base_path / file_name
        td.save_code(seq, file_path)

if __name__ == '__main__':
    main()
