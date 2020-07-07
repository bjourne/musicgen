# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""
Music generation
================

Usage:
    model-generator.py [options] <code-type> lstm <corpus-path>
        --emb-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-generator.py [options] <code-type> transformer <corpus-path>
        --dropout=<f>

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
from musicgen.utils import SP, find_subseq
from musicgen.training_data import load_training_data
from musicgen.tensorflow import ModelParams, select_strategy
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

def lstm_continuation(model, temps, top_ps, seed, n_samples):
    SP.print('Priming the model with %d tokens.' % seed.shape[1])
    for i in trange(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])

    preds = np.expand_dims(seed[:, -1], 0)
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps
    log_probs = [0] * n_preds

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        last_word = preds[-1]
        Ps = model.predict(last_word)[:, 0, :]
        Ps = softmax(Ps).numpy()

        for i in range(n_temps):
            Ps[i] = temperature_skew(Ps[i], temps[i])

        for i in range(n_top_ps):
            Ps[i + n_temps] = top_p_skew(Ps[i + n_temps], top_ps[i])

        ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
        preds = np.vstack((preds, ixs))

    SP.leave()
    # Skip the first element which is not actually a prediction.
    return preds.T[:,1:]

def transformer_continuation(model, temps, top_ps, seed, n_samples):
    seed = np.array(seed, dtype = np.int32)
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    preds = []

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        y = model.predict(seed)
        Ps = y[:, -1, :]
        Ps = softmax(Ps).numpy()

        for i in range(n_temps):
            Ps[i] = temperature_skew(Ps[i], temps[i])
        for i in range(n_top_ps):
            Ps[i + n_temps] = top_p_skew(Ps[i + n_temps], top_ps[i])
        ixs = np.array([np.random.choice(len(P), p = P) for P in Ps])
        preds.append(ixs)

        seed = np.hstack((seed, np.expand_dims(ixs, 1)))
        seed = np.roll(seed, -1, axis = 1)
        seed[:,-1] = ixs
    return [[int(preds[j][i]) for j in range(n_samples)]
            for i in range(n_temps + n_top_ps)]

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])

    params = ModelParams.from_docopt_args(args)
    _, _, data = load_training_data(params.code_type, path)
    temps = [0.90, 0.95, 1.0, 1.05, 1.10]
    top_ps = [0.85, 0.90, 0.94, 0.98, 0.99]

    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps
    n_samples = 1200
    n_seed = 256
    encoder = data.encoder
    vocab_size = len(encoder.ix2ch)
    seed_idx = args['--seed-idx']
    file_format = args['--file-format']

    SP.header('%d PREDICTIONS' % n_preds)
    #with select_strategy().scope():
    model = params.model(vocab_size, n_preds, True)
    weights_path = path / params.weights_file()
    SP.print('Loading weights from %s.' % weights_path)
    model.load_weights(str(weights_path))
    model.reset_states()

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

    if params.model_type == 'transformer':
        cont_fn = transformer_continuation
    else:
        cont_fn = lstm_continuation
    seqs = cont_fn(model, temps, top_ps, seed, n_samples)

    # Add the original
    orig = seq[seed_idx + n_seed:seed_idx + n_seed + n_samples]
    seqs = np.vstack((seqs, orig))

    # Cut seed in half cause it is long.
    seed = np.vstack((seed, seed[0]))
    seed = seed[:, n_seed // 2 :]

    join = np.repeat(np.expand_dims(pause, 0), len(seqs), axis = 0)
    seqs = np.hstack((seed, join, seqs))

    prefix = '%s-%09d' % (data.code_type, seed_idx)
    file_names = ['%s-t%.3f' % (prefix, t) for t in temps]
    file_names += ['%s-p%.3f' % (prefix, p) for p in top_ps]
    file_names.append('%s-orig' % prefix)

    file_names = ['%s.%s' % (f, file_format) for f in file_names]
    for file_name, seq in zip(file_names, seqs):
        data.save_code(seq, path / file_name)

if __name__ == '__main__':
    main()
