# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Crummy script for computing plagiarism ratios.
from gzip import decompress
from musicgen.utils import SP, flatten, load_pickle
from random import choice, randint, randrange
from sys import argv
from time import time

import numpy as np

def sample_seq(seqs, n):
    seq = choice(seqs)
    i = randrange(len(seq) - n)
    return np.array(seq[i:i+n], dtype = np.uint16)

def find_subseqs(sa, sb):
    start = 0
    n = len(sa)
    while start < n:
        i = sa.find(sb, start)
        if i == -1:
            return
        yield i
        start = i + 1

def find_samples(tokens, samples):
    sa = tokens.tostring()
    ea = tokens.itemsize
    n_matches = 0
    for samp in samples:
        #SP.header('FINDING SAMPLE %s' % samp)
        sb = samp.tostring()
        assert samp.itemsize == ea

        indices = list(find_subseqs(sa, sb))
        # Filter out matches on seams.
        indices = [i // ea for i in indices if (i / ea) == (i // ea)]
        nb = len(samp)
        nc = max(nb // 2, 1)

        #fmt = '%%-10s %%-%ds %%-%ds %%-%ds' % (nc * 4, nb * 4, nc * 4)
        # if len(indices):
        #     SP.print(fmt % ('Index', 'Prefix', 'Match', 'Postfix'))
        # for i in indices[:10]:
        #     pre = tokens[i - nc : i]
        #     match = tokens[i : i + nb]
        #     post = tokens[i + nb: i + nb + nc]
        #     SP.print(fmt % (i, pre, match, post))
        # if len(indices) > 10:
        #     SP.print('...')
        # SP.print('%d matches' % len(indices))
        # SP.leave()
        if indices:
            n_matches += 1
    return n_matches

def draw_plagiarism(lo_ngram, hi_ngram, code_gen, measurements, training_data):

    SP.print('Loading training data...')
    td = load_pickle(training_data)
    songs = td[1]

    # Strip names
    songs = [c for (n, c) in songs]

    SP.print('Flattening %d songs...' % len(songs))
    tokens = flatten(flatten(flatten(songs)))
    tokens = np.array(tokens, dtype = np.uint16)

    SP.print('Loading samples...')
    data = load_pickle(measurements)
    stats = data[code_gen]

    gen = stats[False]
    seqs = list(gen.values())
    seqs = [s[0] for s in seqs]

    n_samples = 1000
    plag_ratios = {}
    for ngram in range(lo_ngram, hi_ngram):
        SP.header('FINDING MATCHES FOR NGRAMS OF LENGTH %d' % ngram)
        samples = [sample_seq(seqs, ngram) for _ in range(n_samples)]
        n_matches = find_samples(tokens, samples)
        frac = n_matches / n_samples
        SP.print('%d samples matches, %.2f%%.' % (n_matches, 100 * frac))
        SP.leave()
        plag_ratios[ngram] = frac
    print(plag_ratios)


if __name__ == '__main__':
    SP.enabled = True
    draw_plagiarism(int(argv[1]), int(argv[2]),
                    argv[3], argv[4], argv[5])
