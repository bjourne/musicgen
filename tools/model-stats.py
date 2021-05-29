# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
"""
Model stats
===========
Usage:
    model-stats.py [options] <root-path> <model>

Options:
    -h --help              show this screen
    -v --verbose           print more output

Generates plots and figures for the given model and saves them in
the directory <root-path>/stats.
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from docopt import docopt
from matplotlib.ticker import MaxNLocator
from musicgen.code_generators import (file_stem,
                                      get_code_generator,
                                      log_file,
                                      weights_file)
from musicgen.training_data import (TrainingData,
                                    load_training_data,
                                    random_rel_ofs,
                                    tally_tokens)
from musicgen.utils import SP, load_pickle_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def loss_plot(log_file, png_path):
    lines = [l.split() for l in open(log_file, 'rt')]
    losses = [float(l[1]) for l in lines]
    val_losses = [float(l[2]) for l in lines]

    fig, ax = plt.subplots(figsize = (8, 4))
    n = len(losses)
    ax.plot(range(1, n + 1), losses, label = 'Training loss')
    ax.plot(range(1, n + 1), val_losses, label = 'Validation loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax.set(xlabel = 'epoch', ylabel = 'loss')
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(png_path)

def token_distribution_plot(td, png_path):
    counts = tally_tokens(td.encoder, td.songs)
    names = ['%s%s' % ch for (ch, _) in counts]
    values = [v for (_, v) in counts]

    tot = sum(values)
    values = [v / tot for v in values]

    type_colors = {
        'D' : 'C0',
        'P' : 'C1',
        'R' : 'C1',
        'S' : 'C2',
        'X' : 'C3'
    }

    fig, ax = plt.subplots(figsize = (12, 4))
    bars = ax.bar(np.arange(len(values)), values, width = 0.80)
    for bar, name in zip(bars, names):
        bar.set_color(type_colors[name[0]])
    ax.set_xticks(range(0, len(values)))
    ax.set_xticklabels(names, rotation = 45,
                       rotation_mode = 'anchor',
                       ha = 'right')
    tot_fmt = '{:,}'.format(sum(values))
    ax.set(xlabel = 'token', ylabel = 'freq.')
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(png_path)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Model stats 1.0')
    SP.enabled = args['--verbose']
    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<model>'])

    # Load training data
    td = TrainingData(g['code-type'])
    td.load_disk_cache(root_path, 150)

    stats_path = root_path / 'stats'
    stats_path.mkdir(exist_ok = True)

    png_path = stats_path / ('tokens-%s.png' % g['code-type'])
    token_distribution_plot(td, png_path)

    weights_dir = root_path / 'weights'
    log_path = weights_dir / log_file(g)

    png_path = stats_path / ('loss-%s.png' % file_stem(g))
    loss_plot(log_path, png_path)

if __name__ == '__main__':
    main()
