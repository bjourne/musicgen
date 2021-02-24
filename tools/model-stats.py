# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
"""
Model stats
===========
Usage:
    model-stats.py [options] <code-type> lstm <corpus-path>
        --emb-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-stats.py [options] <code-type> transformer <corpus-path>
    model-stats.py [options] <code-type> gpt2 <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --lr=<f>               learning rate
    --epochs=<i>           epochs to train for
    --seq-len=<i>          training sequence length
    --batch-size=<i>       size of training batches
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from docopt import docopt
from musicgen.params import ModelParams
from musicgen.training_data import TrainingData, tally_tokens
from musicgen.utils import SP
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
    ax.set(xlabel = 'epoch', ylabel = 'loss',
           title = 'Categorical cross-entropy loss')
    ax.grid()
    ax.legend()
    fig.savefig(png_path)

def token_distribution_plot(td, png_path):
    counts = tally_tokens(td.encoder, td.data)
    names = ['%s%s' % ch for (ch, _) in counts]
    values = [v for (_, v) in counts]

    tot = sum(values)
    print('tot', tot)
    values = [v / tot for v in values]

    type_colors = {'D' : 'C0', 'P' : 'C1', 'S' : 'C2', 'X' : 'C3'}

    fig, ax = plt.subplots(figsize = (12, 6))
    bars = ax.bar(np.arange(len(values)), values, width = 0.80)
    for bar, name in zip(bars, names):
        bar.set_color(type_colors[name[0]])
    ax.set_xticks(range(0, len(values)))
    ax.set_xticklabels(names, rotation = 45,
                       rotation_mode = 'anchor',
                       ha = 'right')
    tot_fmt = '{:,}'.format(sum(values))
    ax.set(xlabel = 'token', ylabel = 'freq.') #, title = title)
    ax.grid()
    fig.savefig(png_path)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Model stats 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])

    # Hyperparameters
    params = ModelParams.from_docopt_args(args)

    td = TrainingData(params.code_type)
    td.load_disk_cache(path, 150)

    png_path = path / 'tokens.png'
    token_distribution_plot(td, png_path)

    log_path = path / params.log_file()
    png_path = path / ('loss-%s.png' % params.to_string())
    loss_plot(log_path, png_path)

if __name__ == '__main__':
    main()
