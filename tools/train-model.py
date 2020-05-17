# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Model trainer

Usage:
    train-model.py [-v --programs=<seq>] --win-size=<int> <corpus-path>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --programs=<seq>       melodic and percussive programs
                           [default: 1,36:40,36,31]
"""
from docopt import docopt
from musicgen.generation import parse_programs
from musicgen.ml import train_model
from musicgen.utils import SP
from os import environ
from pathlib import Path

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    args = docopt(__doc__, version = 'MOD model builder 1.0')
    SP.enabled = args['--verbose']

    corpus_path = Path(args['<corpus-path>'])
    programs = parse_programs(args['--programs'])

    # Should step be configurable too?
    win_size = int(args['--win-size'])
    step = 1

    train_model(corpus_path, win_size, 1, 128, programs)

if __name__ == '__main__':
    main()
