# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""MOD sample extractor

Usage:
    extract-samples.py [-v] <corpus/module>

Options:
    -h --help              show this screen
    -v --verbose           print more output
"""
from docopt import docopt
from musicgen.defs import SAMPLE_RATE
from musicgen.parser import load_file
from musicgen.samples import load_samples
from musicgen.utils import SP
from pathlib import Path
from wave import open as wave_open
import numpy as np

def write_sample(sample, fname):
    with wave_open(fname, 'wb') as sfile:
        sfile.setframerate(SAMPLE_RATE)
        sfile.setnchannels(1)
        sfile.setsampwidth(2)
        sfile.writeframes(sample.arr.astype(np.int16))

def extract_mod_file_samples(mod_file):
    mod = load_file(mod_file)
    samples = load_samples(mod)

    name_prefix = mod_file.stem
    for idx, sample in enumerate(samples):
        fname = '%s-%02d.wav' % (name_prefix, idx + 1)
        write_sample(sample, fname)

def main():
    args = docopt(__doc__, version = 'MOD sample extractor 1.0')
    SP.enabled = args['--verbose']

    path = Path(args['<corpus/module>'])
    if path.is_dir():
        # Not implemented yet
        pass
    else:
        extract_mod_file_samples(path)

if __name__ == '__main__':
    main()
