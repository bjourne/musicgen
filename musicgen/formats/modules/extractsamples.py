# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module
from os.path import basename, splitext
from wave import open as wave_open
import numpy as np

def write_sample(sample, fname):
    with wave_open(fname, 'wb') as sfile:
        sfile.setframerate(SAMPLE_RATE)
        sfile.setnchannels(1)
        sfile.setsampwidth(2)
        sfile.writeframes(sample.arr.astype(np.int16))

def main():
    parser = ArgumentParser(
        description = 'Extract samples from MOD files')
    parser.add_argument('module', type = FileType('rb'))
    args = parser.parse_args()

    with args.module as inf:
        mod = Module.parse(inf.read())
    samples = load_samples(mod)
    name_prefix = splitext(basename(args.module.name))[0]
    for idx, sample in enumerate(samples):
        fname = '%s-%02d.wav' % (name_prefix, idx)
        write_sample(sample, fname)

if __name__ == '__main__':
    main()
