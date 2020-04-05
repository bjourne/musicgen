# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module
from pathlib import Path

def zero_effect(cell):
    cell.effect_cmd = 0
    cell.effect_arg1 = 0
    cell.effect_arg2 = 0

def strip_cell(cell, sample_indices):
    if cell.sample_idx not in sample_indices:
        cell.sample_hi = 0
        cell.sample_lo = 0
        cell.sample_idx = 0
        cell.period = 0
    if cell.sample_idx == 0:
        zero_effect(cell)

def main():
    parser = ArgumentParser(description='Module stripper')
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    parser.add_argument('--samples',
                        required = True,
                        help = 'Samples to keep')
    args = parser.parse_args()
    with args.input as inf:
        mod = Module.parse(inf.read())

    rows = linearize_rows(mod)[:64]
    print(rows_to_string(rows))

    sample_indices = [int(s) for s in args.samples.split(',')]
    for pattern in mod.patterns:
        for row in pattern.rows:
            for cell in row:
                strip_cell(cell, sample_indices)

    with args.output as outf:
        outf.write(Module.build(mod))

if __name__ == '__main__':
    main()
