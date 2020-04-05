# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module

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

    if (cell.sample_idx == 0 and cell.effect_cmd not in (11, 13)
        or cell.effect_cmd in (3, 4, 5, 6, 7)):
        zero_effect(cell)

def main():
    parser = ArgumentParser(description='Module stripper')
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    parser.add_argument('--samples',
                        required = True,
                        help = 'Samples to keep')
    parser.add_argument('--pattern-table',
                        required = True,
                        help = 'Pattern table')
    parser.add_argument('--info',
                        help = 'Print module information',
                        action = 'store_true')
    args = parser.parse_args()
    with args.input as inf:
        mod = Module.parse(inf.read())

    # Parse pattern indices
    pattern_indices = [int(p) for p in args.pattern_table.split(',')]

    if args.info:
        old_pattern_indices = [mod.pattern_table[i]
                               for i in range(mod.n_played_patterns)]
        print(old_pattern_indices)
        for idx in pattern_indices:
            print(f'Pattern #{idx}')
            print(rows_to_string(mod.patterns[idx].rows))

    sample_indices = [int(s) for s in args.samples.split(',')]
    for pattern in mod.patterns:
        for row in pattern.rows:
            for cell in row:
                strip_cell(cell, sample_indices)

    # Create new pattern table

    n_played_patterns = len(pattern_indices)

    patterns = []
    old2new = {}
    at = 0
    for idx in pattern_indices:
        if idx not in old2new:
            old2new[idx] = at
            at +=1
            patterns.append(mod.patterns[idx])
    pattern_indices = [old2new[p] for p in pattern_indices]
    mod.patterns = patterns
    mod.n_played_patterns = n_played_patterns
    pattern_indices += [0] * (128 - n_played_patterns)
    mod.pattern_table = bytearray(pattern_indices)

    with args.output as outf:
        outf.write(Module.build(mod))

if __name__ == '__main__':
    main()
