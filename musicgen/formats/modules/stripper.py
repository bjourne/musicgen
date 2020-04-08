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

    if cell.effect_cmd not in (11, 12, 13, 15):
        zero_effect(cell)
    if cell.sample_idx == 0 and cell.effect_cmd == 12:
        zero_effect(cell)

def main():
    parser = ArgumentParser(description='Module stripper')
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    parser.add_argument('--samples',
                        required = False,
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

    # Parse sample indices
    if not args.samples:
        sample_indices = list(range(32))
    else:
        sample_indices = [int(s) for s in args.samples.split(',')]

    # Print pattern table
    if args.info:
        pattern_table = [mod.pattern_table[i]
                         for i in range(mod.n_played_patterns)]
        s = ', '.join(map(str, pattern_table))
        print(f'Pattern table: {s}')
        for idx in sorted(set(pattern_indices)):
            print(f'Pattern #{idx}:')
            print(rows_to_string(mod.patterns[idx].rows))

    # Install new pattern table
    new_patterns = []
    old2new = {}
    at = 0
    for idx in pattern_indices:
        if idx not in old2new:
            old2new[idx] = at
            at += 1
            new_patterns.append(mod.patterns[idx])
    new_pattern_table = [old2new[p] for p in pattern_indices]
    mod.patterns = new_patterns
    n_played_patterns = len(new_pattern_table)
    mod.n_played_patterns = n_played_patterns
    new_pattern_table += [0] * (128 - n_played_patterns)
    mod.pattern_table = bytearray(new_pattern_table)

    # Strip effects
    for pattern in mod.patterns:
        for row in pattern.rows:
            for cell in row:
                strip_cell(cell, sample_indices)

    with args.output as outf:
        outf.write(Module.build(mod))

if __name__ == '__main__':
    main()
