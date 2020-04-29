# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.utils import StructuredPrinter, parse_comma_list
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import load_file, save_file

def zero_effect(cell):
    cell.effect_cmd = 0
    cell.effect_arg1 = 0
    cell.effect_arg2 = 0

def strip_cell(cell, channel_idx, sample_indices, channel_indices):
    if ((cell.sample_idx not in sample_indices)
        or channel_idx not in channel_indices):
        cell.sample_hi = 0
        cell.sample_lo = 0
        cell.sample_idx = 0
        cell.period = 0

    if cell.effect_cmd not in (11, 12, 13, 15):
        zero_effect(cell)
    if cell.sample_idx == 0 and cell.effect_cmd == 12:
        zero_effect(cell)

def update_pattern_table(mod, pattern_indices):
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
    n_orders = len(new_pattern_table)
    mod.n_orders = n_orders
    new_pattern_table += [0] * (128 - n_orders)
    mod.pattern_table = bytearray(new_pattern_table)


def main():
    parser = ArgumentParser(description='Module stripper')
    parser.add_argument('input', type = FileType('rb'))
    parser.add_argument('output', type = FileType('wb'))
    parser.add_argument('--samples',
                        help = 'Samples to keep (default: all)')
    parser.add_argument('--pattern-table',
                        help = 'Pattern table (default: existing)')
    parser.add_argument('--channels',
                        help = 'Channels to keep (default: all)')
    parser.add_argument('--info',
                        help = 'Print module information',
                        action = 'store_true')
    args = parser.parse_args()
    args.input.close()
    args.output.close()
    mod = load_file(args.input.name)

    sp = StructuredPrinter(args.info)

    # Parse sample indices
    sample_indices = list(range(1, 32))
    if args.samples:
        sample_indices = parse_comma_list(args.samples)

    # Parse channel indices
    channel_indices = [1, 2, 3, 4]
    if args.channels:
        channel_indices = parse_comma_list(args.channels)

    # Print pattern table
    pattern_table = [mod.pattern_table[i] for i in range(mod.n_orders)]
    s = ' '.join(map(str, pattern_table))
    sp.print('Input pattern table: %s', s)

    if args.pattern_table:
        # Parse pattern indices
        pattern_indices = parse_comma_list(args.pattern_table)
        # Install new pattern table
        update_pattern_table(mod, pattern_indices)

    # Strip effects
    for pattern in mod.patterns:
        for row in pattern.rows:
            for i, cell in enumerate(row):
                strip_cell(cell, i + 1, sample_indices, channel_indices)

    sp.header('Output patterns')
    for idx, pattern in enumerate(mod.patterns):
        sp.header('Pattern', '%2d', idx)
        for row in pattern.rows:
            sp.print(row_to_string(row))
        sp.leave()
    sp.leave()

    save_file(args.output.name, mod)

if __name__ == '__main__':
    main()
