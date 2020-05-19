# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.utils import StructuredPrinter, parse_comma_list
from musicgen.defs import (EFFECT_CMD_JUMP_TO_OFFSET,
                           EFFECT_CMD_JUMP_TO_ROW,
                           EFFECT_CMD_SET_VOLUME,
                           EFFECT_CMD_UPDATE_TIMING)
from musicgen.parser import load_file, save_file
from musicgen.prettyprint import row_to_string

def zero_effect(cell):
    cell.effect_cmd = 0
    cell.effect_arg1 = 0
    cell.effect_arg2 = 0

def strip_column(rows, col_idx, sample_indices, col_indices):
    removed_col = col_idx not in col_indices
    current_sample = None
    for row in rows:
        cell = row[col_idx]
        if cell.sample_idx != 0:
            current_sample = cell.sample_idx
        removed_sample = current_sample not in sample_indices
        if removed_col or removed_sample:
            cell.sample_hi = 0
            cell.sample_lo = 0
            cell.sample_idx = 0
            cell.period = 0
        midi_compatible_commands = (EFFECT_CMD_JUMP_TO_OFFSET,
                                    EFFECT_CMD_SET_VOLUME,
                                    EFFECT_CMD_JUMP_TO_ROW,
                                    EFFECT_CMD_UPDATE_TIMING)
        if cell.effect_cmd not in midi_compatible_commands:
            zero_effect(cell)

        if (cell.effect_cmd == EFFECT_CMD_SET_VOLUME
            and (removed_sample or removed_col)):
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
    col_indices = list(range(4))
    if args.channels:
        col_indices = [c - 1 for c in parse_comma_list(args.channels)]

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
        for i in range(4):
            strip_column(pattern.rows, i, sample_indices, col_indices)

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
