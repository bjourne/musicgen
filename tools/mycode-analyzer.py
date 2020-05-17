# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Tool for converting stuff to mycode
from argparse import ArgumentParser, FileType
from collections import Counter
from musicgen.mycode import corpus_to_mycode, mod_file_to_mycode
from musicgen.utils import SP
from pathlib import Path
from termtables import print as tt_print
from termtables.styles import markdown

def analyze_mycode(mycode):
    code_counts = Counter(mycode)
    total = sum(code_counts.values())

    header = ['Command', 'Argument', 'Count', 'Freq.']
    data = [(cmd, arg, v, '%.5f' % (v / total))
            for ((cmd, arg), v) in code_counts.items()]
    data = sorted(data)
    tt_print(data,
             padding = (0, 1),
             alignment = 'lrrr',
             style = markdown,
             header = header)
    print('%d tokens and %d token types.' %
          (len(mycode), len(set(mycode))))

def main():
    parser = ArgumentParser(
        description = 'MyCode Analyzer')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'Print information')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--module', type = FileType('rb'),
        help = 'Path to module to analyze')
    group.add_argument(
        '--corpus-path',
        help = 'Path to corpus to analyze')
    args = parser.parse_args()
    SP.enabled = args.info

    if args.module:
        args.module.close()
        mycode = list(mod_file_to_mycode(args.module.name))
    elif args.corpus_path:
        mycode = corpus_to_mycode(Path(args.corpus_path), 150)
    analyze_mycode(mycode)

if __name__ == '__main__':
    main()
