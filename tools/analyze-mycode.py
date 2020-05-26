# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Tool for analyzing mycode.
"""MyCode analyzer

Usage:
    analyze-mycode.py [-v] [--kb-limit=<int> --pack-mycode --print]
        <corpus/module>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --kb-limit=<int>       kb limit [default: 150]
    --pack-mycode          use packed mycode
    --print                print the code
"""
from collections import Counter
from docopt import docopt
from musicgen.mycode import (INSN_PROGRAM,
                             corpus_to_mycode_mods,
                             mod_file_to_mycode,
                             mycode_to_string)
from musicgen.utils import SP, flatten
from pathlib import Path
from termtables import print as tt_print
from termtables.styles import markdown

def analyze_mycode(mycode):
    print('%d tokens and %d token types.' %
          (len(mycode), len(set(mycode))))
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

def main():
    args = docopt(__doc__, version = 'MyCode analyzer 1.0')
    SP.enabled = args['--verbose']

    kb_limit = int(args['--kb-limit'])
    do_pack = args['--pack-mycode']
    path = Path(args['<corpus/module>'])
    if path.is_dir():
        mods = corpus_to_mycode_mods(path, kb_limit, do_pack)
    else:
        mods = [mod_file_to_mycode(path, do_pack)]

    pad_token = INSN_PROGRAM, 0
    seqs = [[c[1] + [pad_token] for c in mod.cols] for mod in mods]
    seq = flatten(flatten(seqs))
    analyze_mycode(seq)

    if args['--print']:
        SP.print(mycode_to_string(seq))

if __name__ == '__main__':
    main()
