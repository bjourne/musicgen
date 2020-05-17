# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Tool for analyzing mycode.
"""MyCode analyzer

Usage:
    analyze-mycode.py [-v] <corpus/module>

Options:
    -h --help              show this screen
    -v --verbose           print more output
"""
from docopt import docopt
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
    args = docopt(__doc__, version = 'MyCode analyzer 1.0')
    SP.enabled = args['--verbose']

    path = Path(args['<corpus/module>'])
    if path.is_dir():
        mycode = corpus_to_mycode(path, 150)
    else:
        mycode = mod_file_to_mycode(path)
    analyze_mycode(list(mycode))

if __name__ == '__main__':
    main()
