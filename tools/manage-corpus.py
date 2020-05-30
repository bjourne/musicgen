# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# I couldn't get the torrents working. Hence this tool.
"""The Mod Archive download tool

Usage:
    manage-corpus.py [-hv] download <corpus-path>
        [--format=<s> --kb-limit=<i>]
    manage-corpus.py [-hv] update-index <corpus-path>
        ( --genre-id=<i> | --random=<i> | --module-id=<i> )
    manage-corpus.py [-hv] print-stats <corpus-path>
        [--format=<s> --kb-limit=<i>]

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --format=<s>           only include songs in the specified format;
                           AHX, IT, MOD, S3M or XM
    --kb-limit=<kb>        only include songs smaller than the specified
                           size in kb [default: 150]
    --genre-id=<i>         The Mod Archive genre id to scrape
    --module-id=<i>        The Mod Archive module id to scrape
    --random=<i>           Number of random modules to scrape
"""
from collections import Counter
from docopt import docopt
from pathlib import Path
from musicgen.corpus import *
from musicgen.utils import SP, sort_groupby
from termtables import print as tt_print
from termtables.styles import markdown

def print_freq_table(seq, header, sort_key):
    counts = Counter(seq)
    total = sum(counts.values())
    header = [header] + ['Count', 'Freq.']
    rows = [(k, v, v / total) for (k, v) in counts.items()]
    rows = sorted(rows, key = sort_key)
    rows = [(k, v, '%.4f' % t) for (k, v, t) in rows]

    if type(seq[0]) == int:
        col_align = 'r'
    else:
        col_align = 'l'

    tt_print(rows,
             padding = (0, 1),
             alignment = col_align + 'rr',
             style = markdown,
             header = header)
    print()

def bin_value(value, thresholds):
    for threshold in thresholds:
        if value < threshold:
            return threshold
    assert False

def print_stats(mods, format, kb_limit):
    mods = [m for m in mods
            if m.kb_size <= kb_limit and
            (not format or m.format == format)]


    genres = [m.genre for m in mods]
    print_freq_table(genres, 'Genre', lambda x: -x[1])
    years = [m.year for m in mods]
    print_freq_table(years, 'Year', lambda x: x[0])

    size_bins = [10, 20, 50, 100, 200, 500, 1000, 5000]
    sizes = [bin_value(m.kb_size, size_bins) for m in mods]
    print_freq_table(sizes, 'Size <', lambda x: x[0])

    download_bins = [50, 100, 200, 500, 1000, 10000, 150000]
    downloads = [bin_value(m.n_downloads, download_bins) for m in mods]
    print_freq_table(downloads, 'Downloads <', lambda x: x[0])

def main():
    args = docopt(__doc__, version = 'The Mod Archive download tool 1.0')
    SP.enabled = args['--verbose']

    corpus_path = Path(args['<corpus-path>'])
    corpus_path.mkdir(parents = True, exist_ok = True)

    if args['download']:
        format = args['--format']
        kb_limit = int(args['--kb-limit'])
        download_mods(corpus_path, format, kb_limit)
    elif args['update-index']:
        genre_id = args['--genre-id']
        n_random = args['--random']
        module_id = args['--module-id']
        index = load_index(corpus_path)
        if genre_id is not None:
            SP.header('GENRE', '%d', genre_id)
            mods = modules_for_genre(genre_id)
        elif module_id is not None:
            module_id = int(module_id)
            SP.header('MODULE', '%d', module_id)
            mods = [IndexedModule.from_modarchive(module_id)]
        elif n_random is not None:
            n_random = int(n_random)
            SP.header('RANDOM', '%d', n_random)
            mods = [IndexedModule.from_modarchive_random()
                    for _ in range(n_random)]
        mods = [m for m in mods if m.id not in index]
        SP.leave()
        SP.header('%d ENTRIES' % len(mods))
        for mod in mods:
            SP.print(short_line(mod))
        SP.leave()
        for mod in mods:
            index[mod.id] = mod
        save_index(corpus_path, index)
    elif args['print-stats']:
        format = args['--format']
        kb_limit = int(args['--kb-limit'])
        index = load_index(corpus_path)
        print_stats(index.values(), format, kb_limit)


if __name__ == '__main__':
    main()
