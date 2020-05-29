# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# I couldn't get the torrents working. Hence this tool.
"""The Mod Archive download tool

Usage:
    manage-corpus.py [-hv] download <corpus-path>
        [--format=<s> --kb-limit=<i>]
    manage-corpus.py [-hv] update-index <corpus-path>
        ( --genre-id=<i> | --random=<i> | --module-id=<i> )

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --format=<s>           format selection; AHX, IT, MOD, S3M or XM
    --kb-limit=<kb>        only download song smaller than the specified
                           size in kb [default: 150]
    --genre-id=<i>         The Mod Archive genre id to scrape
    --module-id=<i>        The Mod Archive module id to scrape
    --random=<i>           Number of random modules to scrape
"""
from docopt import docopt
from pathlib import Path
from musicgen.corpus import *
from musicgen.utils import SP

def main():
    args = docopt(__doc__, version = 'The Mod Archive download tool 1.0')
    SP.enabled = args['--verbose']

    corpus_path = Path(args['<corpus-path>'])
    corpus_path.mkdir(parents = True, exist_ok = True)

    if args['download']:
        kb_limit = int(args['--kb-limit'])
        download_mods(corpus_path, args['--format'], kb_limit)
    elif args['update-index']:
        genre_id = args['--genre-id']
        n_random = args['--random']
        module_id = args['--module-id']
        index = load_index(corpus_path)
        if genre_id is not None:
            SP.header('GENRE', '%d', genre_id)
            mods = modules_for_genre(genre_id)
        elif module_id is not None:
            SP.header('MODULE', '%d', module_Id)
            mods = [IndexedModule.from_modarchive(module_id)]
        elif n_random is not None:
            SP.header('RANDOM', '%d', n_random)
            mods = [IndexedModule.from_modarchive_random()
                    for _ in range(n_random)]
        SP.leave()
        SP.header('%d ENTRIES' % len(mods))
        for mod in mods:
            SP.print(short_line(mod))
        SP.leave()
        for mod in mods:
            index[mod.id] = mod
        save_index(corpus_path, index)

if __name__ == '__main__':
    main()
