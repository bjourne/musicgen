# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# I couldn't get the torrents working. Hence this tool.
from argparse import ArgumentParser, FileType
from pathlib import Path
from musicgen.corpus import *
from musicgen.utils import SP

def main():
    parser = ArgumentParser(description = 'Bulk MOD downloader')
    parser.add_argument(
        '--corpus-path', required = True,
        help = 'Path to corpus')
    parser.add_argument(
        '--info', action = 'store_true',
        help = 'Print information')

    subparser = parser.add_subparsers(dest = 'subparser', required = True)
    sync_index = subparser.add_parser('update-index')
    group = sync_index.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--genre-id', type = int,
        help = 'The Mod Archive genre id')
    group.add_argument(
        '--module-id', type = int,
        help = 'The Mod Archive module id')
    group.add_argument(
        '--random', type = int,
        help = 'Random module from The Mod Archive')

    download = subparser.add_parser('download')
    download.add_argument(
        '--max-size', type = int,
        default = 150,
        help = 'Only download mods smaller than specified size (in KB)')
    formats = ['AHX', 'IT', 'MOD', 'S3M', 'XM']
    download.add_argument(
        '--format', choices = formats)

    args = parser.parse_args()

    SP.enabled = args.info
    corpus_path = Path(args.corpus_path)
    corpus_path.mkdir(parents = True, exist_ok = True)

    cmd = args.subparser
    if cmd == 'update-index':
        genre_id = args.genre_id
        module_id = args.module_id
        n_random = args.random

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
    elif cmd == 'download':
        download_mods(corpus_path, args.format, args.max_size)

if __name__ == '__main__':
    main()
