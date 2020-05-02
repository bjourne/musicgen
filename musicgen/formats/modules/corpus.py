# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# I couldn't get the torrents working. Hence this tool.
from lxml import html
from musicgen.utils import StructuredPrinter
from pathlib import Path
from requests import get
from sys import exit
from time import sleep

FIRST_RESULTS_FMT = 'https://modarchive.org' \
    + '/index.php?query=%d&request=search&search_type=genre'

RESULTS_PAGE_FMT = 'https://modarchive.org' \
    '/index.php?query=%d&request=search&search_type=genre&page=%d'

SP = StructuredPrinter(True)

def get_and_parse_url(url):
    SP.print('GET %s' % url)
    r = get(url)
    sleep(0.2)
    return html.fromstring(r.text)

def parse_mod_stats(el):
    els = el.xpath('li[@class="stats"]/text()')
    keyvals = [[t.strip() for t in el.split(':')] for el in els]
    return {k : v for [k, v] in keyvals}

def clean_genre(str):
    toks = str.lower().split()
    toks = [t for t in toks if t != '-']
    return '-'.join(toks)

class IndexedModule:
    @classmethod
    def from_line(cls, line):
        toks = line.split()
        return IndexedModule(
            toks[0], int(toks[1]), toks[2], int(toks[3]),
            int(toks[4]), toks[5], int(toks[6]), int(toks[7]),
            toks[8], toks[9])

    @classmethod
    def from_modarchive(cls, id):
        url_fmt = 'https://modarchive.org' \
            '/index.php?request=view_by_moduleid&query=%d'
        url = url_fmt % id
        root = get_and_parse_url(url)

        # Parse file name
        fname = root.xpath('//h1[1]/span/text()')[0][1:-1]

        # Parse other metadata
        ul_el = root.xpath('//div[@class="mod-page-archive-info"]'
                           '/ul[@class="nolist"]')[1]
        stats = parse_mod_stats(ul_el)

        id = int(stats['Mod Archive ID'])
        n_channels = int(stats['Channels'])
        n_downloads = int(stats['Downloads'])
        format = stats['Format']
        kb_size = round(float(stats['Uncompressed Size'][:-2]))

        genre = clean_genre(stats['Genre'])

        # Upload year
        year = root.xpath(
            '//div[@class="mod-page-archive-info"]'
            '/ul/li[@class="stats"]/text()[last()]')[0]
        year = int(year.split()[-2])

        # Ratings
        els = root.xpath('//div[@class="mod-page-ratings"]/ul/li/text()')
        member_rating = els[1].strip()[1:-1]
        if member_rating == 'Unrated':
            member_rating = '?'
        else:
            member_rating = int(member_rating.split('/')[0].strip())

        reviewer_rating = els[3].strip()[1:-1]
        if reviewer_rating == 'Unrated':
            reviewer_rating = '?'
        else:
            print('Reviewer rating', reviewer_rating)
            reviewer_rating = int(reviewer_rating)

        return IndexedModule(fname, id, format, kb_size,
                             n_channels, genre, year, n_downloads,
                             member_rating, reviewer_rating)

    def __init__(self, fname, id, format, kb_size,
                 n_channels, genre, year, n_downloads,
                 member_rating, reviewer_rating):
        self.fname = fname
        self.id = id
        self.format = format
        self.kb_size = kb_size
        self.n_channels = n_channels
        self.genre = genre
        self.year = year
        self.n_downloads = n_downloads
        self.member_rating = member_rating
        self.reviewer_rating = reviewer_rating

def load_index(corpus_path):
    index_file = corpus_path / 'index'
    if not index_file.exists():
        SP.print('Empty index.')
        return {}
    with open(index_file, 'rt') as f:
        mods = [IndexedModule.from_line(line) for line in f]
    return {mod.id : mod for mod in mods}

def save_index(corpus_path, index):
    index_file = corpus_path / 'index'
    mods = sorted(index.values(), key = lambda x: x.fname)
    SP.print('Saving index with %d modules.' % len(mods))
    # name id format kb-size channels genre \
    #   year downloads u-rating r-rating
    line_fmt = '%-50s %6d %-4s %5d %2d %-10s %4d %5d %2s %2s\n'
    with open(index_file, 'wt') as f:
        for mod in mods:
            args = (mod.fname, mod.id, mod.format, mod.kb_size,
                    mod.n_channels,
                    mod.genre, mod.year, mod.n_downloads,
                    mod.member_rating, mod.reviewer_rating)
            line = line_fmt % args
            f.write(line)

def modules_for_genre_page(genre_id, page_id):
    url = RESULTS_PAGE_FMT % (genre_id, page_id)
    root = get_and_parse_url(url)

    rel_urls = root.xpath('//table/tr/td/a[@class="standard-link"]/@href')

    ids = [int(rel_url.split('=')[-1]) for rel_url in rel_urls]
    return [IndexedModule.from_modarchive(id) for id in ids]

def modules_for_genre(genre_id):
    url = FIRST_RESULTS_FMT % genre_id
    root = get_and_parse_url(url)

    els = root.xpath('//select[@class="pagination"]/option[last()]/@value')
    page_count = int(els[0])

    page_indices = list(range(1, page_count + 1))
    #page_indices = [1]

    return sum([modules_for_genre_page(genre_id, page_id)
                for page_id in page_indices], [])

def download_mods(corpus_path):
    index_file = corpus_path / f'test'
    if not index_file.exists():
        raise ValueError('Index file %s missing.' % index_file)

    with open(index_file, 'rt') as f:
        for line in f:
            imod = parse_index_line(line)

def main():
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description = 'Bulk MOD downloader')
    parser.add_argument(
        '--corpus-path', required = True,
        help = 'Path to corpus')
    parser.add_argument('--info',
                        help = 'Print information',
                        action = 'store_true')

    subparser = parser.add_subparsers(dest = 'subparser')

    sync_index = subparser.add_parser('update-index')
    group = sync_index.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--genre-id', type = int,
        help = 'The Mod Archive genre id')
    group.add_argument(
        '--module-id', type = int,
        help = 'The Mod Archive module id')

    download = subparser.add_parser('download')
    download.add_argument(
        '--max-size', type = int,
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

        index = load_index(corpus_path)
        if genre_id is not None:
            mods = modules_for_genre(genre_id)
        elif module_id is not None:
            mods = [IndexedModule.from_modarchive(module_id)]
        SP.print('Updating %d entries.' % len(mods))
        for mod in mods:
            index[mod.id] = mod
        save_index(corpus_path, index)
    elif cmd == 'download':
        download_mods(corpus_path)

if __name__ == '__main__':
    main()
