# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# I couldn't get the torrents working. Hence this tool.
from collections import Counter, defaultdict
from lxml import html
from musicgen.utils import SP
from re import sub
from requests import get
from sys import exit
from time import sleep

FIRST_RESULTS_FMT = 'https://modarchive.org' \
    + '/index.php?query=%d&request=search&search_type=genre'

RESULTS_PAGE_FMT = 'https://modarchive.org' \
    '/index.php?query=%d&request=search&search_type=genre&page=%d'

def get_url(url, delay = 0.4):
    SP.print('GET %s' % url)
    r = get(url)
    sleep(delay)
    return r

def get_and_parse_url(url):
    return html.fromstring(get_url(url).text)

def parse_mod_stats(el):
    els = el.xpath('li[@class="stats"]/text()')
    keyvals = [[t.strip() for t in el.split(':')] for el in els]
    return {k : v for [k, v] in keyvals}

def clean_genre(str):
    str = sub(r'\W', ' ', str.lower())
    genre = '-'.join(str.split())
    return 'uncategorized' if genre == 'n-a' else genre

class IndexedModule:
    URL_SITE = 'https://modarchive.org'
    URL_RANDOM_PAGE = '%s/index.php?request=view_random' % (URL_SITE)
    @classmethod
    def from_line(cls, line):
        toks = line.split()
        return IndexedModule(
            toks[0], int(toks[1]), toks[2], int(toks[3]),
            int(toks[4]), toks[5], int(toks[6]), int(toks[7]),
            toks[8], toks[9])

    @classmethod
    def from_modarchive_url(cls, url):
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

    @classmethod
    def from_modarchive_random(cls):
        return cls.from_modarchive_url(cls.URL_RANDOM_PAGE)

    @classmethod
    def from_modarchive(cls, id):
        url_fmt = 'https://modarchive.org' \
            '/index.php?request=view_by_moduleid&query=%d'
        url = url_fmt % id
        return cls.from_modarchive_url(url)

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

def short_line(mod):
    return '%-40s %3d %2d %4d %-12s' % (mod.fname, mod.kb_size,
                                        mod.n_channels,
                                        mod.year,
                                        mod.genre)

def long_line(mod):
    fmt = '%-55s %6d %-4s %5d %2d %-20s %4d %5d %2s %2s\n'
    args = (mod.fname, mod.id, mod.format, mod.kb_size,
            mod.n_channels,
            mod.genre, mod.year, mod.n_downloads,
            mod.member_rating, mod.reviewer_rating)
    return fmt % args

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
    with open(index_file, 'wt') as f:
        for mod in mods:
            args = (mod.fname, mod.id, mod.format, mod.kb_size,
                    mod.n_channels,
                    mod.genre, mod.year, mod.n_downloads,
                    mod.member_rating, mod.reviewer_rating)
            f.write(long_line(mod))

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
    return sum([modules_for_genre_page(genre_id, page_id)
                for page_id in page_indices], [])

def download_mods(corpus_path, selected_format, max_size):
    index = load_index(corpus_path)
    def local_path(mod):
        return corpus_path / mod.genre / mod.fname

    def classify_indexed_mod(mod):
        if selected_format and selected_format != mod.format:
            return 'format'
        elif local_path(mod).exists():
            return 'exists'
        elif max_size and mod.kb_size > max_size:
            return 'size'
        else:
            return 'download'

    mods = index.values()
    mod_classes = {m : classify_indexed_mod(m) for m in mods}

    tally = Counter(mod_classes.values())
    n_skips = tally['exists'] + tally['size'] + tally['format']
    SP.header('%d SKIPS' % n_skips)
    SP.print('Exists       : %5d' % tally['exists'])
    SP.print('Over max size: %5d' % tally['size'])
    SP.print('Wrong format : %5d' % tally['format'])
    SP.leave()

    SP.header('%d DOWNLOADS' % tally['download'])
    new_mods = {m for m in mods if mod_classes[m] == 'download'}
    for mod in new_mods:
        SP.print(short_line(mod))

    url_fmt = 'https://api.modarchive.org/downloads.php?moduleid=%d'
    for mod in new_mods:
        file_path = local_path(mod)
        file_path.parent.mkdir(parents = True, exist_ok = True)
        url = url_fmt % mod.id
        with open(file_path, 'wb') as f:
            f.write(get_url(url, 0.4).content)
    SP.leave()
