# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Random utils.
from collections import namedtuple, defaultdict
from functools import reduce
from itertools import chain, groupby, islice
from musicgen.suffix_array import lcp_array, suffix_array
from operator import iconcat
from pickle import dump, load
import numpy as np

class StructuredPrinter:
    def __init__(self, enabled):
        self.indent = 0
        self.enabled = enabled

    def print_indented(self, text):
        if self.enabled:
            print(' ' * self.indent + text)

    def header(self, name, fmt = None, args = None):
        if fmt is not None:
            self.print_indented('* %s %s' % (name, fmt % args))
        else:
            self.print_indented('* %s' % name)
        self.indent += 2

    def print(self, fmt, args = None):
        if args is not None:
            s = fmt % args
        else:
            s = str(fmt)
        self.print_indented(s)

    def leave(self):
        self.indent -= 2

SP = StructuredPrinter(False)

def sort_groupby(seq, keyfun):
    return groupby(sorted(seq, key = keyfun), keyfun)

def flatten(seq):
    return reduce(iconcat, seq, [])

def parse_comma_list(seq):
    return [int(e) for e in seq.split(',')]

def load_pickle(pickle_path):
    assert pickle_path.exists()
    with open(pickle_path, 'rb') as f:
        return load(f)

def save_pickle(pickle_path, obj):
    with open(pickle_path, 'wb') as f:
        dump(obj, f)

def file_name_for_params(base, ext, params):
    def param_to_fmt(p):
        if type(p) == int:
            n = len(str(p))
            if n > 5:
                n = 10
            elif n > 3:
                n = 5
            else:
                n = 3
        else:
            return '%.2f'
        return '%%0%dd' % n
    strs = [param_to_fmt(p) % p for p in params]
    return '%s_%s.%s' % (base, '-'.join(strs), ext)

# https://stackoverflow.com/questions/61758735/find-longest-adjacent-repeating-non-overlapping-substring
def crunch2(s):
    from collections import deque

    # There are zcount equal characters starting
    # at index starti.
    def update(starti, zcount):
        nonlocal bestlen
        while zcount >= width:
            numreps = 1 + zcount // width
            count = width * numreps
            if count >= bestlen:
                if count > bestlen:
                    results.clear()
                results.append((starti, width, numreps))
                bestlen = count
            else:
                break
            zcount -= 1
            starti += 1

    bestlen, results = 0, []
    t = deque(s)
    for width in range(1, len(s) // 2 + 1):
        t.popleft()
        zcount = 0
        for i, (a, b) in enumerate(zip(s, t)):
            if a == b:
                if not zcount: # new run starts here
                    starti = i
                zcount += 1
            # else a != b, so equal run (if any) ended
            elif zcount:
                update(starti, zcount)
                zcount = 0
        if zcount:
            update(starti, zcount)
    return bestlen, results

def crunch6(seq):
    sa = suffix_array(seq)
    lcp = lcp_array(seq, sa)
    bestlen, results = 0, []
    n = len(seq)

    # Generate maximal sets of indices s such that for all i and j
    # in s the suffixes starting at s[i] and s[j] start with a
    # common prefix of at least len minc.
    def genixs(minc, sa=sa, lcp=lcp, n=n):
        i = 1
        while i < n:
            c = lcp[i]
            if c < minc:
                i += 1
                continue
            ixs = {sa[i-1], sa[i]}
            i += 1
            while i < n:
                c = min(c, lcp[i])
                if c < minc:
                    yield ixs
                    i += 1
                    break
                else:
                    ixs.add(sa[i])
                    i += 1
            else: # ran off the end of lcp
                yield ixs

    # Check an index set for _adjacent_ repeated substrings
    # w apart.  CAUTION: this empties s.
    def check(s, w):
        nonlocal bestlen
        while s:
            current = start = s.pop()
            count = 1
            while current + w in s:
                count += 1
                current += w
                s.remove(current)
            while start - w in s:
                count += 1
                start -= w
                s.remove(start)
            if count > 1:
                total = count * w
                if total >= bestlen:
                    if total > bestlen:
                        results.clear()
                        bestlen = total
                    results.append((start, w, count))

    c = 0
    found = True
    while found:
        c += 1
        found = False
        for s in genixs(c):
            found = True
            check(s, c)
    return bestlen, results

def genlcpi(lcp):
    lcp.append(0)
    stack = [(0, 0)]
    for i in range(1, len(lcp)):
        c = lcp[i]
        lb = i - 1
        while c < stack[-1][0]:
            i_c, lb = stack.pop()
            interval = i_c, lb, i - 1
            yield interval
        if c > stack[-1][0]:
            stack.append((c, lb))
    lcp.pop()

def crunch9(text):
    sa = suffix_array(text)
    lcp = lcp_array(text, sa)
    bestlen, results = 0, []
    n = len(text)

    # generate branching tandem repeats
    def gen_btr(text=text, n=n, sa=sa):
        for c, lb, rb in genlcpi(lcp):
            i = sa[lb]
            basic = text[i : i + c]
            # Binary searches to find subrange beginning with
            # basic+basic. A more gonzo implementation would do this
            # character by character, never materialzing the common
            # prefix in `basic`.
            rb += 1
            hi = rb
            while lb < hi:  # like bisect.bisect_left
                mid = (lb + hi) // 2
                i = sa[mid] + c
                if text[i : i + c] < basic:
                    lb = mid + 1
                else:
                    hi = mid
            lo = lb
            while lo < rb:  # like bisect.bisect_right
                mid = (lo + rb) // 2
                i = sa[mid] + c
                if basic < text[i : i + c]:
                    rb = mid
                else:
                    lo = mid + 1
            lead = basic[0]
            for sai in range(lb, rb):
                i = sa[sai]
                j = i + 2*c
                assert j <= n
                if j < n and text[j] == lead:
                    continue # it's branching
                yield (i, c, 2)

    for start, c, _ in gen_btr():
        # extend left
        numreps = 2
        for i in range(start - c, -1, -c):
            if all(text[i+k] == text[start+k] for k in range(c)):
                start = i
                numreps += 1
            else:
                break
        totallen = c * numreps
        if totallen < bestlen:
            continue
        if totallen > bestlen:
            bestlen = totallen
            results.clear()
        results.append((start, c, numreps))
        # add branches
        while start:
            if text[start - 1] == text[start + c - 1]:
                start -= 1
                results.append((start, c, numreps))
            else:
                break
    return bestlen, results

def find_min_ssr(seq):
    bestlen, ssrs = crunch9(seq)
    if bestlen == 0:
        return 0, len(seq), 1
    return sorted(ssrs, key = lambda x: (x[1], x[0]))[0]

def test_find_min_ssr():
    examples = [
        ('EEEFGAFFGAFFGAFCD', (3, 4, 3)),
        ('ACCCCCCCCCA', (1, 1, 9)),
        ('ABCD', (0, 4, 1)),
        ('BAMBAMBAMBAM', (0, 3, 4)),
        ('AAAAAAAAAAAA', (0, 1, 12)),
        ('ABBBCABBBC', (0, 5, 2)),
        ('ABBBC', (1, 1, 3)),
        (['P2', 'P2', 'P2', 'P0', 'P0', 'P0',
          'P-2', 'P-2', 'P-2', 'P0', 'P0', 'P0'],
         (0, 1, 3))
        ]
    for seq, best_ssr in examples:
        min_ssr = find_min_ssr(seq)
        print(seq, min_ssr)
        assert min_ssr == best_ssr

if __name__ == '__main__':
    test_find_min_ssr()
