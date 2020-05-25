# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Algorithm to find SSRs. Invented by Tim Peters.
# https://stackoverflow.com/questions/61758735/find-longest-adjacent-repeating-non-overlapping-substring/61765774
from musicgen.suffix_array import lcp_array, suffix_array

# Generate lcp intervals from the lcp array.
def genlcpi_9(lcp):
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

def crunch_9(text):
    sa = suffix_array(text)
    rank, lcp = lcp_array(text, sa)
    bestlen, results = 0, []
    n = len(text)

    # generate branching tandem repeats
    def gen_btr(text=text, n=n, sa=sa):
        for c, lb, rb in genlcpi_9(lcp):
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

def genlcpi_11(lcp):
    lcp.append(0)
    stack = [(0, 0)]
    for i in range(1, len(lcp)):
        c = lcp[i]
        lb = i - 1
        while c < stack[-1][0]:
            i_c, lb = stack.pop()
            yield (i_c, lb, i)
        if c > stack[-1][0]:
            stack.append((c, lb))
    lcp.pop()

def crunch_11(text):
    sa = suffix_array(text)
    rank, lcp = lcp_array(text, sa)
    bestlen, results = 0, []
    n = len(text)

    # Generate branching tandem repeats.
    # (i, c, 2) is branching tandem iff
    #     i+c in interval with prefix text[i : i+c], and
    #     i+c not in subinterval with prefix text[i : i+c + 1]
    # Caution: this pragmatically relies on that, in Python 3,
    # `range()` returns a tiny object with O(1) membership testing.
    # In Python 2 it returns a list - ahould still work, but very
    # much slower.
    def gen_btr(text=text, n=n, sa=sa, rank=rank):
        from itertools import chain

        for c, lb, rb in genlcpi_11(lcp):
            origlb, origrb = lb, rb
            origrange = range(lb, rb)
            i = sa[lb]
            if type(text) == list:
                lead = [text[i]]
            else:
                lead = text[i]
            # Binary searches to find subrange beginning with
            # text[i : i+c+1]. Note we take slices of length 1
            # rather than just index to avoid special-casing for
            # i >= n.
            # A more elaborate traversal of the lcp array could also
            # give us a list of child intervals, and then we'd just
            # need to pick the right one. But that would be even
            # more hairy code, and unclear to me it would actually
            # help the worst cases (yes, the interval can be large,
            # but so can a list of child intervals).
            hi = rb
            while lb < hi:  # like bisect.bisect_left
                mid = (lb + hi) // 2
                i = sa[mid] + c
                if text[i : i+1] < lead:
                    lb = mid + 1
                else:
                    hi = mid
            lo = lb
            while lo < rb:  # like bisect.bisect_right
                mid = (lo + rb) // 2
                i = sa[mid] + c
                if lead < text[i : i+1]:
                    rb = mid
                else:
                    lo = mid + 1
            subrange = range(lb, rb)
            if 2 * len(subrange) <= len(origrange):
                # Subrange is at most half the size.
                # Iterate over it to find candidates i, starting
                # with wa.  If i+c is also in origrange, but not
                # in subrange, good:  then i is of the form wwx.
                for sai in subrange:
                    i = sa[sai]
                    ic = i + c
                    if ic < n:
                        r = rank[ic]
                        if r in origrange and r not in subrange:
                            yield (i, c, 2, subrange)
            else:
                # Iterate over the parts outside subrange instead.
                # Candidates i are then the trailing wx in the
                # hoped-for wwx. We win if i-c is in subrange too
                # (or, for that matter, if it's in origrange).
                for sai in chain(range(origlb, lb),
                                 range(rb, origrb)):
                    ic = sa[sai] - c
                    if ic >= 0 and rank[ic] in subrange:
                        yield (ic, c, 2, subrange)

    for start, c, numreps, irange in gen_btr():
        # extend left
        crange = range(start - c, -1, -c)
        if (numreps + len(crange)) * c < bestlen:
            continue
        for i in crange:
            if rank[i] in irange:
                start = i
                numreps += 1
            else:
                break
        # check for best
        totallen = c * numreps
        if totallen < bestlen:
            continue
        if totallen > bestlen:
            bestlen = totallen
            results.clear()
        results.append((start, c, numreps))
        # add non-branches
        while start and text[start - 1] == text[start + c - 1]:
                start -= 1
                results.append((start, c, numreps))
    return bestlen, results


# Find the SSR with minimum weight.
def find_min_ssr(seq):
    bestlen, ssrs = crunch_11(seq)
    default = 0, len(seq), 1
    return min(ssrs, key = lambda x: (x[1], x[0]), default = default)
