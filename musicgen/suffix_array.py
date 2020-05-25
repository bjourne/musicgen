import numpy as np

def suffix_array(seq):
    if not seq:
        return []
    vocab = sorted(set(seq))

    ch2idx = {ch: i for i, ch in enumerate(vocab)}
    cls = np.array([ch2idx[t] for t in seq] + [-1])

    n = 1
    while n < len(seq):
        cls1 = np.roll(cls, -n)
        inds = np.lexsort((cls1, cls))
        result = np.logical_or(np.diff(cls[inds]),
                               np.diff(cls1[inds]))
        cls[inds[0]] = 0
        cls[inds[1:]] = np.cumsum(result)

        n *= 2
    cls1 = np.roll(cls, n // 2)
    return np.lexsort((cls1, cls))[1:].tolist()

def lcp_array(seq, sa):
    '''Returns both the rank array and the lcp array.'''
    n = len(sa)
    lcp = [0] * n
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    k = 0
    for i, rank_el in enumerate(rank):
        if rank_el == n - 1:
            k = 0
            continue
        j = sa[rank_el + 1]
        while i + k < n and j + k < n and seq[i + k] == seq[j + k]:
            k += 1
        lcp[rank_el] = k
        if k > 0:
            k -= 1
    lcp = [lcp[-1]] + lcp[:-1]
    return rank, lcp
