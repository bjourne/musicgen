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
