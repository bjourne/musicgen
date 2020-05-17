# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Sample management stuff.
from musicgen.defs import AMIGA_SAMPLE_RATE, SAMPLE_RATE
import numpy as np

class Sample:
    def __init__(self, bytes, repeat_from, repeat_len):
        arr = np.frombuffer(bytes, dtype = np.int8)
        arr = arr.astype(np.int16) * 256

        ratio = SAMPLE_RATE / AMIGA_SAMPLE_RATE
        n_samples = int(arr.size * ratio)
        if arr.size:
            x_old = np.linspace(0, 1, arr.size)
            x_new = np.linspace(0, 1, n_samples)
            arr = np.interp(x_new, x_old, arr)
        self.arr = arr.astype(np.float)
        if repeat_len > 1:
            self.repeat_from = round(2 * repeat_from * ratio)
            self.repeat_len = round(2 * repeat_len * ratio)
        else:
            self.repeat_from = self.repeat_len = 0

def load_samples(mod):
    return [Sample(data.bytes, header.repeat_from, header.repeat_len)
            for data, header in zip(mod.samples, mod.sample_headers)]


def repeat_sample(sample, arr, dur_s):
    '''
    arr is the frequency interpolated sample array.
    '''
    # Duration in nr of samples if repeating
    n_samples = int(SAMPLE_RATE * dur_s)

    # Repeating
    repeat_from = sample.repeat_from
    repeat_len = sample.repeat_len

    ratio = arr.size / sample.arr.size
    repeat_from = int(repeat_from * ratio)
    repeat_len = int(repeat_len * ratio)

    if repeat_len:
        repeat_to = repeat_from + repeat_len
        n_tail = arr.size - repeat_to
        repeat_body_len = n_samples - arr.size + repeat_len
        if repeat_body_len > repeat_len:
            n_repeats = int(repeat_body_len / repeat_len)
            head = arr[:repeat_from]
            tail = arr[repeat_to:]
            body = arr[repeat_from:repeat_to]
            repeated_body = np.tile(body, n_repeats)
            arr = np.concatenate((head, repeated_body, tail))
    return arr
