# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.formats.modules import *
from musicgen.formats.modules.parser import Module
from pygame.mixer import (Channel, get_busy, init, pre_init,
                          set_num_channels)
from pygame.sndarray import make_sound
from time import sleep

def wait_for_channels():
    while get_busy() or any(Channel(i).get_queue() for i in range(4)):
        sleep(0.01)

def init_player(sr):
    pre_init(sr, -16, 1)
    init()
    set_num_channels(4)
    for i in range(4):
        Channel(i).stop()
    wait_for_channels()

def play_sample(arr):
    snd = make_sound(arr.astype(np.int16))
    Channel(0).queue(snd)
    wait_for_channels()

def interp_freq(arr, freq):
    x_old = np.linspace(0, 1, arr.size)
    x_new = np.linspace(0, 1, int(arr.size * BASE_FREQ / freq))
    arr_new = np.interp(x_new, x_old, arr)
    fmt = 'Frequency interpolation %d -> %d (%.2f Hz)'
    print(fmt % (arr.size, arr_new.size, freq))
    return arr_new

def play_sample_at_freq(sample, freq):
    arr = sample.arr
    arr = interp_freq(arr, freq)

    # Duration in nr of samples if repeating
    n_samples = int(SAMPLE_RATE * 2.0)

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
    play_sample(arr)


def main():
    parser = ArgumentParser(description = 'Sample synthesizer and player')
    parser.add_argument('module', type = FileType('rb'))
    parser.add_argument('--samples',
                        required = True,
                        help = 'Indices of samples to play')
    parser.add_argument('--period',
                        required = True,
                        help = 'Sample period')
    args = parser.parse_args()
    with args.module as inf:
        mod = Module.parse(inf.read())
    samples = load_samples(mod)

    init_player(SAMPLE_RATE)

    sample_indices = [int(s) - 1 for s in args.samples.split(',')]
    note_idx = notestr_to_index(args.period)
    freq = FREQS[note_idx]

    for sample_idx in sample_indices:
        header = mod.sample_headers[sample_idx]
        name = header.name.decode('utf-8', 'ignore')
        volume = header.volume
        fine_tune = header.fine_tune
        sample = samples[sample_idx]

        print('-------------------------------')
        print(f'Sample name: {name}')
        print(f'Volume     : {volume}')
        print(f'Fine tune  : {fine_tune}')
        print(f'Length     : {len(sample.arr)}')
        print(f'Repeat from: {sample.repeat_from}')
        print(f'Repeat len : {sample.repeat_len}')

        play_sample_at_freq(sample, freq)
    return

    arr = sample.arr
    note_idx = notestr_to_index(args.period)
    freq = FREQS[note_idx]
    arr = interp_freq(arr, freq)

    # Duration in nr of samples if repeating
    n_samples = int(SAMPLE_RATE * 2.0)

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


    play_sample(arr)

if __name__ == '__main__':
    main()
