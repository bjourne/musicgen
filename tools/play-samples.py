# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser, FileType
from musicgen.defs import BASE_FREQ, FREQS, SAMPLE_RATE
from musicgen.parser import load_file
from musicgen.prettyprint import notestr_to_idx
from musicgen.samples import load_samples, repeat_sample
from pygame.mixer import (Channel, get_busy, init, pre_init,
                          set_num_channels)
from pygame.sndarray import make_sound
from time import sleep
import numpy as np

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
    sleep(2)

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

def play_sample_at_freq(sample, freq, volume):
    arr = sample.arr
    arr = interp_freq(arr, freq)

    # Repeating
    arr = repeat_sample(sample, arr, 1.0)

    vol_frac = volume / 64
    arr *= vol_frac

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
    parser.add_argument('--volume',
                        type = int)
    args = parser.parse_args()
    args.module.close()

    mod = load_file(args.module.name)
    samples = load_samples(mod)

    init_player(SAMPLE_RATE)

    sample_indices = [int(s) - 1 for s in args.samples.split(',')]
    note_idx = notestr_to_idx(args.period)
    freq = FREQS[note_idx]
    for sample_idx in sample_indices:
        header = mod.sample_headers[sample_idx]
        name = header.name
        volume = args.volume
        if not args.volume:
            volume = header.volume
        fine_tune = header.fine_tune
        sample = samples[sample_idx]

        length = header.size * 2
        repeat_from = header.repeat_from
        repeat_len = 0 if header.repeat_len < 2 else header.repeat_len

        print(f'*** Sample "{name}" (#{sample_idx + 1}) ***')
        print(f'Volume     : {volume}')
        print(f'Fine tune  : {fine_tune}')
        print(f'Length     : {length}')
        print(f'Repeat from: {repeat_from}')
        print(f'Repeat len : {repeat_len}')

        play_sample_at_freq(sample, freq, volume)
        sleep(0.5)

if __name__ == '__main__':
    main()
