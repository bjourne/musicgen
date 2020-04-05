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

def main():
    parser = ArgumentParser(description = 'Sample synthesizer and player')
    parser.add_argument('module', type = FileType('rb'))
    parser.add_argument('--sample',
                        required = True,
                        type = int,
                        help = 'Sample index')
    parser.add_argument('--period',
                        required = True,
                        help = 'Sample period')
    args = parser.parse_args()
    with args.module as inf:
        mod = Module.parse(inf.read())
    samples = load_samples(mod)

    header = mod.sample_headers[args.sample - 1]
    name = header.name.decode('utf-8')
    volume = header.volume
    fine_tune = header.fine_tune
    print(f'Sample name: {name}')
    print(f'Volume     : {volume}')
    print(f'Fine tune  : {fine_tune}')

    arr = samples[args.sample - 1].arr
    note_idx = notestr_to_index(args.period)
    freq = FREQS[note_idx]
    arr = interp_freq(arr, freq)

    init_player(SAMPLE_RATE)
    play_sample(arr)

if __name__ == '__main__':
    main()
