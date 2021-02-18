# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
'''
Pickle to audio
===============
Helper program to convert generated sequences to audio files.

Usage:
    pickle-to-audio.py [options] <files>...

Options:
    -h --help              show this screen
    -v --verbose           print more output
'''
from docopt import docopt
from musicgen.code_utils import CODE_MIDI_MAPPING
from musicgen.generation import notes_to_audio_file
from musicgen.pcode import pcode_to_notes
from musicgen.utils import SP, load_pickle
from pathlib import Path

def main():
    # Prologue
    args = docopt(__doc__, version = 'Pickle to audio 1.0')
    SP.enabled = args['--verbose']
    files = args['<files>']
    file_paths = [Path(f) for f in files]

    for file_path in file_paths:
        code_type = file_path.name.split('-')[0]
        obj = load_pickle(file_path)
        if code_type == 'pcode_abs':
            notes = pcode_to_notes(obj, False)

        dir = file_path.parent
        stem = file_path.stem
        output_file = dir / (stem + '.mid')
        SP.print('Creating %s.' %  output_file)
        notes_to_audio_file(notes, output_file, CODE_MIDI_MAPPING, False)

if __name__ == '__main__':
    main()
