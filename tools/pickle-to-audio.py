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
    --n-prompt=<i>         number of tokens in the prompt
    --format=<fmt>         output format [default: mid]
'''
from docopt import docopt
from musicgen.code_generators import get_code_generator
from musicgen.code_utils import CODE_MIDI_MAPPING
from musicgen.generation import notes_to_audio_file
from musicgen.utils import SP, load_pickle
from musicgen.training_data import CODE_MODULES
from pathlib import Path

def main():
    # Prologue
    args = docopt(__doc__, version = 'Pickle to audio 1.0')
    SP.enabled = args['--verbose']
    files = args['<files>']
    file_paths = [Path(f) for f in files]

    # Prompt is used to estimate tempo
    n_prompt = int(args['--n-prompt'])
    assert n_prompt % 2 == 0

    format = args['--format']
    for file_path in file_paths:
        code = load_pickle(file_path)
        code_type = file_path.name.split('-')[2]
        code_mod = CODE_MODULES[code_type]

        # Halve the code length to use for tempo estimation in case
        # dcode is used since its encoding is twice as dense.
        if code_type == 'dcode':
            n_estimate = n_prompt // 2
        else:
            n_estimate = n_prompt
        row_time = code_mod.estimate_row_time(code[:n_estimate])
        notes = code_mod.to_notes(code, row_time)

        prefix = '.'.join(str(file_path).split('.')[:-2])
        output_name = '%s.%s' % (prefix, format)
        output_path = file_path.parent / output_name

        SP.print('Creating %s.' %  output_path)
        stereo = (format == 'mp3')
        notes_to_audio_file(notes, output_path, CODE_MIDI_MAPPING, stereo)

if __name__ == '__main__':
    main()
