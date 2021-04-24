# Copyright (C) 2020-2021 Björn Lindqvist <bjourne@gmail.com>
"""
Music generation
================

Usage:
    model-generator.py [options] <root-path> <generator>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --offset=<s>           prompt offset [default: random]
    --add-pause            insert a pause between the prompt and
                           generated code
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
from musicgen import rcode
from musicgen.code_generators import get_code_generator, weights_file
from musicgen.code_utils import (INSN_SILENCE, normalize_pitches)
from musicgen.tensorflow import load_generating_model, generate_sequences
from musicgen.training_data import (CODE_MODULES,
                                    load_training_data,
                                    random_rel_ofs,
                                    save_generated_sequences)
from musicgen.utils import SP
from pathlib import Path

import numpy as np

def load_fragment(root_path, code_type, n_prompt, n_generate, ofs):
    # Works by loading pcode_abs and coverting to the desired format.
    _, _, td = load_training_data('pcode_abs', root_path)

    n_frag = n_prompt + n_generate

    # Pick a song fragment
    if ofs == 'random':
        ofs = random_rel_ofs(td, n_frag)
    else:
        ofs = tuple([int(s) for s in ofs.split('-')])

    s_i, ss_i, t_i, o = ofs
    name = td.songs[s_i][0]
    song = td.songs[s_i][1][ss_i][t_i]
    assert o + n_frag <= len(song)
    frag = song[o:o + n_frag]

    SP.print('Selected %s:%d of song %s.' % (ofs, len(frag), name))

    code = td.encoder.decode_chars(frag)
    code = normalize_pitches(code)

    # Split it into prompt and remainder.
    prompt, orig = code[:n_prompt], code[n_prompt:]
    if code_type == 'rcode2':
        # This is tricky... both the rcoded length of prompt and orig
        # needs to be divisble by 2.
        prompt = list(rcode.from_pcode(prompt))
        orig = list(rcode.from_pcode(orig))
        if len(prompt) % 2 == 1:
            # Steal one token from orig
            prompt.append(orig[0])
            orig = orig[1:]
        if len(orig) % 2 == 1:
            # Pad
            orig.append((INSN_SILENCE, 1))

    # Convert it back to the native format
    _, _, td = load_training_data(code_type, root_path)
    code_mod = CODE_MODULES[code_type]
    prompt = list(code_mod.from_pcode(prompt))
    orig = list(code_mod.from_pcode(orig))

    prompt = td.encoder.encode_chars(prompt, False)
    orig = td.encoder.encode_chars(orig, False)

    SP.print('%d prompt and %d orig tokens.' % (len(prompt), len(orig)))

    return td, ofs, prompt, orig

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    root_path = Path(args['<root-path>'])

    # Kind of code
    g = get_code_generator(args['<generator>'])

    td, ofs, prompt, orig = load_fragment(
        root_path, g['code-type'], 64, 800, args['--offset'])
    vocab_size = len(td.encoder.ix2ch)

    # Good settings for GPT-2
    temps = [0.90, 0.95, 1.0, 1.01, 1.02]
    top_ps = [0.92, 0.95, 0.98, 0.99, 0.999]

    # Settings for GPT-2 with unlikelihood
    temps = [0.7, 0.8, 0.9, 0.95, 1.0]
    top_ps = [0.7, 0.8, 0.85, 0.9, 0.92]

    skews = [('temperature', t) for t in temps] \
        + [('top-p', p) for p in top_ps]

    n_clips = len(skews)

    # Load the model and the weights
    model = load_generating_model(g, vocab_size, n_clips)
    weights_path = root_path / 'weights' / weights_file(g)
    assert weights_path.exists()
    model.load_weights(str(weights_path))
    model.reset_states()
    model.summary()

    prompt = np.repeat(np.expand_dims(prompt, 0), n_clips, axis = 0)

    # Token(s) to avoid
    seqs, log_probs = generate_sequences(g, model, prompt, len(orig),
                                         [], skews)

    # Add the original
    seqs = np.vstack((seqs, orig))
    prompt = np.vstack((prompt, prompt[0]))
    log_probs = np.append(log_probs, 0)
    skews.append(('original', 0))

    # Maybe add a pause
    if args['--add-pause']:
        pause = td.encoder.encode_chars(td.pause_code(), False)
        join = np.repeat(np.expand_dims(pause, 0), len(seqs), axis = 0)
        seqs = np.hstack((prompt, join, seqs))
    else:
        seqs = np.hstack((prompt, seqs))

    # Same offsets for all generated files
    offsets = [ofs] * len(skews)

    # Save generated code
    output_path = root_path / 'generated'
    output_path.mkdir(exist_ok = True)

    save_generated_sequences(g, output_path, td,
                             seqs, offsets, log_probs, skews)

if __name__ == '__main__':
    main()
