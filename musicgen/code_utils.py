# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Utils common for both code formats.
from musicgen.analyze import sample_props

def guess_percussive_instruments(mod, notes):
    props = sample_props(mod, notes)
    props = [(s, p.n_notes, p.is_percussive) for (s, p) in props.items()
             if p.is_percussive]

    # Sort by the number of notes so that the same instrument
    # assignment is generated every time.
    props = list(reversed(sorted(props, key = lambda x: x[1])))
    percussive_samples = [s for (s, _, _) in props]

    return {s : i % 3 for i, s in enumerate(percussive_samples)}
