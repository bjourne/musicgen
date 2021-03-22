# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# Hardcoded for now, will (maybe) fix later.
CODE_GENERATORS = {
    # Pcode abs
    'transformer-pcode-1' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'transformer',

        'batch-size' : 512,
        'sequence-length' : 512,
        'learning-rate' : 0.0004,

        'sampling-method' : ('top-p', 0.98)
    },
    'lstm-pcode-abs-1' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'lstm',

        'batch-size' : 1024,
        'learning-rate' : 0.004,
        'sequence-length' : 256,

        'dropout' : 0.25, 'recurrent-dropout' : 0.25,
        'embedding-size' : 100,
        'lstm1-units' : 512, 'lstm2-units' : 512,

        'sampling-method' : ('top-p', 0.98)
    },
    'gpt2-pcode-abs-1' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'gpt2',

        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 512,
        'sampling-method' : ('top-p', 0.99)
    },
    'orig-pcode' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'original',
        'sampling-method' : ('original', 0)
    },
    'random-pcode' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'random',
        'sampling-method' : ('random', 0)
    },
    # Pcode rel
    'lstm-pcode-rel-1' : {
        'code-type' : 'pcode_rel',
        'network-type' : 'lstm',

        'batch-size' : 256,
        'learning-rate' : 0.004,
        'sequence-length' : 256,

        'dropout' : 0.25, 'recurrent-dropout' : 0.25,
        'embedding-size' : 100,
        'lstm1-units' : 512, 'lstm2-units' : 512,

        'sampling-method' : ('top-p', 0.98)
    },
    'gpt2-pcode-rel-1' : {
        'code-type' : 'pcode_rel',
        'network-type' : 'gpt2',
        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 32,
        'sampling-method' : ('top-p', 0.98)
    },
    # Dcode
    'gpt2-dcode-1' : {
        'code-type' : 'dcode',
        'network-type' : 'gpt2',

        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 512,
        'sampling-method' : ('top-p', 0.98)
    },
    'orig-dcode' : {
        'code-type' : 'dcode',
        'network-type' : 'original',
        'sampling-method' : ('original', 0)
    },
    # Rcode
    'gpt2-rcode-1' : {
        'code-type' : 'rcode',
        'network-type' : 'gpt2',

        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 512,
        'sampling-method' : ('top-p', 0.98)
    },
    # Rcode2
    'gpt2-rcode2-1' : {
        'code-type' : 'rcode2',
        'network-type' : 'gpt2',

        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 512,
        'sampling-method' : ('top-p', 0.98)
    },
}

def get_code_generator(gen_name):
    generator = CODE_GENERATORS.get(gen_name)
    if generator:
        return generator
    names = sorted(name for name in CODE_GENERATORS)
    name_str = ', '.join(names[:-1]) + ', and ' + names[-1]
    fmt = '%s is not a code generator. Specify one of %s'
    raise ValueError(fmt % (gen_name, name_str))

def file_stem(g):
    if g['network-type'] == 'lstm':
        args = (g['code-type'], g['network-type'],
                g['batch-size'], g['learning-rate'],
                g['sequence-length'],
                g['dropout'], g['recurrent-dropout'],
                g['embedding-size'],
                g['lstm1-units'], g['lstm2-units'])
        fmt = '%s-%s-%04d-%.5f-%03d-%.2f-%.2f-%03d-%04d-%04d'
    elif g['network-type'] == 'gpt2':
        args = (g['code-type'], g['network-type'],
                g['batch-size'], g['learning-rate'],
                g['sequence-length'])
        fmt = '%s-%s-%04d-%.5f-%03d'
    elif g['network-type'] == 'transformer':
        args = (g['code-type'], g['network-type'],
                g['batch-size'], g['learning-rate'],
                g['sequence-length'])
        fmt = '%s-%s-%04d-%.5f-%03d'
    else:
        assert False
    return fmt % args

def weights_file(g):
    return 'weights-%s.h5' % file_stem(g)

def log_file(g):
    return 'log-%s.log' % file_stem(g)
