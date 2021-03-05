# Hardcoded for now, will fix later
CODE_GENERATORS = {
    'transformer-pcode-1' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'transformer',

        'batch-size' : 512,
        'sequence-length' : 512,
        'learning-rate' : 0.0004,

        'sampling-method' : ('top-p', 0.98)
    },
    'lstm-pcode-1' : {
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
    'gpt2-pcode-1' : {
        'code-type' : 'pcode_abs',
        'network-type' : 'gpt2',

        'batch-size' : 64,
        'learning-rate' : 0.00001,
        'sequence-length' : 512,
        'sampling-method' : ('top-p', 0.98)
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
    }
}

def get_code_generator(gen_name):
    generator = CODE_GENERATORS.get(gen_name)
    if not generator:
        names = ', '.join(name for name in CODE_GENERATORS)
        fmt = '%s is not a code generator. Specify one of %s'
        raise ValueError(fmt % (gen_name, names))
    return generator
