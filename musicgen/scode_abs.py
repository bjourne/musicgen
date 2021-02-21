from musicgen import code_utils, scode

def to_code(mod, percussion, min_pitch):
    return scode.to_code(mod, False, percussion, min_pitch)

def metadata(code):
    return scode.metadata(code)

def is_transposable():
    return True

def transpose_code(code):
    return code_utils.transpose_code(code)
