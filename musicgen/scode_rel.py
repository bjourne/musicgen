from musicgen import scode

def to_code(mod, percussion, min_pitch):
    return scode.to_code(mod, True, percussion, min_pitch)

def metadata(code):
    return scode.metadata(code)

def is_transposable():
    return False
