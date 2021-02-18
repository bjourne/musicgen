from musicgen import scode

def to_code(mod):
    return scode.to_code(mod, True, True)

def metadata(code):
    return scode.metadata(code)

def is_transposable():
    return False
