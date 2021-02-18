from musicgen import pcode

def to_code(mod):
    return pcode.to_code(mod, True)

def to_notes(code):
    return pcode.to_notes(code, True)

def pause():
    return pcode.pause()

def metadata(code):
    return pcode.metadata(code)

def is_transposable():
    return False
