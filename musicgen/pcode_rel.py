from musicgen import pcode

def to_code(mod, percussion):
    return pcode.to_code(mod, True, percussion)

def to_notes(code):
    return pcode.to_notes(code, True)

def pause():
    return pcode.pause()

def is_transposable():
    return False
