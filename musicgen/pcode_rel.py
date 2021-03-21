from musicgen import pcode

def to_code(mod, percussion):
    return pcode.to_code(mod, True, percussion)

def to_notes(code, row_time):
    return pcode.to_notes(code, True, row_time)

def pause():
    return pcode.pause()

def is_transposable():
    return False

def estimate_row_time(code):
    return pcode.estimate_row_time(code, True)

def normalize_pitches(code):
    return code
