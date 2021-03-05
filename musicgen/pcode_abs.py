from musicgen import code_utils, pcode

def to_code(notes, percussion):
    return pcode.to_code(notes, False, percussion)

def to_notes(code, row_time):
    return pcode.to_notes(code, False, row_time)

def pause():
    return pcode.pause()

def is_transposable():
    return True

def code_transpositions(code):
    return code_utils.code_transpositions(code)

def normalize_pitches(code):
    return code_utils.normalize_pitches(code)

def estimate_row_time(code):
    return pcode.estimate_row_time(code, False)
