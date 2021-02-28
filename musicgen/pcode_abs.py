from musicgen import code_utils, pcode

def to_code(notes, percussion):
    return pcode.to_code(notes, False, percussion)

def to_notes(code):
    return pcode.to_notes(code, False)

def pause():
    return pcode.pause()

def is_transposable():
    return True

def code_transpositions(code):
    return code_utils.code_transpositions(code)

def normalize_pitches(code):
    return code_utils.normalize_pitches(code)
