from musicgen import code_utils, pcode

def to_code(mod, percussion, min_pitch):
    return pcode.to_code(mod, False, percussion, min_pitch)

def to_notes(code):
    return pcode.to_notes(code, False)

def pause():
    return pcode.pause()

def metadata(code):
    return pcode.metadata(code)

def is_transposable():
    return True

def transpose_code(code):
    return code_utils.transpose_code(code)
