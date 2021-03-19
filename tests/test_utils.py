# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import CharEncoder

def test_format():
    enc = CharEncoder()
    seq = enc.encode_chars('blah', True)
    assert len(seq) == 4
    assert type(seq) == list

def test_missing_item():
    enc = CharEncoder()
    try:
        enc.encode_char((1, 2), False)
    except ValueError as e:
        pass
