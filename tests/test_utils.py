# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import CharEncoder

def test_format():
    enc = CharEncoder()
    seq = enc.encode_chars('blah', True)
    assert len(seq) == 4
    assert type(seq) == list
