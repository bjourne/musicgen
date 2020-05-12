from musicgen.utils import OneHotGenerator

def test_one_hot_generator():
    seq = [0] * 200
    gen = OneHotGenerator(seq, 128, 30, 10)
    assert len(gen) == 2
    assert len(gen[0][0]) == 128
    assert len(gen[1][0]) == 42

    seq = [0] * 500
    gen = OneHotGenerator(seq, 128, 64, 10)
    assert len(gen) == 4
    assert len(gen[0][0]) == 128
    assert len(gen[1][0]) == 128
    assert len(gen[2][0]) == 128
    assert len(gen[3][0]) == 500 - 384 - 64

    gen = OneHotGenerator([0] * 100, 64, 64, 10)
    assert len(gen[0][0]) == 36
    assert len(gen) == 1
