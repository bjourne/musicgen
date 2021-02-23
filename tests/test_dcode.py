from musicgen import dcode, pcode

def test_pause():
    tot1 = sum([a for (c, a) in pcode.pause()])
    tot2 = sum([a1+a2 for (c, (a1, a2)) in dcode.pause()])
    assert tot1 == tot2


    # ppause2 = dcode.pause()
