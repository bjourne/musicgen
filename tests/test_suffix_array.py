from musicgen.suffix_array import lcp_array, suffix_array

def test_suffix_array():
    examples = [
        ('ABAC', [0, 2, 1, 3]),
        ('banana', [5, 3, 1, 0, 4, 2]),
        ('abaab', [2, 3, 0, 4, 1]),
        ('ABABBAB', [5, 0, 2, 6, 4, 1, 3])]

    for seq, sa in examples:
        assert suffix_array(seq) == sa

def test_lcp_array():
    examples = [
        ('BANANA', [0, 1, 3, 0, 0, 2]),
        ('ABABBAB', [0, 2, 2, 0, 1, 3, 1])
        ]
    for seq, lcp in examples:
        sa = suffix_array(seq)
        assert lcp_array(seq, sa)[1] == lcp
