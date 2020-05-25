from musicgen.mycode import INSN_JUMP
from musicgen.ssrs import find_min_ssr

def test_find_min_ssr():
    silence_seq = [(INSN_JUMP, 64)] * 38 + [(INSN_JUMP, 32)]
    examples = [
        ('EEEFGAFFGAFFGAFCD', (3, 4, 3)),
        ('ACCCCCCCCCA', (1, 1, 9)),
        ('ABCD', (0, 4, 1)),
        ('BAMBAMBAMBAM', (0, 3, 4)),
        ('AAAAAAAAAAAA', (0, 1, 12)),
        ('ABBBCABBBC', (0, 5, 2)),
        ('ABBBC', (1, 1, 3)),
        (['P2', 'P2', 'P2', 'P0', 'P0', 'P0',
          'P-2', 'P-2', 'P-2', 'P0', 'P0', 'P0'],
         (0, 1, 3)),
        ('B' * 38 + 'A', (0, 1, 38)),
        (silence_seq, (0, 1, 38))
        ]
    for seq, best_ssr in examples:
        min_ssr = find_min_ssr(seq)
        print(seq, min_ssr)
        assert min_ssr == best_ssr
