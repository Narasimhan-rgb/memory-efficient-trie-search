import random
import unittest
from prefix_indexes import CharacterPrefixIndex, TernaryPrefixIndex, LinearPrefixIndex, SortedPrefixIndex, RadixPrefixIndex, DoubleArrayPrefixIndex

class PrefixTests(unittest.TestCase):
    def test_same_contract(self):
        rng=random.Random(2026)
        keys=['','a','ab','abc','abacus','அனா','ana','😀']+[''.join(rng.choices('abc xyz',k=rng.randrange(1,40))) for _ in range(500)]
        indexes=[cls(keys) for cls in (SortedPrefixIndex,RadixPrefixIndex,DoubleArrayPrefixIndex,CharacterPrefixIndex,TernaryPrefixIndex,LinearPrefixIndex)]
        prefixes={key[:n] for key in keys for n in (0,1,2,3,10)}|{'missing','அ','😀','abcd'}
        for prefix in prefixes:
            for k in (0,1,10,1000):
                expected=[s for s in sorted(set(keys)) if s.startswith(prefix)][:k]
                for index in indexes:self.assertEqual(index.search(prefix,k),expected,(type(index),prefix,k))
    def test_empty_and_long(self):
        for cls in (SortedPrefixIndex,RadixPrefixIndex,DoubleArrayPrefixIndex,CharacterPrefixIndex,TernaryPrefixIndex,LinearPrefixIndex):
            self.assertEqual(cls([]).search(''),[])
            self.assertEqual(cls(['a'*2000]).search('a'*1000),['a'*2000])

if __name__=='__main__':unittest.main()
