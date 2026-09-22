# Repository status — 22 September 2026

Prefix-search research prototype, independent of AAQ sorting. New prefix_indexes.py implements static compressed Radix and typed-array Double-Array indexes alongside character Trie, Ternary Search Tree, linear and binary-search baselines. All six use deduplicated Unicode keys and lexicographic top-k ranking. The older popularity-ranked builder remains separate. Run `python -m unittest discover -s tests -v`, then `python benchmark_prefix.py data/raw/data.csv --output prefix-results.json`. Build peak includes temporary structures; traced Python allocations are not process RSS. Large-data measurements remain outstanding.

## Checks executed in this pass

Two parameterized unittest methods passed across all six indexes using hundreds of generated keys, Unicode, empty inputs and long prefixes. The benchmark CLI ran on a five-title smoke fixture only; no large-dataset performance claim is made.
