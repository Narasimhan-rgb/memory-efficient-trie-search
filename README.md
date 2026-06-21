# Memory-Efficient Trie Search

A Python research prototype for evaluating **prefix search**, **substring search**, and **fuzzy matching** on book-title data.

This repository supports the implementation and benchmarking work behind the paper:

> **Memory-Efficient Trie Variants for Large-Scale Text Search: A Comparative Study with Classical Search Algorithms**

The project focuses on practical search-system ideas: memory-conscious Trie nodes, Top-K suggestions, SQLite-backed deduplication, flattened binary-index export, and comparative benchmarking.

![Search benchmark](benchmark_figure.png)

## What this project contains

- **Prefix Trie engine** for autocomplete-style lookup
- **3-Gram inverted index** for substring retrieval
- **Fuzzy matching** using RapidFuzz / Levenshtein-style scoring
- **SQLite deduplication** of repeated titles using a popularity score
- **Top-K suggestion storage** at Trie nodes with a min-heap
- **Binary export** of title and prefix metadata for compact index storage
- **Benchmark visualizations** for latency, returned suggestions, build time, and memory

## Repository files

```text
.
├── build_index.py          # Builds the deduplicated Trie-backed binary index
├── benchmark_engines.py    # Compares Trie, 3-Gram, and Fuzzy engines
├── main.py                 # Generates analytics for the built index
├── benchmark_figure.png    # Example benchmark output
├── requirements.txt        # Python dependencies
└── .gitignore              # Excludes local data, generated indexes, and .env
```

## Requirements

- Python 3.10 or newer
- A CSV file containing at least a `title` column

Install the dependencies:

```bash
python -m venv .venv
```

**Windows CMD**

```cmd
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset setup

The raw dataset is intentionally not included in this repository. It can be large and may be subject to source-specific licensing or redistribution conditions.

Create this local structure:

```text
data/
└── raw/
    └── data.csv
```

The scripts use this default path:

```text
data/raw/data.csv
```

The CSV should include:

```csv
title,reading_stats
Harry Potter and the Philosopher's Stone,"100 want, 50 currently reading, 500 have read"
```

- `title` is required.
- `reading_stats` is optional and is used by `build_index.py` to calculate a simple popularity score.

You can override paths with a local `.env` file. Do not commit `.env`.

```env
RAW_CSV=data/raw/data.csv
INDEX_DIR=data/index
MAX_PREFIX=4
TOPK_STORE=50
KEEP_DEDUPE_DB=false
```

## Run the project

### 1. Build the compact search index

```bash
python build_index.py
```

This step:

1. Reads and normalizes titles.
2. Deduplicates titles with SQLite.
3. Stores Top-K title identifiers for indexed prefixes.
4. Writes compact binary files to `data/index/`.

### 2. Run the engine benchmark

```bash
python benchmark_engines.py
```

The benchmark compares:

| Engine | Best use case |
|---|---|
| Trie | Prefix autocomplete, such as `harry pot` |
| 3-Gram | Substring-style matching, such as `stone` |
| Fuzzy | Typo-tolerant matching, such as `arry potter` |

The script writes `benchmark_figure.png` and displays the plots.

### 3. Generate Trie-index analytics

```bash
python main.py
```

This visualizes title lengths, popularity-score distribution, indexed prefix depth, and average suggestions per prefix node.

## Implementation notes

### Memory-conscious Trie nodes

`TrieNode` uses `__slots__` to avoid creating a per-instance `__dict__`, reducing Python object overhead for large numbers of nodes.

### Top-K suggestions

Each indexed node keeps a bounded min-heap of high-scoring title IDs. This avoids a full tree traversal for every autocomplete query.

### Flattened binary index

The builder exports offsets, lengths, prefix metadata, and title IDs into typed binary arrays. This keeps generated index data separate from the source code and makes future memory-mapped lookup experiments possible.

## Important accuracy note

The `RadixTrie` class name reflects the project direction, but the current implementation is a **memory-optimized character-level prefix Trie with limited prefix indexing**. It does **not yet implement full edge-label path compression** used by a classical Radix Trie / PATRICIA Trie.

Future work includes:

- true path compression for Radix/PATRICIA nodes
- Double-Array Trie implementation
- scalable incremental updates
- FastAPI search endpoints
- controlled benchmarks on larger datasets and repeatable hardware settings
- semantic reranking with vector search

## Reproducibility and safety

The repository intentionally excludes:

- `.env` files
- raw datasets
- generated binary index files
- SQLite databases
- Python cache files

Never upload API keys, passwords, tokens, recovery codes, or private datasets.

## Research context

The accompanying research work studies memory-efficient Trie variants for large-scale text search and compares classic search methods with Trie-based approaches. The current code repository is a reproducible implementation and benchmark prototype for that work.

## Author

**Narasimhan D**  
Computer Science Student, VIT Chennai  
Research interests: algorithms, information retrieval, intelligent systems, and healthcare AI.
