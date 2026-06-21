import os
import re
import csv
import json
import sqlite3
import heapq
from array import array
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# --- Config ---
RAW_CSV = Path(os.getenv("RAW_CSV", r"data\raw\data.csv"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", r"data\index"))

MAX_PREFIX = int(os.getenv("MAX_PREFIX", "3"))     # Trie depth
TOPK_STORE = int(os.getenv("TOPK_STORE", "20"))    # Items per node
KEEP_DEDUPE_DB = os.getenv("KEEP_DEDUPE_DB", "false").lower() == "true"

# --- Normalization ---
_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")

def normalize(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = _CLEAN_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s)
    return s

# --- Scoring ---
_INT_RE = re.compile(r"(\d+)")
def popularity_score(reading_stats: str) -> int:
    if not reading_stats: return 0
    nums = [int(x) for x in _INT_RE.findall(reading_stats)]
    want = nums[0] if len(nums) > 0 else 0
    current = nums[1] if len(nums) > 1 else 0
    have = nums[2] if len(nums) > 2 else 0
    return have * 3 + current * 2 + want

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# DSA: Memory Optimized Trie (Inspired by Radix/CoCo Trie)
# ---------------------------------------------------------
class TrieNode:
    # __slots__ saves huge memory by denying __dict__ creation for every node
    # This solves "Standard Trie Problem: High consumption" from your image
    __slots__ = ('children', 'top_items') 

    def __init__(self):
        self.children = {}   # Dictionary mapping char -> TrieNode
        self.top_items = []  # Min-Heap for storing Top-K items

class RadixTrie:
    def __init__(self, limit=20):
        self.root = TrieNode()
        self.limit = limit

    def insert(self, text: str, title_id: int, score: int):
        node = self.root
        # Traverse or Create nodes (Standard Trie construction)
        for char in text:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Update Top-K at this specific node
            item = (score, title_id)
            if len(node.top_items) < self.limit:
                heapq.heappush(node.top_items, item)
            else:
                if item > node.top_items[0]:
                    heapq.heapreplace(node.top_items, item)

    def export_flat_data(self):
        """
        DFS Traversal to flatten the Tree into Arrays.
        Matches 'Coordinate Hash Trie' logic: converting pointers to O(N) arrays.
        """
        results = []
        stack = [("", self.root)]
        
        while stack:
            path, node = stack.pop()
            
            # If this node corresponds to a valid prefix (not root)
            if path:
                # Extract IDs from heap and sort by Score DESC
                sorted_items = sorted(node.top_items, key=lambda x: -x[0])
                ids = [x[1] for x in sorted_items]
                results.append((path, ids))
            
            # Push children to stack
            for char, child in node.children.items():
                stack.append((path + char, child))
        
        # Sort by prefix string for Binary Search compatibility
        results.sort(key=lambda x: x[0])
        return results

# ---------------------------------------------------------
# Main Builder Logic
# ---------------------------------------------------------
def main():
    print("\n=== BookSearch Index Builder (Radix Trie Logic) ===")
    
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {RAW_CSV}")

    ensure_dir(INDEX_DIR)

    # 1. Deduplication using SQLite
    dedupe_db = INDEX_DIR / "dedupe.sqlite"
    if dedupe_db.exists(): dedupe_db.unlink()
    
    db = sqlite3.connect(str(dedupe_db))
    
    # --- FIX IS HERE (Split commands) ---
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA synchronous=OFF;")
    
    db.execute("CREATE TABLE IF NOT EXISTS titles(title TEXT PRIMARY KEY, score INTEGER);")
    
    print("[1/3] Reading CSV and Deduplicating...")
    count = 0
    batch_data = []
    
    with open(RAW_CSV, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = normalize(row.get("title", ""))
            if not t: continue
            s = popularity_score(row.get("reading_stats", ""))
            batch_data.append((t, s))
            
            if len(batch_data) >= 20000:
                db.executemany("""
                    INSERT INTO titles(title, score) VALUES(?, ?)
                    ON CONFLICT(title) DO UPDATE SET
                    score = MAX(score, excluded.score)
                """, batch_data)
                batch_data = []
                count += 20000
                print(f"  - Processed {count} rows...", end="\r")
        
        if batch_data:
            db.executemany("""
                INSERT INTO titles(title, score) VALUES(?, ?)
                ON CONFLICT(title) DO UPDATE SET
                score = MAX(score, excluded.score)
            """, batch_data)
    
    db.commit()
    print(f"\n[1/3] Deduplication Done.")

    # 2. Binary Export & Trie Building
    titles_bin = INDEX_DIR / "titles.bin"
    offsets_bin = INDEX_DIR / "offsets.bin"
    lengths_bin = INDEX_DIR / "lengths.bin"
    scores_bin = INDEX_DIR / "scores.bin"
    
    prefixes_bin = INDEX_DIR / "prefixes.bin"
    prefix_offsets_bin = INDEX_DIR / "prefix_offsets.bin"
    prefix_lengths_bin = INDEX_DIR / "prefix_lengths.bin"
    prefix_id_offsets_bin = INDEX_DIR / "prefix_id_offsets.bin"
    prefix_id_lengths_bin = INDEX_DIR / "prefix_id_lengths.bin"
    prefix_ids_bin = INDEX_DIR / "prefix_ids.bin"
    
    meta_json = INDEX_DIR / "meta.json"

    offsets = array("Q")
    lengths = array("I")
    scores = array("I")
    
    # Initialize our Custom Trie
    trie = RadixTrie(limit=TOPK_STORE)
    
    print("[2/3] Building Trie & Writing Title Data...")
    cur = db.cursor()
    cur.execute("SELECT title, score FROM titles ORDER BY title;")
    
    pos = 0
    total_titles = 0
    
    with open(titles_bin, "wb") as ft:
        for title_id, (title, score) in enumerate(cur):
            b = title.encode("utf-8")
            offsets.append(pos)
            lengths.append(len(b))
            scores.append(int(score))
            ft.write(b)
            pos += len(b)
            total_titles += 1
            
            # --- Insert into Trie ---
            clean_title = title 
            limit_len = min(len(clean_title), MAX_PREFIX)
            prefix_part = clean_title[:limit_len]
            
            trie.insert(prefix_part, title_id, int(score))

            if title_id % 100000 == 0:
                print(f"  - Indexed {title_id} titles...")

    with open(offsets_bin, "wb") as f: offsets.tofile(f)
    with open(lengths_bin, "wb") as f: lengths.tofile(f)
    with open(scores_bin, "wb") as f: scores.tofile(f)

    # 3. Export Trie to Flat Binary Arrays
    print("[3/3] Exporting Trie to Binary Index...")
    
    flat_data = trie.export_flat_data()
    
    p_offsets = array("I")
    p_lengths = array("H")
    id_offsets = array("I")
    id_lengths = array("H")
    all_ids = array("I")
    
    ppos = 0
    with open(prefixes_bin, "wb") as fp:
        for pref, ids in flat_data:
            pb = pref.encode("utf-8")
            p_offsets.append(ppos)
            p_lengths.append(len(pb))
            fp.write(pb)
            ppos += len(pb)
            
            id_offsets.append(len(all_ids))
            id_lengths.append(len(ids))
            all_ids.extend(ids)
            
    with open(prefix_offsets_bin, "wb") as f: p_offsets.tofile(f)
    with open(prefix_lengths_bin, "wb") as f: p_lengths.tofile(f)
    with open(prefix_id_offsets_bin, "wb") as f: id_offsets.tofile(f)
    with open(prefix_id_lengths_bin, "wb") as f: id_lengths.tofile(f)
    with open(prefix_ids_bin, "wb") as f: all_ids.tofile(f)

    meta = {
        "total_titles": total_titles,
        "max_prefix": MAX_PREFIX,
        "structure": "RadixTrie-Flattened"
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    
    db.close()
    if not KEEP_DEDUPE_DB:
        dedupe_db.unlink()
        
    print("\n[SUCCESS] Index Built Successfully using Trie Strategy.")
    print("Run 'python main.py' to view the Graphs.")

if __name__ == "__main__":
    main()