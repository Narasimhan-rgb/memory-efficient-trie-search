import os
import mmap
import json
import matplotlib.pyplot as plt
from array import array
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path(os.getenv("INDEX_DIR", r"data\index"))

# --- Globals for Loaded Data ---
READY = False
LENGTHS = array("I")
SCORES = array("I")
PFX_LENGTHS = array("H")
PFX_ID_LENGTHS = array("H")

def file_item_count(path: Path, item_size: int) -> int:
    return path.stat().st_size // item_size

def load_index():
    global READY, LENGTHS, SCORES, PFX_LENGTHS, PFX_ID_LENGTHS
    
    lengths_bin = INDEX_DIR / "lengths.bin"
    scores_bin  = INDEX_DIR / "scores.bin"
    prefix_lengths_bin = INDEX_DIR / "prefix_lengths.bin"
    prefix_id_lengths_bin = INDEX_DIR / "prefix_id_lengths.bin"
    
    needed = [lengths_bin, scores_bin, prefix_lengths_bin, prefix_id_lengths_bin]
    if not all(p.exists() for p in needed):
        print(f"Error: Index files missing in {INDEX_DIR}")
        print("Please run build_index.py first.")
        return

    print("Loading Index for Visualization...")
    
    LENGTHS = array("I")
    with open(lengths_bin, "rb") as f:
        LENGTHS.fromfile(f, file_item_count(lengths_bin, 4))
        
    SCORES = array("I")
    with open(scores_bin, "rb") as f:
        SCORES.fromfile(f, file_item_count(scores_bin, 4))
        
    PFX_LENGTHS = array("H")
    with open(prefix_lengths_bin, "rb") as f:
        PFX_LENGTHS.fromfile(f, file_item_count(prefix_lengths_bin, 2))
        
    PFX_ID_LENGTHS = array("H")
    with open(prefix_id_lengths_bin, "rb") as f:
        PFX_ID_LENGTHS.fromfile(f, file_item_count(prefix_id_lengths_bin, 2))

    READY = True
    print("Index Loaded Successfully.")

def show_graphs():
    if not READY:
        return

    print("\n=== Generating Trie Statistics ===")
    
    # 1. Helper for Histograms
    def hist_counts(values, bins=40):
        if not values: return [], []
        vmin, vmax = int(min(values)), int(max(values))
        if vmin == vmax: return [vmin], [len(values)]
        step = (vmax - vmin) / bins
        counts = [0]*bins
        for v in values:
            idx = int((int(v) - vmin) / step)
            if idx >= bins: idx = bins - 1
            counts[idx] += 1
        centers = [vmin + (i + 0.5)*step for i in range(bins)]
        return centers, counts

    # Data 1: Title Lengths
    x1, y1 = hist_counts(LENGTHS, bins=30)
    
    # Data 2: Popularity Scores
    x2, y2 = hist_counts(SCORES, bins=30)
    
    # Data 3: Trie Depth Analysis (Prefix Counts)
    # Finding Max Prefix used
    max_pfx = 0
    if len(PFX_LENGTHS) > 0:
        max_pfx = max(PFX_LENGTHS)
        
    pfx_counts = [0] * (max_pfx + 1)
    for ln in PFX_LENGTHS:
        pfx_counts[ln] += 1
        
    # Data 4: Suggestions per Node (How efficient is our Trie?)
    suggestion_density = [0] * (max_pfx + 1)
    suggestion_counts = [0] * (max_pfx + 1)
    
    for i in range(len(PFX_ID_LENGTHS)):
        p_len = PFX_LENGTHS[i]
        n_ids = PFX_ID_LENGTHS[i]
        suggestion_density[p_len] += n_ids
        suggestion_counts[p_len] += 1
        
    avg_suggestions = []
    depths = range(1, max_pfx + 1)
    for d in depths:
        if suggestion_counts[d] > 0:
            avg_suggestions.append(suggestion_density[d] / suggestion_counts[d])
        else:
            avg_suggestions.append(0)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Radix Trie Index Analytics", fontsize=16)
    
    # Plot 1: Title Length
    ax = axes[0][0]
    ax.bar(x1, y1, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Book Title Lengths")
    ax.set_xlabel("Length (bytes)")
    ax.set_ylabel("Frequency")
    
    # Plot 2: Scores
    ax = axes[0][1]
    ax.bar(x2, y2, color='salmon', edgecolor='black')
    ax.set_title("Distribution of Popularity Scores")
    ax.set_xlabel("Score")
    
    # Plot 3: Trie Nodes per Depth
    ax = axes[1][0]
    ax.bar(depths, [pfx_counts[d] for d in depths], color='lightgreen', edgecolor='black')
    ax.set_title("Trie Nodes count by Prefix Length")
    ax.set_xlabel("Prefix Length (Depth)")
    ax.set_ylabel("Node Count")
    
    # Plot 4: Avg Suggestions
    ax = axes[1][1]
    ax.plot(depths, avg_suggestions, marker='o', linestyle='-', color='purple')
    ax.set_title("Avg Suggestions Stored per Trie Node")
    ax.set_xlabel("Prefix Length")
    ax.set_ylabel("Avg IDs Count")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    load_index()
    show_graphs()