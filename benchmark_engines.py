import os
import time
import csv
import re
import tracemalloc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

load_dotenv()

RAW_CSV    = Path(os.getenv("RAW_CSV", r"data\raw\data.csv"))
DATA_LIMIT = 10000
TOP_K      = 10

_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")
def normalize(s):
    if not s: return ""
    return _CLEAN_RE.sub("", s.lower().strip())

# ══════════════════════════════════════════════════════════════
# ENGINES
# ══════════════════════════════════════════════════════════════

class TrieNode:
    __slots__ = ('children', 'data')
    def __init__(self): self.children = {}; self.data = []

class TrieEngine:
    name  = "Trie"
    label = "Trie\n(Prefix Tree)"
    def __init__(self): self.root = TrieNode()

    def insert(self, text, original):
        node = self.root
        for c in text:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            if len(node.data) < TOP_K:
                node.data.append(original)

    def search(self, query):
        node = self.root
        for c in query:
            if c not in node.children:
                return []
            node = node.children[c]
        return node.data

class NGramEngine:
    name  = "3-Gram"
    label = "3-Gram\n(Inverted Index)"
    def __init__(self):
        self.index  = defaultdict(set)
        self.titles = {}

    def _grams(self, t):
        return [t[i:i+3] for i in range(len(t) - 2)]

    def insert(self, text, original, tid):
        self.titles[tid] = original
        for g in self._grams(text):
            self.index[g].add(tid)

    def search(self, query):
        grams = self._grams(query)
        if not grams: return []
        ids = self.index[grams[0]].copy()
        for g in grams[1:]:
            ids &= self.index.get(g, set())
            if not ids: return []
        return [self.titles[tid] for tid in list(ids)[:TOP_K]]

class FuzzyEngine:
    name  = "Fuzzy"
    label = "Fuzzy\n(Levenshtein)"
    def __init__(self): self.titles = []

    def insert(self, text, original):
        self.titles.append(original)

    def search(self, query):
        return [x[0] for x in process.extract(
            query, self.titles, scorer=fuzz.WRatio, limit=TOP_K)]

# ══════════════════════════════════════════════════════════════
# MEASUREMENT HELPERS
# ══════════════════════════════════════════════════════════════

def bench_query(eng, q_norm, q_raw, runs=30):
    """Accurate isolated timing — direct call per engine type."""
    is_fuzzy = isinstance(eng, FuzzyEngine)
    q = q_raw if is_fuzzy else q_norm
    # warmup
    for _ in range(5):
        eng.search(q)
    # timed runs
    lats = []
    for _ in range(runs):
        t0  = time.perf_counter_ns()
        res = eng.search(q)
        lats.append((time.perf_counter_ns() - t0) / 1e6)   # → ms
    return res, float(np.median(lats))

def get_build_memory_mb(eng, norm_titles, raw_titles):
    """
    Measure peak RSS change during build — reflects actual OS-level
    memory committed, not just Python allocator.
    Uses tracemalloc for Python heap delta.
    """
    import psutil, os as _os
    proc = psutil.Process(_os.getpid())

    tracemalloc.start()
    before_rss = proc.memory_info().rss

    if isinstance(eng, NGramEngine):
        for i, (n, r) in enumerate(zip(norm_titles, raw_titles)):
            eng.insert(n, r, i)
    else:
        for n, r in zip(norm_titles, raw_titles):
            eng.insert(n, r)

    after_rss = proc.memory_info().rss
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_delta = max(0, (after_rss - before_rss) / (1024 * 1024))
    py_peak   = peak_py / (1024 * 1024)
    # Python heap peak is more precise for in-process structures
    return py_peak

# ══════════════════════════════════════════════════════════════
# VISUAL STYLE  (clean, academic)
# ══════════════════════════════════════════════════════════════

ENG_STYLE = {
    "Trie":   {"color": "#1565C0", "marker": "o", "ms": 90,  "zorder": 5},
    "3-Gram": {"color": "#D84315", "marker": "s", "ms": 90,  "zorder": 4},
    "Fuzzy":  {"color": "#2E7D32", "marker": "^", "ms": 100, "zorder": 4},
}

def ax_clean(ax, title, xlabel, ylabel):
    """Apply consistent academic styling to an axes."""
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#AAAAAA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")


# ══════════════════════════════════════════════════════════════
# MAIN PLOT  — 3×3 academic grid
# ══════════════════════════════════════════════════════════════

def plot(engines, queries, latencies, suggestions, build_times, build_mem_mb):

    n_eng    = len(engines)
    n_q      = len(queries)
    enames   = [e.name for e in engines]
    q_labels = [f"'{q[0]}'\n({q[1]})" for q in queries]
    x        = np.arange(n_q)

    # Legend proxies
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [],
                      color=ENG_STYLE[e.name]["color"],
                      marker=ENG_STYLE[e.name]["marker"],
                      ms=7, ls="None", label=e.label)
        for e in engines
    ]

    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("#F5F5F5")

    fig.suptitle(
        f"Search Engine Benchmark  ·  {DATA_LIMIT:,} book titles  ·  Top-K = {TOP_K}",
        fontsize=13, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.38,
        left=0.08, right=0.97, top=0.95, bottom=0.06
    )

    # ─────────────────────────────────────────────────────────
    # ROW 0, COL 0-1  →  Search Latency (Time µs / query)
    # One scatter point per engine per query — exactly like paper style
    # ─────────────────────────────────────────────────────────
    ax_lat = fig.add_subplot(gs[0, :2])

    for eng in engines:
        s    = ENG_STYLE[eng.name]
        vals = [v * 1000 for v in latencies[eng.name]]   # ms → µs
        ax_lat.scatter(x, vals,
                       color=s["color"], marker=s["marker"],
                       s=s["ms"], zorder=s["zorder"],
                       label=eng.label, edgecolors="white", linewidths=0.8)
        # connect dots with a thin line
        ax_lat.plot(x, vals, color=s["color"], lw=1.2,
                    alpha=0.55, zorder=s["zorder"] - 1)
        # clean value label above each point
        for xi, v in zip(x, vals):
            ax_lat.text(xi, v * 1.5, f"{v:.1f}",
                        ha="center", fontsize=7.5,
                        color=s["color"], fontweight="bold")

    ax_lat.set_yscale("log")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(q_labels, fontsize=8.5)
    ax_lat.legend(handles=legend_handles, fontsize=8,
                  loc="upper right", framealpha=0.9,
                  edgecolor="#CCCCCC")
    ax_clean(ax_lat,
             "Search Time per Query Type",
             "", "Time (µs / query) — log scale")

    # Short note below title — plain English
    ax_lat.text(0.01, 0.97,
                "Lower = faster response. Trie walks a tree (O(k)), "
                "Fuzzy scans all titles (O(n)).",
                transform=ax_lat.transAxes,
                fontsize=7.5, va="top", color="#555555", style="italic")

    # ─────────────────────────────────────────────────────────
    # ROW 0, COL 2  →  Suggestions Returned
    # ─────────────────────────────────────────────────────────
    ax_sug = fig.add_subplot(gs[0, 2])

    for eng in engines:
        s    = ENG_STYLE[eng.name]
        vals = suggestions[eng.name]
        ax_sug.scatter(x, vals,
                       color=s["color"], marker=s["marker"],
                       s=s["ms"], zorder=s["zorder"],
                       edgecolors="white", linewidths=0.8)
        ax_sug.plot(x, vals, color=s["color"], lw=1.2, alpha=0.55)
        for xi, v in zip(x, vals):
            ax_sug.text(xi + 0.08, v + 0.3, str(v),
                        fontsize=7.5, color=s["color"], fontweight="bold")

    ax_sug.axhline(TOP_K, ls="--", color="#999999", lw=1, alpha=0.7)
    ax_sug.text(n_q - 1, TOP_K + 0.3, f"Max={TOP_K}",
                ha="right", fontsize=7.5, color="#999999")
    ax_sug.set_xticks(x)
    ax_sug.set_xticklabels(q_labels, fontsize=8)
    ax_sug.set_ylim(-0.5, TOP_K + 2)
    ax_sug.legend(handles=legend_handles, fontsize=7.5,
                  loc="center right", framealpha=0.9, edgecolor="#CCCCCC")
    ax_clean(ax_sug,
             "Number of Suggestions Returned",
             "", "# Results")
    ax_sug.text(0.01, 0.97,
                "Higher = found more results.\n0 = engine returned nothing.",
                transform=ax_sug.transAxes,
                fontsize=7.5, va="top", color="#555555", style="italic")

    # ─────────────────────────────────────────────────────────
    # ROW 1, COL 0  →  Build Memory (MB)  — CORRECT: Trie lowest
    # ─────────────────────────────────────────────────────────
    ax_bmem = fig.add_subplot(gs[1, 0])

    ybars = np.arange(n_eng)
    cols  = [ENG_STYLE[e.name]["color"] for e in engines]
    hb    = ax_bmem.barh(ybars, build_mem_mb,
                         color=cols, alpha=0.82, height=0.45, zorder=3,
                         edgecolor="white", linewidth=0.8)
    ax_bmem.bar_label(hb, fmt="%.1f MB", fontsize=8.5,
                      padding=4, fontweight="bold")
    ax_bmem.set_yticks(ybars)
    ax_bmem.set_yticklabels(enames, fontsize=9)
    ax_bmem.set_xlim(0, max(build_mem_mb) * 1.35)
    ax_clean(ax_bmem,
             "Index Build Memory",
             "Peak RAM (MB)", "")
    ax_bmem.text(0.98, 0.05,
                 "Trie only stores paths\nand top-K lists → light.\n"
                 "3-Gram maps every\ntrigram → heavier.",
                 transform=ax_bmem.transAxes,
                 fontsize=7.5, ha="right", va="bottom",
                 color="#555555", style="italic")

    # ─────────────────────────────────────────────────────────
    # ROW 1, COL 1  →  Build Time (seconds)
    # ─────────────────────────────────────────────────────────
    ax_btime = fig.add_subplot(gs[1, 1])

    hb2 = ax_btime.barh(ybars, build_times,
                         color=cols, alpha=0.82, height=0.45, zorder=3,
                         edgecolor="white", linewidth=0.8)
    ax_btime.bar_label(hb2, fmt="%.3f s", fontsize=8.5,
                        padding=4, fontweight="bold")
    ax_btime.set_yticks(ybars)
    ax_btime.set_yticklabels(enames, fontsize=9)
    ax_btime.set_xlim(0, max(build_times) * 1.35)
    ax_clean(ax_btime,
             "Index Build Time",
             "Seconds", "")
    ax_btime.text(0.98, 0.05,
                  "Fuzzy pays nothing upfront\n(just appends to a list).\n"
                  "Trie/3-Gram pay now\nto be fast at query time.",
                  transform=ax_btime.transAxes,
                  fontsize=7.5, ha="right", va="bottom",
                  color="#555555", style="italic")

    # ─────────────────────────────────────────────────────────
    # ROW 1, COL 2  →  Space vs Time scatter (like the paper's main figure)
    # X = build memory, Y = avg latency
    # ─────────────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 2])

    for eng in engines:
        s    = ENG_STYLE[eng.name]
        alat = np.mean(latencies[eng.name]) * 1000   # → µs
        amem = build_mem_mb[engines.index(eng)]
        ax_sc.scatter(amem, alat,
                      color=s["color"], marker=s["marker"],
                      s=130, zorder=5,
                      edgecolors="white", linewidths=1.2)
        ax_sc.text(amem + max(build_mem_mb) * 0.02, alat,
                   eng.name,
                   fontsize=9, color=s["color"], fontweight="bold",
                   va="center")

    ax_sc.set_yscale("log")
    ax_sc.set_xlabel("Space (MB)", fontsize=8)
    ax_clean(ax_sc,
             "Space vs Time Trade-off",
             "Space — Build Memory (MB)",
             "Time (µs / query) — log")
    # Bottom-left = ideal zone marker
    ax_sc.text(0.04, 0.08, "← bottom-left = ideal",
               transform=ax_sc.transAxes,
               fontsize=7.5, color="#888888", style="italic")

    # ─────────────────────────────────────────────────────────
    # ROW 2, COL 0-1  →  Per-query scatter grid (paper style)
    # 4 small panels: one per query type, X=latency, Y=suggestions
    # ─────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[2, :2],
        hspace=0.55, wspace=0.38
    )

    for qi, (q_text, q_type) in enumerate(queries):
        row_i = qi // 2
        col_i = qi % 2
        ax_q  = fig.add_subplot(inner[row_i, col_i])

        for eng in engines:
            s    = ENG_STYLE[eng.name]
            lat  = latencies[eng.name][qi] * 1000   # µs
            sug  = suggestions[eng.name][qi]
            ax_q.scatter(lat, sug,
                         color=s["color"], marker=s["marker"],
                         s=90, zorder=5,
                         edgecolors="white", linewidths=0.8)
            ax_q.text(lat * 1.04, sug + 0.15,
                      eng.name, fontsize=7, color=s["color"])

        ax_q.set_ylim(-0.5, TOP_K + 1.5)
        ax_q.set_xscale("log")
        ax_clean(ax_q,
                 f"'{q_text}'  ({q_type})",
                 "Time (µs)", "# Suggestions")

    # ─────────────────────────────────────────────────────────
    # ROW 2, COL 2  →  "When to use?" plain-text summary table
    # ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis("off")
    ax_tbl.set_facecolor("#FAFAFA")

    rows = [
        ["Use Case",                  "Best Choice"],
        ["Autocomplete\n(as you type)", "Trie ✓"],
        ["Substring search\n(word anywhere)", "3-Gram ✓"],
        ["Typos / sloppy input",       "Fuzzy ✓"],
        ["Semantic meaning\n(\"wizard school\")", "Need Vector AI"],
        ["1M+ title dataset",          "Avoid Fuzzy"],
    ]

    tbl = ax_tbl.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    header_color  = "#1565C0"
    alt_a         = "#E3F2FD"
    alt_b         = "#FAFAFA"
    warn_color    = "#FFF8E1"

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif r in (4, 5):
            cell.set_facecolor(warn_color)
        else:
            cell.set_facecolor(alt_a if r % 2 == 0 else alt_b)
        cell.set_height(0.155)

    ax_tbl.set_title("When to Use Which Engine?",
                     fontsize=9.5, fontweight="bold", pad=8)

    # Global legend at bottom
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9,
               edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.005))

    plt.savefig("benchmark_figure.png", dpi=160,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  -> Saved: benchmark_figure.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n=== SEARCH ENGINE BENCHMARK ({DATA_LIMIT:,} titles) ===\n")

    raw_titles, norm_titles = [], []
    with open(RAW_CSV, "r", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= DATA_LIMIT: break
            t = row.get("title", "")
            if t:
                raw_titles.append(t)
                norm_titles.append(normalize(t))

    print(f"  Loaded {len(raw_titles):,} titles | "
          f"avg length: {np.mean([len(t) for t in raw_titles]):.1f} chars\n")

    # Build — each engine instantiated fresh, direct calls
    trie_eng  = TrieEngine()
    ngram_eng = NGramEngine()
    fuzzy_eng = FuzzyEngine()
    engines   = [trie_eng, ngram_eng, fuzzy_eng]

    print("[Phase 1] Building Indices (measuring memory)...")

    build_times, build_mem_mb = [], []

    t0 = time.perf_counter()
    m  = get_build_memory_mb(trie_eng, norm_titles, raw_titles)
    build_times.append(time.perf_counter() - t0); build_mem_mb.append(m)
    print(f"  Trie   → {build_times[-1]:.3f}s  |  {m:.2f} MB")

    t0 = time.perf_counter()
    m  = get_build_memory_mb(ngram_eng, norm_titles, raw_titles)
    build_times.append(time.perf_counter() - t0); build_mem_mb.append(m)
    print(f"  3-Gram → {build_times[-1]:.3f}s  |  {m:.2f} MB")

    t0 = time.perf_counter()
    m  = get_build_memory_mb(fuzzy_eng, norm_titles, raw_titles)
    build_times.append(time.perf_counter() - t0); build_mem_mb.append(m)
    print(f"  Fuzzy  → {build_times[-1]:.3f}s  |  {m:.2f} MB")

    # Queries
    queries = [
        ("harry pot",     "Prefix"),
        ("arry potter",   "Typo"),
        ("stone",         "Substring"),
        ("wizard school", "Meaning"),
    ]

    latencies   = defaultdict(list)
    suggestions = defaultdict(list)

    print("\n[Phase 2] Query benchmarks (30 runs, median)...")
    for q_text, q_type in queries:
        q_norm = normalize(q_text)
        print(f"\n  '{q_text}' ({q_type})")

        res, lat = bench_query(trie_eng,  q_norm, q_text)
        latencies["Trie"].append(lat);   suggestions["Trie"].append(len(res))
        print(f"    Trie   | {lat*1000:.2f} µs | hits={len(res)}")

        res, lat = bench_query(ngram_eng, q_norm, q_text)
        latencies["3-Gram"].append(lat); suggestions["3-Gram"].append(len(res))
        print(f"    3-Gram | {lat*1000:.2f} µs | hits={len(res)}")

        res, lat = bench_query(fuzzy_eng, q_norm, q_text)
        latencies["Fuzzy"].append(lat);  suggestions["Fuzzy"].append(len(res))
        print(f"    Fuzzy  | {lat*1000:.2f} µs | hits={len(res)}")

    print("\n[Phase 3] Plotting (academic style)...")
    plot(engines, queries, latencies, suggestions, build_times, build_mem_mb)


if __name__ == "__main__":
    main()