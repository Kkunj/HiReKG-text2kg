"""
Aggregate object resolution logs across all 9 (dataset x LLM) configs.

For each (dataset, LLM) config, this script walks the experiment directory,
loads every doc's object_resolution.json, and produces:
  - per-doc stats
  - per-config (dataset x LLM) aggregate
  - per-dataset aggregate (across LLMs)
  - per-LLM aggregate (across datasets)
  - global aggregate

Output is written as one JSON file.
"""

import json
from pathlib import Path
from collections import Counter

# ---- (dataset, LLM) -> root experiment directory ----
CONFIG_DIRS = {
    ("MINE-1",    "GPT-4o"):    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\MINE_batch_results",
    ("Re-DocRED", "GPT-4o"):    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_redocred_gpt4o",
    ("SciERC",    "GPT-4o"):    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_scierc_gpt4o",
    ("MINE-1",    "Qwen3-14B"): r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_14b_batch",
    ("Re-DocRED", "Qwen3-14B"): r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_14b",
    ("SciERC",    "Qwen3-14B"): r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_14b",
    ("MINE-1",    "Qwen3-8B"):  r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_8b",
    ("Re-DocRED", "Qwen3-8B"):  r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_8b",
    ("SciERC",    "Qwen3-8B"):  r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_8b",
}

OUTPUT_FILE = Path(__file__).parent / "object_resolution_aggregate.json"

COUNT_KEYS = (
    "n_in", "n_out", "n_clean", "n_repaired", "n_failed",
    "n_single_ent", "n_multi_ent", "n_multi_out", "n_multi_entries",
)


def summarize_one(path):
    """Compute per-doc statistics from one object_resolution.json file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("resolution_details", [])
    stats   = data.get("statistics", {})

    n_in   = len(details)
    n_out  = sum(len(d.get("after", [])) for d in details)

    n_clean    = stats.get("clean_triples", 0)
    n_repaired = stats.get("repaired_triples", 0)
    n_failed   = stats.get("failed_repairs", 0)

    n_single_ent = sum(1 for d in details if d.get("repair_method") == "single_entity")
    n_multi_ent  = sum(1 for d in details if d.get("repair_method") == "multiple_entities")

    multi_entries = [d for d in details if d.get("repair_method") == "multiple_entities"]
    n_multi_out   = sum(len(d.get("after", [])) for d in multi_entries)

    return {
        "n_in":            n_in,
        "n_out":           n_out,
        "n_clean":         n_clean,
        "n_repaired":      n_repaired,
        "n_failed":        n_failed,
        "n_single_ent":    n_single_ent,
        "n_multi_ent":     n_multi_ent,
        "n_multi_out":     n_multi_out,
        "n_multi_entries": len(multi_entries),
    }


def derived(c):
    """Add percentages / derived ratios to a counter-like dict."""
    n_in = c["n_in"]
    n_out = c["n_out"]
    out = dict(c)
    out["clean_pct"]  = 100 * c["n_clean"]    / n_in if n_in else 0.0
    out["repair_pct"] = 100 * c["n_repaired"] / n_in if n_in else 0.0
    out["fail_pct"]   = 100 * c["n_failed"]   / n_in if n_in else 0.0
    out["single_pct"] = 100 * c["n_single_ent"] / n_in if n_in else 0.0
    out["multi_pct"]  = 100 * c["n_multi_ent"]  / n_in if n_in else 0.0
    out["yield_pct"]  = 100 * (n_out - n_in) / n_in if n_in else 0.0
    out["avg_split"]  = (c["n_multi_out"] / c["n_multi_entries"]) if c["n_multi_entries"] else 0.0
    return out


def walk_config(root):
    """Find every object_resolution.json under root (one per doc)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob("*/object_resolution.json"))


# ---- Run per-config aggregation ----
per_config = {}   # (dataset, llm) -> { per_doc, totals }
rows = []         # flat list for cross-cuts

for (dataset, llm), root in CONFIG_DIRS.items():
    files = walk_config(root)
    if not files:
        print(f"[skip] no object_resolution.json files under: {root}")
        continue

    per_doc = {}
    totals = Counter()
    for fp in files:
        doc_name = fp.parent.name
        s = summarize_one(fp)
        per_doc[doc_name] = s
        for k in COUNT_KEYS:
            totals[k] += s[k]

    config_summary = {
        "dataset":  dataset,
        "llm":      llm,
        "root":     str(root),
        "n_docs":   len(per_doc),
        "totals":   derived(dict(totals)),
        "per_doc":  {name: derived(s) for name, s in per_doc.items()},
    }
    per_config[f"{dataset} | {llm}"] = config_summary
    rows.append((dataset, llm, dict(totals)))
    print(f"[ok] {dataset:<11} {llm:<11} {len(per_doc):>3} docs, "
          f"{totals['n_in']:>6} in -> {totals['n_out']:>6} out")

if not rows:
    print("No data found. Check CONFIG_DIRS paths.")
    raise SystemExit

# ---- Cross-cut aggregates ----
def aggregate(rows_subset):
    a = Counter()
    for c in rows_subset:
        for k in COUNT_KEYS:
            a[k] += c[k]
    return derived(dict(a))


by_dataset = {}
for dataset in sorted({r[0] for r in rows}):
    subset = [c for d, _, c in rows if d == dataset]
    by_dataset[dataset] = {"n_configs": len(subset), "totals": aggregate(subset)}

by_llm = {}
for llm in sorted({r[1] for r in rows}):
    subset = [c for _, l, c in rows if l == llm]
    by_llm[llm] = {"n_configs": len(subset), "totals": aggregate(subset)}

global_totals = aggregate([c for _, _, c in rows])

# ---- Assemble final JSON ----
output = {
    "per_config":        per_config,
    "aggregate_by_dataset": by_dataset,
    "aggregate_by_llm":     by_llm,
    "global_aggregate":     global_totals,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote: {OUTPUT_FILE}")

# ---- Console summary ----
print("\n=== Per-(dataset x LLM) totals ===\n")
hdr = (
    f"{'Dataset':<11}{'LLM':<12}{'#docs':>7}"
    f"{'#in':>8}{'#out':>8}"
    f"{'Clean%':>8}{'Rep%':>7}{'Fail%':>7}"
    f"{'Multi%':>8}{'AvgSplit':>10}{'Yield%':>9}"
)
print(hdr)
print("-" * len(hdr))
for key, cfg in per_config.items():
    t = cfg["totals"]
    print(
        f"{cfg['dataset']:<11}{cfg['llm']:<12}{cfg['n_docs']:>7}"
        f"{t['n_in']:>8}{t['n_out']:>8}"
        f"{t['clean_pct']:>8.1f}{t['repair_pct']:>7.1f}{t['fail_pct']:>7.1f}"
        f"{t['multi_pct']:>8.1f}{t['avg_split']:>10.2f}{t['yield_pct']:>+9.1f}"
    )

print("\n=== Aggregated by dataset ===\n")
print(f"{'Dataset':<11}{'#in':>8}{'#out':>8}{'Clean%':>9}{'Rep%':>8}{'Fail%':>8}{'Multi%':>9}{'Yield%':>9}")
for dataset, info in by_dataset.items():
    t = info["totals"]
    print(
        f"{dataset:<11}{t['n_in']:>8}{t['n_out']:>8}"
        f"{t['clean_pct']:>9.1f}{t['repair_pct']:>8.1f}"
        f"{t['fail_pct']:>8.1f}{t['multi_pct']:>9.1f}"
        f"{t['yield_pct']:>+9.1f}"
    )

print("\n=== Aggregated by LLM ===\n")
print(f"{'LLM':<12}{'#in':>8}{'#out':>8}{'Clean%':>9}{'Rep%':>8}{'Fail%':>8}{'Multi%':>9}{'Yield%':>9}")
for llm, info in by_llm.items():
    t = info["totals"]
    print(
        f"{llm:<12}{t['n_in']:>8}{t['n_out']:>8}"
        f"{t['clean_pct']:>9.1f}{t['repair_pct']:>8.1f}"
        f"{t['fail_pct']:>8.1f}{t['multi_pct']:>9.1f}"
        f"{t['yield_pct']:>+9.1f}"
    )

print("\n=== Global headline ===\n")
t = global_totals
print(f"Total raw triples processed:    {t['n_in']}")
print(f"Total final triples produced:   {t['n_out']}")
print(f"  Clean:                        {t['n_clean']:>7} ({t['clean_pct']:.1f}%)")
print(f"  Repaired:                     {t['n_repaired']:>7} ({t['repair_pct']:.1f}%)")
print(f"    single-entity:              {t['n_single_ent']:>7} ({t['single_pct']:.1f}%)")
print(f"    multiple-entity:            {t['n_multi_ent']:>7} ({t['multi_pct']:.1f}%)")
print(f"  Failed (kept as-is):          {t['n_failed']:>7} ({t['fail_pct']:.1f}%)")
print(f"Net triple yield:               {t['yield_pct']:+.1f}%")