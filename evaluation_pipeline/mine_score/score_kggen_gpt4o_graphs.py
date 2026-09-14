"""
Compute the single-/multi-entity MINE decomposition for the KGGen GPT-4o
MINE-1 cell (Table 1, row "GPT-4o / KGGen / MINE-1").

Every other cell is handled directly by compute_single_multi_scores.py, which
expects per-doc files named `results_<MINE row_idx>.json`. The KGGen GPT-4o
MINE graphs came from the upstream kg-gen release and its per-doc result files
are numbered by kg-gen's own file order, which does not line up with MINE row
indices. This script recovers the mapping by fingerprinting each result file on
its first `correct_answer`, stages the files under the expected names in a temp
directory, and then defers to the shared `process_experiment`.

Usage:
    python score_kggen_gpt4o_graphs.py
"""
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_SRC = (
    REPO_ROOT / "baselines" / "kggen" / "experiments"
    / "kggen_mine_gpt4o_graphs" / "mine_results"
)

sys.path.insert(0, str(SCRIPT_DIR))
from compute_single_multi_scores import load_mine_classifications, process_experiment

mine_cls = load_mine_classifications()
print(f"Loaded MINE classifications: {len(mine_cls)} rows")

# Build content -> row_idx lookup (use the first answer of each row as a fingerprint).
import json
ans_to_row = {}
for ri, data in mine_cls.items():
    for a in data["answers"]:
        ans_to_row.setdefault(a, ri)

# process_experiment expects files named results_<N>.json (N = MINE row_idx).
# Our filename indices don't map cleanly, so derive row_idx per file from its first answer.
with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    staged = 0
    skipped_unknown = 0
    skipped_collision = 0
    used_rows = set()
    for src in sorted(RESULTS_SRC.glob("*_results*.json"),
                      key=lambda p: int(p.stem.split("_")[0])):
        with open(src, encoding="utf-8") as f:
            res = json.load(f)
        first = next((r["correct_answer"] for r in res if "correct_answer" in r), None)
        row_idx = ans_to_row.get(first) if first else None
        if row_idx is None:
            skipped_unknown += 1
            continue
        if row_idx in used_rows:
            skipped_collision += 1
            continue
        used_rows.add(row_idx)
        dst = tdp / f"results_{row_idx}.json"
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
        staged += 1
    print(f"Staged {staged} files (skipped: unknown-content={skipped_unknown}, collisions={skipped_collision})")

    result = process_experiment(tdp, "mine", mine_cls)

if result is None:
    print("No result produced")
    sys.exit(1)

r = result
print()
print("=== Summary (KGGen GPT-4o graphs, MINE dataset) ===")
print(f"Overall:       {r['overall']['correct']}/{r['overall']['total']} = {r['overall']['accuracy_pct']}%")
print(f"Single-entity: {r['single_entity']['correct']}/{r['single_entity']['total']} = {r['single_entity']['accuracy_pct']}%")
print(f"Multi-entity:  {r['multi_entity']['correct']}/{r['multi_entity']['total']} = {r['multi_entity']['accuracy_pct']}%")
print(f"Unmatched facts: {r['unmatched_facts']}")
