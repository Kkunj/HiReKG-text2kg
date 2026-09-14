"""
Compute approach-level aggregates and add them to each UIR result JSON,
then print a cross-approach comparison table.

For each of {kggen, rakg, our_approach} this script:
    1. Picks the most recent `uir_<approach>_experiments_*.json` in this
       folder (ignoring .bak.* files).
    2. Computes an `approach_aggregate` block containing:
         - overall   : aggregated across all experiments in that approach
         - by_dataset: separate slices for mine / redocred / scierc
                       (plus `other` for experiments that don't fit those)
         - by_model  : separate slices for gpt4o / qwen3_8b / qwen3_14b /
                       other  (best-effort from experiment names)
    3. Writes the block back into the JSON under top-level
       `approach_aggregate` (with a .bak.<timestamp> backup of the original).
    4. Prints a cross-approach comparison table.

Aggregate metrics (per scope)
-----------------------------
    experiment_mean_uir   : simple mean of per-experiment mean_uir_non_trivial
                            (each experiment counts equally)
    doc_weighted_mean_uir : weighted by docs_processed
                            (each document counts equally)
    pooled_uir            : sum(total_clusters) / sum(total_triples)
                            (each triple counts equally)

These three answer subtly different questions; report them all.

Dataset / model classification is based on experiment names. Anything that
doesn't match is placed under `other` so it's still counted but separately.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


STRUCTURAL_DIR = Path(__file__).resolve().parent

APPROACH_TO_GLOB = {
    "kggen": "uir_kggen_experiments_*.json",
    "rakg": "uir_rakg_experiments_*.json",
    "our_approach": "uir_ours_experiments_*.json",
}

DATASETS = ("mine", "redocred", "scierc")
MODELS = ("gpt4o", "qwen3_8b", "qwen3_14b")


def latest_json(glob_pattern: str) -> Optional[Path]:
    """Newest non-backup file in STRUCTURAL_DIR matching the glob."""
    candidates = [
        p for p in STRUCTURAL_DIR.glob(glob_pattern)
        if p.suffix == ".json" and ".bak." not in p.name
    ]
    return sorted(candidates)[-1] if candidates else None


def classify_dataset(name: str) -> str:
    n = name.lower().replace("-", "_")
    for ds in DATASETS:
        if ds in n:
            return ds
    return "other"


def classify_model(name: str) -> str:
    n = name.lower().replace("-", "_")
    if "qwen3_14b" in n:
        return "qwen3_14b"
    if "qwen3_8b" in n:
        return "qwen3_8b"
    if "gpt4o" in n or "gpt_4o" in n:
        return "gpt4o"
    return "other"


def aggregate(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build an aggregate dict from a list of experiment records."""
    usable = []
    for r in records:
        agg = r.get("aggregate") or {}
        mean_uir = agg.get("mean_uir_non_trivial")
        docs = r.get("docs_processed", 0) or 0
        if mean_uir is None or docs == 0:
            continue
        usable.append({
            "name": r.get("experiment_name"),
            "mean_uir": float(mean_uir),
            "docs": int(docs),
            "triples": int(agg.get("pooled_total_triples", 0) or 0),
            "clusters": int(agg.get("pooled_total_clusters", 0) or 0),
        })
    if not usable:
        return None

    mean_uirs = np.array([u["mean_uir"] for u in usable])
    doc_counts = np.array([u["docs"] for u in usable], dtype=float)
    total_triples = sum(u["triples"] for u in usable)
    total_clusters = sum(u["clusters"] for u in usable)

    return {
        "experiments_count": len(usable),
        "experiment_names": [u["name"] for u in usable],
        "experiment_mean_uir": float(np.mean(mean_uirs)),
        "doc_weighted_mean_uir": float(np.average(mean_uirs, weights=doc_counts)),
        "pooled_uir": (float(total_clusters / total_triples) if total_triples > 0 else None),
        "total_docs": int(sum(u["docs"] for u in usable)),
        "total_triples": int(total_triples),
        "total_clusters": int(total_clusters),
    }


def build_approach_aggregate(experiments: Dict[str, Dict]) -> Dict[str, Any]:
    records = list(experiments.values())

    by_dataset: Dict[str, Any] = {}
    for ds in DATASETS + ("other",):
        slice_recs = [r for r in records if classify_dataset(r.get("experiment_name", "")) == ds]
        agg = aggregate(slice_recs) if slice_recs else None
        if agg:
            by_dataset[ds] = agg

    by_model: Dict[str, Any] = {}
    for m in MODELS + ("other",):
        slice_recs = [r for r in records if classify_model(r.get("experiment_name", "")) == m]
        agg = aggregate(slice_recs) if slice_recs else None
        if agg:
            by_model[m] = agg

    return {
        "overall": aggregate(records),
        "by_dataset": by_dataset,
        "by_model": by_model,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_with_backup(path: Path, payload: Dict[str, Any]) -> None:
    backup = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"    Updated {path.name}")
    print(f"    Backup : {backup.name}")


def main() -> None:
    print("=" * 78)
    print("Computing approach aggregates and updating JSON files")
    print("=" * 78)

    approach_results: Dict[str, Dict[str, Any]] = {}

    for approach, glob in APPROACH_TO_GLOB.items():
        path = latest_json(glob)
        print(f"\n  Approach: {approach}")
        if path is None:
            print(f"    [SKIP] no result JSON found ({glob})")
            continue
        print(f"    Source : {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        agg_block = build_approach_aggregate(payload.get("experiments", {}))
        payload["approach_aggregate"] = agg_block
        write_with_backup(path, payload)
        approach_results[approach] = agg_block

    # ----------------------------------------------------------------------
    # Cross-approach comparison
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CROSS-APPROACH COMPARISON")
    print("=" * 78)

    print("\nOverall (every experiment in each approach):")
    print(f"  {'Approach':<14} {'Exps':>5} {'MeanUIR':>10} {'WeightedUIR':>13} {'PooledUIR':>12} {'Triples':>10}")
    print("  " + "-" * 73)
    for approach, agg_block in approach_results.items():
        overall = agg_block.get("overall") or {}
        if not overall:
            print(f"  {approach:<14}    --- no data ---")
            continue
        print(
            f"  {approach:<14} "
            f"{overall['experiments_count']:>5} "
            f"{overall['experiment_mean_uir']:>10.4f} "
            f"{overall['doc_weighted_mean_uir']:>13.4f} "
            f"{overall['pooled_uir']:>12.4f} "
            f"{overall['total_triples']:>10}"
        )

    print("\nBy dataset:")
    for ds in DATASETS:
        print(f"\n  Dataset: {ds}")
        print(f"    {'Approach':<14} {'Exps':>5} {'MeanUIR':>10} {'PooledUIR':>12} {'Triples':>10}")
        print("    " + "-" * 56)
        for approach, agg_block in approach_results.items():
            slice_agg = agg_block.get("by_dataset", {}).get(ds)
            if slice_agg is None:
                print(f"    {approach:<14} {'N/A':>5}")
                continue
            print(
                f"    {approach:<14} "
                f"{slice_agg['experiments_count']:>5} "
                f"{slice_agg['experiment_mean_uir']:>10.4f} "
                f"{slice_agg['pooled_uir']:>12.4f} "
                f"{slice_agg['total_triples']:>10}"
            )

    print("\nBy model:")
    for m in MODELS:
        print(f"\n  Model: {m}")
        print(f"    {'Approach':<14} {'Exps':>5} {'MeanUIR':>10} {'PooledUIR':>12} {'Triples':>10}")
        print("    " + "-" * 56)
        for approach, agg_block in approach_results.items():
            slice_agg = agg_block.get("by_model", {}).get(m)
            if slice_agg is None:
                print(f"    {approach:<14} {'N/A':>5}")
                continue
            print(
                f"    {approach:<14} "
                f"{slice_agg['experiments_count']:>5} "
                f"{slice_agg['experiment_mean_uir']:>10.4f} "
                f"{slice_agg['pooled_uir']:>12.4f} "
                f"{slice_agg['total_triples']:>10}"
            )

    # Flag any experiments that didn't classify into a dataset, so the user
    # can manually fix the naming or accept the "other" bucket.
    print("\n" + "-" * 78)
    print("Unclassified experiments (slotted under 'other'):")
    any_unclassified = False
    for approach, agg_block in approach_results.items():
        other = agg_block.get("by_dataset", {}).get("other")
        if other:
            any_unclassified = True
            print(f"  {approach}: {other['experiment_names']}")
    if not any_unclassified:
        print("  (none)")


if __name__ == "__main__":
    main()
