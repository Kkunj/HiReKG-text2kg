"""
Run triplet-level UIR on a flat-file KGgen experiment and MERGE the result
into an existing KGgen results JSON.

This handles the `kggen-graphs-gpt-4o` style of output, which differs from
the other KGgen experiments:

    Standard KGgen layout (handled by run_uir_kggen_experiments.py):
        <experiment_dir>/
            doc_0/final_output.json   # has `triples_final` (list of dicts)
            doc_1/...

    Flat layout (handled HERE):
        <experiment_dir>/
            1.json          # has top-level `relations` (list of [s, r, o] lists)
            2.json
            ...

Each top-level *.json file is treated as one "doc" (one input document's
extracted KG). The relations list of triples-as-3-element-lists is
normalised into {subject, relation, object} dicts and fed through the
existing `calculate_uir`, so the algorithm, threshold, and clustering are
byte-identical to the other runs -- numbers stay directly comparable.

After computation the experiment record is INSERTED into the target
KGgen results JSON under `experiments[<name>]`. The original file is
backed up to <file>.bak.<timestamp> before being overwritten.

Defaults match the task at hand:
    --experiment-dir : <PROJECT_ROOT>/temp/kggen/kggen-graphs-gpt-4o
    --target-json    : structural/uir_kggen_experiments_20260525_004405.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uir_ratio import calculate_uir, get_sbert_model  # noqa: E402


# ---------------------------------------------------------------------------
# Loader for the flat <id>.json layout
# ---------------------------------------------------------------------------

def load_triples_from_flat_json(json_path: str) -> List[Dict[str, str]]:
    """
    Read a flat KGgen output file and return a list of
    {subject, relation, object} dicts.

    Accepted shapes (in order of preference):
        1. top-level `relations`: list of [s, r, o] lists OR dicts
        2. top-level `edges` of the same shape (fallback)
        3. top-level `triples_final` (defensive fallback)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw: List[Any] = []
    for key in ("relations", "edges", "triples_final"):
        candidate = data.get(key)
        if isinstance(candidate, list) and candidate:
            # Heuristic: only treat this key as the triples source if its
            # entries are list/tuple of len>=3 OR dicts with s/r/o-ish keys.
            sample = candidate[0]
            if (isinstance(sample, (list, tuple)) and len(sample) >= 3) or (
                isinstance(sample, dict)
                and any(k in sample for k in ("subject", "head", "source"))
            ):
                raw = candidate
                break

    triples: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            s, r, o = item[0], item[1], item[2]
        elif isinstance(item, dict):
            s = item.get("subject") or item.get("head") or item.get("source")
            r = item.get("relation") or item.get("predicate") or item.get("relation_type")
            o = item.get("object") or item.get("tail") or item.get("target")
        else:
            continue
        if s is None or r is None or o is None:
            continue
        triples.append({
            "subject": str(s).strip(),
            "relation": str(r).strip(),
            "object": str(o).strip(),
        })
    return triples


def list_flat_json_files(experiment_dir: str) -> List[str]:
    """List *.json files in `experiment_dir`, numeric-sorted where possible."""
    if not os.path.isdir(experiment_dir):
        return []
    files = [f for f in os.listdir(experiment_dir) if f.endswith(".json")]

    def _key(name: str):
        stem = name[:-len(".json")]
        try:
            return (0, int(stem), "")
        except ValueError:
            return (1, 0, stem)

    return sorted(files, key=_key)


# ---------------------------------------------------------------------------
# Per-experiment evaluator (mirrors run_uir_kggen_experiments.evaluate_experiment
# but reads flat <id>.json files instead of doc_*/final_output.json)
# ---------------------------------------------------------------------------

def evaluate_flat_experiment(
    experiment_dir: str,
    experiment_name: str,
    threshold: float,
    sbert_model_name: str,
    batch_size: int,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"Experiment: {experiment_name}  (flat-JSON layout)")
    print(f"Path:       {experiment_dir}")
    print("=" * 78)

    if not os.path.isdir(experiment_dir):
        print(f"  [MISSING] directory not found")
        return {
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "status": "missing_directory",
            "docs_total": 0,
            "docs_processed": 0,
            "docs_skipped": [],
            "docs_failed": [],
            "aggregate": None,
            "per_doc": [],
        }

    files = list_flat_json_files(experiment_dir)
    print(f"  Discovered {len(files)} JSON files")

    per_doc: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    pooled_triples = 0
    pooled_unique_clusters = 0

    t_start = time.time()
    for idx, fname in enumerate(files, start=1):
        fpath = os.path.join(experiment_dir, fname)
        doc_id = fname[:-len(".json")]

        try:
            triples = load_triples_from_flat_json(fpath)
        except Exception as exc:
            print(f"  [{idx:3d}/{len(files)}] {fname}: LOAD FAILED -- {exc}")
            failed.append({"doc": doc_id, "stage": "load", "error": str(exc)})
            continue

        n = len(triples)
        if n == 0:
            skipped.append({"doc": doc_id, "reason": "no_triples"})
            print(f"  [{idx:3d}/{len(files)}] {fname}: skipped (no triples)")
            continue
        if n == 1:
            per_doc.append({
                "doc": doc_id, "n_triples": 1, "n_unique_clusters": 1,
                "uir": 1.0, "trivial": True,
            })
            pooled_triples += 1
            pooled_unique_clusters += 1
            print(f"  [{idx:3d}/{len(files)}] {fname}: trivial (1 triple)")
            continue

        try:
            uir, n_clusters, n_total, _ = calculate_uir(
                triples,
                similarity_threshold=threshold,
                flag=0,
                verbose=False,
                batch_size=batch_size,
                model_type="sbert",
                sbert_model_name=sbert_model_name,
            )
        except Exception as exc:
            print(f"  [{idx:3d}/{len(files)}] {fname}: UIR FAILED -- {exc}")
            failed.append({"doc": doc_id, "stage": "uir", "error": str(exc)})
            continue

        per_doc.append({
            "doc": doc_id,
            "n_triples": int(n_total),
            "n_unique_clusters": int(n_clusters),
            "uir": float(uir),
            "trivial": False,
        })
        pooled_triples += int(n_total)
        pooled_unique_clusters += int(n_clusters)
        print(
            f"  [{idx:3d}/{len(files)}] {fname}: "
            f"triples={n_total:4d}  clusters={n_clusters:4d}  uir={uir:.4f}"
        )

    elapsed = time.time() - t_start

    non_trivial = [d for d in per_doc if not d["trivial"]]
    if non_trivial:
        uir_arr = np.array([d["uir"] for d in non_trivial], dtype=float)
        aggregate = {
            "docs_non_trivial": len(non_trivial),
            "docs_trivial_excluded": sum(1 for d in per_doc if d["trivial"]),
            "mean_uir_non_trivial": float(np.mean(uir_arr)),
            "median_uir_non_trivial": float(np.median(uir_arr)),
            "std_uir_non_trivial": float(np.std(uir_arr)),
            "min_uir_non_trivial": float(np.min(uir_arr)),
            "max_uir_non_trivial": float(np.max(uir_arr)),
            "pooled_total_triples": int(pooled_triples),
            "pooled_total_clusters": int(pooled_unique_clusters),
            "pooled_uir": (
                float(pooled_unique_clusters / pooled_triples)
                if pooled_triples > 0 else None
            ),
            "wall_time_seconds": round(elapsed, 2),
        }
        all_arr = np.array([d["uir"] for d in per_doc], dtype=float)
        aggregate["mean_uir_all_docs"] = float(np.mean(all_arr))
    else:
        aggregate = {
            "docs_non_trivial": 0,
            "docs_trivial_excluded": sum(1 for d in per_doc if d["trivial"]),
            "wall_time_seconds": round(elapsed, 2),
            "note": "no non-trivial docs to aggregate",
        }

    print("-" * 78)
    if non_trivial:
        print(
            f"  Mean UIR (non-trivial, n={aggregate['docs_non_trivial']}): "
            f"{aggregate['mean_uir_non_trivial']:.4f}"
        )
        print(
            f"  Pooled UIR: {aggregate['pooled_uir']:.4f}  "
            f"({aggregate['pooled_total_clusters']}/{aggregate['pooled_total_triples']})"
        )
    print(f"  Skipped: {len(skipped)}  Failed: {len(failed)}  Elapsed: {elapsed:.1f}s")

    return {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "status": "ok",
        "docs_total": len(files),
        "docs_processed": len(per_doc),
        "docs_skipped": skipped,
        "docs_failed": failed,
        "aggregate": aggregate,
        "per_doc": per_doc,
        "layout": "flat_json_files",
    }


# ---------------------------------------------------------------------------
# Merge result into the existing KGgen results JSON
# ---------------------------------------------------------------------------

def merge_into_target(target_json: Path, exp_name: str, exp_result: Dict[str, Any]) -> None:
    """
    Load target_json, insert/overwrite experiments[exp_name] = exp_result,
    write back. Keeps a .bak.<timestamp> of the original.
    """
    if not target_json.is_file():
        raise FileNotFoundError(f"Target JSON not found: {target_json}")

    with open(target_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    experiments = payload.setdefault("experiments", {})
    is_overwrite = exp_name in experiments

    # Back up the original file before mutation
    backup_path = target_json.with_suffix(
        target_json.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(target_json, backup_path)
    print(f"\nBackup written: {backup_path}")

    experiments[exp_name] = exp_result

    # Stamp the merge in metadata so it is auditable later.
    meta = payload.setdefault("metadata", {})
    merges = meta.setdefault("merged_experiments", [])
    merges.append({
        "experiment_name": exp_name,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overwritten_existing": is_overwrite,
        "layout": exp_result.get("layout", "unknown"),
        "source_dir": exp_result.get("experiment_dir"),
    })

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"{'Overwrote' if is_overwrite else 'Inserted'} experiments[{exp_name!r}] in {target_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_target_json(structural_dir: Path) -> Path:
    """Pick the newest uir_kggen_experiments_*.json in the structural dir."""
    candidates = sorted(structural_dir.glob("uir_kggen_experiments_*.json"))
    # Filter out our own backups / merge stamps
    candidates = [c for c in candidates if c.suffix == ".json"]
    if not candidates:
        raise FileNotFoundError(
            "No uir_kggen_experiments_*.json found; pass --target-json explicitly."
        )
    return candidates[-1]


def main() -> None:
    structural_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run triplet-level UIR on a flat-JSON KGgen experiment and merge into existing results.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=r"C:\<PROJECT_ROOT>\graph_rag\temp\kggen\kggen-graphs-gpt-4o",
        help="Path to the flat-JSON experiment directory.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Key under which to store this experiment in the merged JSON. "
             "Defaults to the basename of --experiment-dir.",
    )
    parser.add_argument(
        "--target-json",
        default=None,
        help="Existing KGgen results JSON to merge into. "
             "Defaults to the newest uir_kggen_experiments_*.json in structural/.",
    )
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    exp_name = args.experiment_name or os.path.basename(args.experiment_dir.rstrip("\\/"))
    target_json = Path(args.target_json) if args.target_json else _default_target_json(structural_dir)

    print("=" * 78)
    print("Flat-JSON KGgen UIR runner (merge mode)")
    print(f"  Experiment dir:   {args.experiment_dir}")
    print(f"  Experiment name:  {exp_name}")
    print(f"  Target JSON:      {target_json}")
    print(f"  SBERT model:      {args.sbert_model}")
    print(f"  Threshold:        {args.threshold}")
    print("=" * 78)

    print(f"Loading SBERT model '{args.sbert_model}' (one-time) ...")
    get_sbert_model(args.sbert_model)
    print("Model loaded.\n")

    result = evaluate_flat_experiment(
        experiment_dir=args.experiment_dir,
        experiment_name=exp_name,
        threshold=args.threshold,
        sbert_model_name=args.sbert_model,
        batch_size=args.batch_size,
    )

    merge_into_target(target_json, exp_name, result)

    agg = result.get("aggregate") or {}
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  Docs processed:        {result['docs_processed']}")
    print(f"  Mean UIR (non-triv):   {agg.get('mean_uir_non_trivial')}")
    print(f"  Pooled UIR:            {agg.get('pooled_uir')}")
    print(f"  Total triples:         {agg.get('pooled_total_triples')}")
    print(f"  Total unique clusters: {agg.get('pooled_total_clusters')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
