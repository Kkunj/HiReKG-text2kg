"""
Run triplet-level UIR (Unique Information Ratio) across a fixed list of
KGgen experiment directories and save a single aggregated JSON in this
folder.

Pipeline per experiment directory
---------------------------------
    1. Discover all `doc_*` subfolders.
    2. For each doc, load `final_output.json` and pull `triples_final`
       (list of {subject, relation, object} dicts). Falls back to
       `chunks[].relations` if `triples_final` is missing.
    3. Compute per-doc UIR via `calculate_uir` (Sentence-BERT cosine +
       connected-components clustering at threshold 0.80 -- see
       uir_ratio.calculate_uir_from_matrix).
    4. Aggregate stats across docs: mean / median / std / min / max,
       and a "pooled" UIR over all triples in the experiment.

Output
------
    structural/uir_kggen_experiments_<timestamp>.json
        {
          metadata: { sbert_model, threshold, timestamp, ... },
          experiments: {
            <experiment_name>: {
              experiment_dir, docs_total, docs_processed, docs_skipped,
              docs_failed, aggregate { ... }, per_doc [ ... ]
            },
            ...
          }
        }

The model is loaded once via `get_sbert_model`'s cache and reused across
all 8 experiments and every doc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Make this folder importable so we can pull in the SBERT + UIR machinery.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uir_ratio import calculate_uir, get_sbert_model  # noqa: E402


# ---------------------------------------------------------------------------
# Default experiment list (user-supplied)
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_DIRS: List[str] = [
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_mine_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_mine_qwen3_14b",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_gpt4o",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_qwen3_14b",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_gpt4o",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_qwen3_14b",
]


# ---------------------------------------------------------------------------
# Triple loading
# ---------------------------------------------------------------------------

def load_triples_from_doc(doc_dir: str) -> List[Dict[str, str]]:
    """
    Load triples from `final_output.json` in a doc_* folder.

    Preference order:
        1. top-level `triples_final` (the standard field in this codebase)
        2. concatenation of `chunks[].relations` (fallback for older runs)

    Returns an empty list if neither is present or the file is missing.
    Only dicts with {subject, relation, object} keys are kept.
    """
    final_path = os.path.join(doc_dir, "final_output.json")
    if not os.path.isfile(final_path):
        return []

    with open(final_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_triples: List[Any] = []
    if isinstance(data.get("triples_final"), list) and data["triples_final"]:
        raw_triples = data["triples_final"]
    else:
        for chunk in data.get("chunks", []) or []:
            rels = chunk.get("relations") if isinstance(chunk, dict) else None
            if isinstance(rels, list):
                raw_triples.extend(rels)

    # Normalise to {subject, relation, object} dicts with stripped strings.
    triples: List[Dict[str, str]] = []
    for t in raw_triples:
        if not isinstance(t, dict):
            continue
        s = t.get("subject") or t.get("head") or t.get("source")
        r = t.get("relation") or t.get("predicate") or t.get("relation_type")
        o = t.get("object") or t.get("tail") or t.get("target")
        if s is None or r is None or o is None:
            continue
        triples.append({
            "subject": str(s).strip(),
            "relation": str(r).strip(),
            "object": str(o).strip(),
        })

    return triples


def list_doc_folders(experiment_dir: str) -> List[str]:
    """
    Return sorted list of subfolder names that contain a `final_output.json`.

    This handles both naming conventions present in the KGgen experiment tree:
        - doc_0, doc_1, ... (e.g. kggen_mine_*, kggen_scierc_*)
        - Wikipedia article names like Asus_VivoTab (kggen_redocred_*)

    For `doc_*` folders we numeric-sort on the suffix; everything else is
    sorted lexically. Files (e.g. _batch.log) are filtered out automatically
    by the final_output.json existence check.
    """
    if not os.path.isdir(experiment_dir):
        return []
    names = [
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
        and os.path.isfile(os.path.join(experiment_dir, d, "final_output.json"))
    ]

    def _key(name: str) -> Tuple[int, str]:
        if name.startswith("doc_"):
            suffix = name[len("doc_"):]
            try:
                return (0, f"{int(suffix):08d}")
            except ValueError:
                return (1, name)
        return (1, name)

    return sorted(names, key=_key)


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment(
    experiment_dir: str,
    threshold: float,
    sbert_model_name: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Compute per-doc UIR and aggregate stats for one experiment directory."""
    experiment_name = os.path.basename(experiment_dir.rstrip("\\/"))
    print("\n" + "=" * 78)
    print(f"Experiment: {experiment_name}")
    print(f"Path:       {experiment_dir}")
    print("=" * 78)

    if not os.path.isdir(experiment_dir):
        print(f"  [MISSING] directory not found -- skipping experiment")
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

    doc_names = list_doc_folders(experiment_dir)
    print(f"  Discovered {len(doc_names)} doc_* folders")

    per_doc: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    # Pooled bookkeeping: count clusters and triples across all docs.
    pooled_triples = 0
    pooled_unique_clusters = 0

    t_start = time.time()
    for idx, doc_name in enumerate(doc_names, start=1):
        doc_dir = os.path.join(experiment_dir, doc_name)
        try:
            triples = load_triples_from_doc(doc_dir)
        except Exception as exc:
            print(f"  [{idx:3d}/{len(doc_names)}] {doc_name}: LOAD FAILED -- {exc}")
            failed.append({"doc": doc_name, "stage": "load", "error": str(exc)})
            continue

        n = len(triples)
        if n == 0:
            skipped.append({"doc": doc_name, "reason": "no_triples"})
            print(f"  [{idx:3d}/{len(doc_names)}] {doc_name}: skipped (no triples)")
            continue
        if n == 1:
            # UIR is trivially 1.0; we record it but mark for exclusion from
            # the mean so a doc with one triple doesn't artificially raise it.
            per_doc.append({
                "doc": doc_name,
                "n_triples": 1,
                "n_unique_clusters": 1,
                "uir": 1.0,
                "trivial": True,
            })
            pooled_triples += 1
            pooled_unique_clusters += 1
            print(f"  [{idx:3d}/{len(doc_names)}] {doc_name}: trivial (1 triple, UIR=1.0)")
            continue

        try:
            uir, n_clusters, n_total, _reps = calculate_uir(
                triples,
                similarity_threshold=threshold,
                flag=0,                       # let split_to_edges convert dicts -> "s;r;o"
                verbose=False,                # keep output tidy
                batch_size=batch_size,
                model_type="sbert",
                sbert_model_name=sbert_model_name,
            )
        except Exception as exc:
            print(f"  [{idx:3d}/{len(doc_names)}] {doc_name}: UIR FAILED -- {exc}")
            traceback.print_exc()
            failed.append({"doc": doc_name, "stage": "uir", "error": str(exc)})
            continue

        per_doc.append({
            "doc": doc_name,
            "n_triples": int(n_total),
            "n_unique_clusters": int(n_clusters),
            "uir": float(uir),
            "trivial": False,
        })
        pooled_triples += int(n_total)
        pooled_unique_clusters += int(n_clusters)
        print(
            f"  [{idx:3d}/{len(doc_names)}] {doc_name}: "
            f"triples={n_total:4d}  clusters={n_clusters:4d}  uir={uir:.4f}"
        )

    elapsed = time.time() - t_start

    # --- Aggregate ---
    non_trivial = [d for d in per_doc if not d["trivial"]]
    if non_trivial:
        uir_arr = np.array([d["uir"] for d in non_trivial], dtype=float)
        aggregate: Dict[str, Any] = {
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
        # Mean over ALL docs incl. trivial (for completeness / reference).
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
    print(f"  Skipped: {len(skipped)}  Failed: {len(failed)}  "
          f"Elapsed: {elapsed:.1f}s")

    return {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "status": "ok",
        "docs_total": len(doc_names),
        "docs_processed": len(per_doc),
        "docs_skipped": skipped,
        "docs_failed": failed,
        "aggregate": aggregate,
        "per_doc": per_doc,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run triplet-level UIR over a list of KGgen experiment directories.",
    )
    parser.add_argument(
        "--experiment-dirs", nargs="+", default=None,
        help="Override the default list of experiment directories.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="SBERT cosine similarity threshold for clustering. Default: 0.80",
    )
    parser.add_argument(
        "--sbert-model", default="all-MiniLM-L6-v2",
        help="Sentence-BERT model name. Default: all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Encoder batch size. Default: 256",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: structural/uir_kggen_experiments_<timestamp>.json",
    )
    args = parser.parse_args()

    experiment_dirs = args.experiment_dirs or DEFAULT_EXPERIMENT_DIRS
    structural_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (
        structural_dir / f"uir_kggen_experiments_{timestamp}.json"
    )

    print("=" * 78)
    print("Triplet UIR -- KGgen experiments")
    print(f"  Experiments:  {len(experiment_dirs)}")
    print(f"  SBERT model:  {args.sbert_model}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Output:       {output_path}")
    print("=" * 78)

    # Warm up the SBERT model once so the per-experiment loops reuse it.
    print(f"Loading SBERT model '{args.sbert_model}' (one-time) ...")
    get_sbert_model(args.sbert_model)
    print("Model loaded.\n")

    wall_start = time.time()
    experiments_result: Dict[str, Any] = {}
    for exp_dir in experiment_dirs:
        result = evaluate_experiment(
            experiment_dir=exp_dir,
            threshold=args.threshold,
            sbert_model_name=args.sbert_model,
            batch_size=args.batch_size,
        )
        experiments_result[result["experiment_name"]] = result

    wall_elapsed = time.time() - wall_start

    # --- Cross-experiment summary table ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Experiment':<34} {'Docs':>6} {'MeanUIR':>10} {'PooledUIR':>12} {'Triples':>10}")
    print("-" * 78)
    for name, res in experiments_result.items():
        agg = res.get("aggregate") or {}
        mean_uir = agg.get("mean_uir_non_trivial")
        pooled = agg.get("pooled_uir")
        total = agg.get("pooled_total_triples")
        mean_str = f"{mean_uir:.4f}" if mean_uir is not None else "  N/A "
        pooled_str = f"{pooled:.4f}" if pooled is not None else "  N/A   "
        total_str = f"{total}" if total is not None else "N/A"
        print(f"{name:<34} {res['docs_processed']:>6} {mean_str:>10} {pooled_str:>12} {total_str:>10}")

    output_payload = {
        "metadata": {
            "timestamp": timestamp,
            "sbert_model": args.sbert_model,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "experiments_count": len(experiment_dirs),
            "clustering": "connected_components_union_find",
            "wall_time_seconds": round(wall_elapsed, 2),
        },
        "experiments": experiments_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"Saved: {output_path}")
    print(f"Total wall time: {wall_elapsed:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
