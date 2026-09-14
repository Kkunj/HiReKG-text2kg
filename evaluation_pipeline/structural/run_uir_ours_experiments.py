"""
Run triplet-level UIR across all `our_approach` experiment directories and
save a single aggregated JSON in this folder.

Our pipeline's `final_output.json` shares the same schema used elsewhere
in this codebase (`triples_final` -> list of {subject, relation, object,
evidence, ...} dicts; only subject/relation/object are used), so this
script reuses the loaders and per-experiment evaluator from
`run_uir_kggen_experiments.py` verbatim and only changes:
    - the list of default experiment directories
    - the output filename prefix

Output:
    structural/uir_ours_experiments_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_uir_kggen_experiments import evaluate_experiment  # noqa: E402
from uir_ratio import get_sbert_model  # noqa: E402


# ---------------------------------------------------------------------------
# Default experiment list (user-supplied)
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_DIRS: List[str] = [
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\MINE_batch_results",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_14b_batch",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_redocred_gpt4o",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_14b",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_scierc_gpt4o",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_8b",
    r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_14b",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run triplet-level UIR over our_approach experiment directories.",
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
        help="Output JSON path. Default: structural/uir_ours_experiments_<timestamp>.json",
    )
    args = parser.parse_args()

    experiment_dirs = args.experiment_dirs or DEFAULT_EXPERIMENT_DIRS
    structural_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (
        structural_dir / f"uir_ours_experiments_{timestamp}.json"
    )

    print("=" * 78)
    print("Triplet UIR -- our_approach experiments")
    print(f"  Experiments:  {len(experiment_dirs)}")
    print(f"  SBERT model:  {args.sbert_model}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Output:       {output_path}")
    print("=" * 78)

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
            "approach": "our_approach",
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
