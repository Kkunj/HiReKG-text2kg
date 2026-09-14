"""
Aggregate faithfulness results across all experiments.

For each experiment subdirectory in results/, loads its dataset_summary.json
and combines them into a single JSON file capturing overall faithfulness,
support counts, and per-doc statistics.

Outputs: results/summary_faithfulness.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_DIR / "summary_faithfulness.json"

SKIP_FILENAMES = {"summary_faithfulness.json"}


def detect_dataset(experiment_name: str) -> str:
    lower = experiment_name.lower()
    if "redocred" in lower:
        return "redocred"
    if "scierc" in lower:
        return "scierc"
    return "mine"


def detect_approach(experiment_name: str) -> str:
    lower = experiment_name.lower()
    if lower.startswith("ours") or "_ours_" in lower or lower.startswith("our_"):
        return "ours"
    if lower.startswith("kggen"):
        return "kggen"
    if lower.startswith("rakg"):
        return "rakg"
    return "unknown"


def process_experiment(exp_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = exp_dir / "dataset_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    dataset = detect_dataset(exp_dir.name)
    approach = detect_approach(exp_dir.name)

    return {
        "experiment": exp_dir.name,
        "dataset": dataset,
        "approach": approach,
        "n_docs": summary.get("n_docs", 0),
        "n_triples_total": summary.get("n_triples_total", 0),
        "n_evaluated_total": summary.get("n_evaluated_total", 0),
        "n_supported_total": summary.get("n_supported_total", 0),
        "n_not_supported_total": summary.get("n_not_supported_total", 0),
        "n_malformed_total": summary.get("n_malformed_total", 0),
        "n_judge_errors_total": summary.get("n_judge_errors_total", 0),
        "overall_micro_faithfulness": summary.get("overall_micro_faithfulness", 0.0),
        "faithfulness_mean": summary.get("faithfulness_mean", 0.0),
        "faithfulness_std": summary.get("faithfulness_std", 0.0),
        "not_supported_rate_mean": summary.get("not_supported_rate_mean", 0.0),
        "not_supported_rate_std": summary.get("not_supported_rate_std", 0.0),
    }


def main():
    all_results: List[Dict[str, Any]] = []

    for entry in sorted(RESULTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_FILENAMES:
            continue

        result = process_experiment(entry)
        if result is None:
            print(f"  SKIP {entry.name}: no dataset_summary.json")
            continue

        all_results.append(result)
        print(
            f"  {result['experiment']:60s} | "
            f"docs={result['n_docs']:4d} "
            f"triples={result['n_triples_total']:6d} "
            f"micro_faithfulness={result['overall_micro_faithfulness']:.4f} "
            f"mean={result['faithfulness_mean']:.4f}"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_results)} experiment summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
