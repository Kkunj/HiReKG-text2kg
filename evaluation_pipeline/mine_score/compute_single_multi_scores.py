"""
Compute MINE scores broken down by single-entity vs multi-entity facts.

For each experiment in the results/ directory, loads the per-doc result files,
looks up the entity classification from the corresponding dataset (MINE, scierc,
or redocred), and computes separate accuracy scores for single-entity and
multi-entity facts. Ambiguous facts are treated as multi-entity.

Outputs a single JSON file: results/summary_single_multiple.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Dataset paths
MINE_ANSWERS = Path(__file__).resolve().parent.parent.parent / "datasets" / "MINE" / "answers.json"
SCIERC_STAGE3 = Path(__file__).resolve().parent.parent.parent / "datasets" / "scierc" / "atomic_facts" / "stage3_final"
REDOCRED_STAGE3 = Path(__file__).resolve().parent.parent.parent / "datasets" / "windows_redocred" / "atomic_facts" / "stage3_final"

# Directories to skip
SKIP_DIRS = {"comparisons"}


def detect_dataset(experiment_name: str) -> str:
    """Detect which dataset an experiment belongs to based on its name."""
    lower = experiment_name.lower()
    if "redocred" in lower:
        return "redocred"
    elif "scierc" in lower:
        return "scierc"
    else:
        return "mine"


def load_mine_classifications() -> Dict[int, Dict[str, List[int]]]:
    """
    Load MINE answers.json and return per-row classification indices.
    Returns: {row_idx: {"single": [...], "multi": [...]}}
    """
    with open(MINE_ANSWERS, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for row_entry in data["rows"]:
        row_idx = row_entry["row_idx"]
        row = row_entry["row"]
        single = set(row.get("single-entity", []))
        multi = set(row.get("multiple-entity", []))
        ambiguous = set(row.get("ambiguous", []))
        # Ambiguous treated as multi-entity
        multi = multi | ambiguous
        # Also build the answer list for text matching
        answers = [a["answer"] for a in row["answers"]]
        result[row_idx] = {
            "single": single,
            "multi": multi,
            "answers": answers,
        }
    return result


def load_stage3_classifications(stage3_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all stage3_final JSON files sorted by filename.
    Returns list of dicts with single/multi indices and facts list.
    """
    entries = []
    for p in sorted(stage3_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        single = set(doc.get("single-entity", []))
        multi = set(doc.get("multiple-entity", []))
        ambiguous = set(doc.get("ambiguous", []))
        multi = multi | ambiguous
        entries.append({
            "single": single,
            "multi": multi,
            "facts": doc["facts"],
        })
    return entries


def process_experiment(exp_dir: Path, dataset: str, classifications) -> Optional[Dict[str, Any]]:
    """
    Process one experiment directory: compute overall, single-entity, and
    multi-entity MINE scores.
    """
    # Find all results_N.json files
    result_files = sorted(
        exp_dir.glob("results_*.json"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    if not result_files:
        return None

    total_single = 0
    correct_single = 0
    total_multi = 0
    correct_multi = 0
    total_overall = 0
    correct_overall = 0
    unmatched = 0

    for rf in result_files:
        doc_idx = int(rf.stem.split("_")[1])

        with open(rf, encoding="utf-8") as f:
            results = json.load(f)

        # Get the classification data for this doc
        if dataset == "mine":
            if doc_idx not in classifications:
                continue
            cls_data = classifications[doc_idx]
            doc_facts = cls_data["answers"]
            single_indices = cls_data["single"]
            multi_indices = cls_data["multi"]
        else:
            if doc_idx >= len(classifications):
                continue
            cls_data = classifications[doc_idx]
            doc_facts = cls_data["facts"]
            single_indices = cls_data["single"]
            multi_indices = cls_data["multi"]

        # For each result entry, match it to a fact index by text
        for item in results:
            if "correct_answer" not in item:
                # Some result files contain only aggregate accuracy — skip
                continue
            answer = item["correct_answer"]
            eval_val = item["evaluation"]
            is_correct = int(eval_val) == 1 if isinstance(eval_val, (int, float)) else str(eval_val).strip() == "1"

            total_overall += 1
            if is_correct:
                correct_overall += 1

            # Find which fact index this answer corresponds to
            fact_idx = None
            for i, fact in enumerate(doc_facts):
                if fact == answer:
                    fact_idx = i
                    break

            if fact_idx is None:
                # Answer not found in classification data — count as multi
                total_multi += 1
                if is_correct:
                    correct_multi += 1
                unmatched += 1
                continue

            if fact_idx in single_indices:
                total_single += 1
                if is_correct:
                    correct_single += 1
            else:
                # multi or ambiguous or unclassified → multi bucket
                total_multi += 1
                if is_correct:
                    correct_multi += 1

    return {
        "experiment": exp_dir.name,
        "dataset": dataset,
        "overall": {
            "total": total_overall,
            "correct": correct_overall,
            "accuracy_pct": round(correct_overall / total_overall * 100, 2) if total_overall else 0,
        },
        "single_entity": {
            "total": total_single,
            "correct": correct_single,
            "accuracy_pct": round(correct_single / total_single * 100, 2) if total_single else 0,
        },
        "multi_entity": {
            "total": total_multi,
            "correct": correct_multi,
            "accuracy_pct": round(correct_multi / total_multi * 100, 2) if total_multi else 0,
        },
        "unmatched_facts": unmatched,
    }


def main():
    print("Loading dataset classifications...")

    mine_cls = load_mine_classifications()
    scierc_cls = load_stage3_classifications(SCIERC_STAGE3)
    redocred_cls = load_stage3_classifications(REDOCRED_STAGE3)

    print(f"  MINE: {len(mine_cls)} rows")
    print(f"  SciERC: {len(scierc_cls)} docs")
    print(f"  ReDOCRED: {len(redocred_cls)} docs")

    # Process each experiment
    all_results = []

    for entry in sorted(RESULTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS:
            continue

        dataset = detect_dataset(entry.name)
        if dataset == "mine":
            cls = mine_cls
        elif dataset == "scierc":
            cls = scierc_cls
        else:
            cls = redocred_cls

        result = process_experiment(entry, dataset, cls)
        if result is None:
            print(f"  SKIP {entry.name}: no result files")
            continue

        all_results.append(result)
        r = result
        print(
            f"  {r['experiment']:50s} | "
            f"overall={r['overall']['accuracy_pct']:5.1f}% "
            f"single={r['single_entity']['accuracy_pct']:5.1f}% ({r['single_entity']['total']:4d}) "
            f"multi={r['multi_entity']['accuracy_pct']:5.1f}% ({r['multi_entity']['total']:4d})"
        )

    # Save
    output_path = RESULTS_DIR / "summary_single_multiple.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_results)} experiment results to {output_path}")


if __name__ == "__main__":
    main()
