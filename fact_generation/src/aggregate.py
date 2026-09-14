"""
Stage 4 — Aggregate dataset-level statistics.

Computes summary statistics across all documents for sanity-checking and paper reporting.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("atomic_facts")


def run_stage4(
    stage1_dir: str,
    audit_dir: str,
    final_dir: str,
    stats_output_path: str,
    config: Dict[str, Any],
    dataset_name: str = "",
) -> Dict[str, Any]:
    """
    Compute and save dataset-level aggregate statistics.

    Args:
        stage1_dir:        Directory containing Stage 1 candidate JSONs
        audit_dir:         Directory containing Stage 3 audit JSONs
        final_dir:         Directory containing Stage 3 final JSONs
        stats_output_path: Path for the output dataset_stats.json
        config:            Parsed config.yaml dict
        dataset_name:      Name of the dataset (for reporting)

    Returns:
        The computed statistics dict.
    """
    audit_path = Path(audit_dir)
    final_path = Path(final_dir)
    stage1_path = Path(stage1_dir)

    audit_files = sorted(audit_path.glob("*.json"))
    final_files = sorted(final_path.glob("*.json"))

    if not audit_files:
        logger.warning("[Stage4] No audit files found. Nothing to aggregate.")
        return {}

    # Collect per-document stats
    total_candidates = 0
    total_kept = 0
    drop_reasons: Dict[str, int] = {
        "span_not_in_text": 0,
        "contradicted": 0,
        "not_stated": 0,
        "parse_error": 0,
        "unknown_verdict": 0,
    }
    per_doc_verification_rates: List[float] = []
    per_doc_fact_counts: List[int] = []

    for audit_file in audit_files:
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

        entries = audit_data.get("entries", [])
        if not entries:
            continue

        n_candidates = len(entries)
        n_kept = sum(1 for e in entries if e.get("kept_in_final", False))
        total_candidates += n_candidates
        total_kept += n_kept
        per_doc_fact_counts.append(n_kept)

        # Verification rate for this doc
        rate = n_kept / n_candidates if n_candidates > 0 else 0.0
        per_doc_verification_rates.append(rate)

        # Count drop reasons
        for entry in entries:
            reason = entry.get("drop_reason")
            if reason and reason in drop_reasons:
                drop_reasons[reason] += 1
            elif reason:
                drop_reasons.setdefault(reason, 0)
                drop_reasons[reason] += 1

    # Compute distribution stats
    n_documents = len(audit_files)
    filtering_rate = 1.0 - (total_kept / total_candidates) if total_candidates > 0 else 0.0

    def _dist_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    stats = {
        "dataset": dataset_name,
        "n_documents": n_documents,
        "total_candidates_generated": total_candidates,
        "total_facts_after_filtering": total_kept,
        "filtering_rate": round(filtering_rate, 4),
        "verification_rate_distribution": _dist_stats(per_doc_verification_rates),
        "drop_reasons": drop_reasons,
        "facts_per_document": _dist_stats([float(x) for x in per_doc_fact_counts]),
    }

    # Sanity-check warnings
    mean_vr = stats["verification_rate_distribution"]["mean"]
    if mean_vr > 0.95:
        logger.warning(
            f"[Stage4] SANITY CHECK: Mean verification rate is {mean_vr:.2%} (>95%). "
            "Generator and verifier may be agreeing too readily. Manual spot-check recommended."
        )
    elif mean_vr < 0.60:
        logger.warning(
            f"[Stage4] SANITY CHECK: Mean verification rate is {mean_vr:.2%} (<60%). "
            "Generator quality may be low or verifier may be too strict. Investigate."
        )
    else:
        logger.info(f"[Stage4] Verification rate: {mean_vr:.2%} (healthy range 60-95%)")

    # Write stats
    output_file = Path(stats_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[Stage4] Aggregate stats: {n_documents} docs, "
        f"{total_candidates} candidates, {total_kept} kept, "
        f"filtering rate {filtering_rate:.1%}"
    )

    return stats
