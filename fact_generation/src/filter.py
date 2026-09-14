"""
Stage 3 — Filtering & final fact list.

Applies deterministic filtering rules to produce the locked fact list per document.
Outputs both the clean final list and a full audit log.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("atomic_facts")


def filter_facts(verified_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply filtering rules to verified facts.

    Keep a fact if:
        - span_mismatch == False
        - verifier_verdict == "SUPPORTED"

    Returns the full list with `kept_in_final` and `drop_reason` added to each entry.
    """
    results = []
    for entry in verified_facts:
        kept = True
        drop_reason = None

        # Check for parse / API errors first
        if entry.get("verifier_error") or entry.get("verifier_verdict") is None:
            kept = False
            drop_reason = "parse_error"
        elif entry.get("span_mismatch", False):
            kept = False
            drop_reason = "span_not_in_text"
        elif entry["verifier_verdict"] == "CONTRADICTED":
            kept = False
            drop_reason = "contradicted"
        elif entry["verifier_verdict"] == "NOT_STATED":
            kept = False
            drop_reason = "not_stated"
        elif entry["verifier_verdict"] != "SUPPORTED":
            kept = False
            drop_reason = "unknown_verdict"

        results.append({
            **entry,
            "kept_in_final": kept,
            "drop_reason": drop_reason,
        })

    return results


def run_stage3(
    stage2_dir: str,
    final_dir: str,
    audit_dir: str,
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[str]:
    """
    Run Stage 3 across all documents with Stage 2 output.

    Args:
        stage2_dir: Directory containing Stage 2 verified JSONs
        final_dir:  Directory for stage3_final output (clean fact lists)
        audit_dir:  Directory for stage3_audit output (full audit logs)
        config:     Parsed config.yaml dict
        limit:      Optional limit on number of docs

    Returns:
        List of doc_ids that were processed.
    """
    stage2_path = Path(stage2_dir)
    final_path = Path(final_dir)
    audit_path = Path(audit_dir)
    final_path.mkdir(parents=True, exist_ok=True)
    audit_path.mkdir(parents=True, exist_ok=True)

    stage2_files = sorted(stage2_path.glob("*.json"))
    if limit:
        stage2_files = stage2_files[:limit]

    processed = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for s2_file in stage2_files:
        doc_id = s2_file.stem

        with open(s2_file, "r", encoding="utf-8") as f:
            stage2_data = json.load(f)

        verified_facts = stage2_data.get("verified_facts", [])
        generator_model = stage2_data.get("generator_model", "unknown")
        verifier_model = stage2_data.get("verifier_model", "unknown")

        # Apply filtering
        filtered = filter_facts(verified_facts)

        # Build final fact list (just the strings, in order)
        final_facts = [
            entry["fact"]
            for entry in filtered
            if entry["kept_in_final"]
        ]

        # Build audit log (one entry per candidate fact)
        audit_entries = []
        for entry in filtered:
            audit_entries.append({
                "doc_id": doc_id,
                "fact_idx": entry["fact_idx"],
                "fact": entry["fact"],
                "generator_supporting_span": entry["generator_supporting_span"],
                "span_mismatch": entry["span_mismatch"],
                "generator_model": generator_model,
                "verifier_supporting_span": entry.get("verifier_supporting_span"),
                "verifier_reasoning": entry.get("verifier_reasoning"),
                "verifier_verdict": entry.get("verifier_verdict"),
                "verifier_model": verifier_model,
                "kept_in_final": entry["kept_in_final"],
                "drop_reason": entry["drop_reason"],
                "timestamp": timestamp,
            })

        # Write final fact list
        final_file = final_path / f"{doc_id}.json"
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "facts": final_facts,
                "n_candidates": len(filtered),
                "n_kept": len(final_facts),
            }, f, indent=2, ensure_ascii=False)

        # Write audit log
        audit_file = audit_path / f"{doc_id}.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "generator_model": generator_model,
                "verifier_model": verifier_model,
                "entries": audit_entries,
            }, f, indent=2, ensure_ascii=False)

        n_kept = len(final_facts)
        n_dropped = len(filtered) - n_kept
        logger.info(f"[Stage3] {doc_id}: {n_kept} kept, {n_dropped} dropped")
        processed.append(doc_id)

    logger.info(f"[Stage3] Complete. {len(processed)} documents filtered.")
    return processed
