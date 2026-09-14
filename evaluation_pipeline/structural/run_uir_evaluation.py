"""
UIR (Unique Information Ratio) evaluation runner.

Runs entity_redundancy_metrics.calculate_entity_redundancy across all
doc_* folders in an experiment directory and aggregates the results.

Usage:
    python run_uir_evaluation.py <experiment_dir> <approach_name> <output_dir>

Outputs (written to <output_dir>):
    - <approach_name>_uir_results.json   : per-doc + aggregate metrics
    - <approach_name>_uir.log            : full execution log
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---- make the structural package importable ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entity_redundancy_metrics import calculate_entity_redundancy


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str, approach: str) -> logging.Logger:
    logger = logging.getLogger(f"uir_{approach}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Entity + description loading (handles all three approaches)
# ---------------------------------------------------------------------------

def _build_descriptions_from_triples(entities: list[str], triples: list[dict]) -> list[str]:
    """
    Build a description for each entity by collecting all triples where it
    appears as subject or object.  Format: "relation -> other_entity" lines.
    """
    entity_set = {e.lower() for e in entities}
    desc_map: dict[str, list[str]] = {e.lower(): [] for e in entities}

    for t in triples:
        subj = str(t.get("subject", "")).lower()
        rel = str(t.get("relation", ""))
        obj = str(t.get("object", "")).lower()

        if subj in desc_map:
            desc_map[subj].append(f"{rel} -> {obj}")
        if obj in desc_map:
            desc_map[obj].append(f"{subj} -> {rel}")

    descriptions = []
    for e in entities:
        parts = desc_map.get(e.lower(), [])
        descriptions.append("; ".join(parts) if parts else e)
    return descriptions


def load_entities_and_descriptions(
    doc_dir: str, approach: str, logger: logging.Logger,
) -> tuple[list[str], list[str]]:
    """
    Load entity names and generate descriptions from a single doc folder.

    Returns (entities, descriptions) — same-length lists.
    """
    final_path = os.path.join(doc_dir, "final_output.json")
    with open(final_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 1. Get the refined entity list ---
    entities_refined = data.get("entities_refined", [])
    if not entities_refined:
        logger.warning(f"  No entities_refined found in {final_path}")
        return [], []

    # Normalise: some approaches store dicts, others store strings
    entities = []
    for e in entities_refined:
        if isinstance(e, dict):
            entities.append(str(e.get("name", e)))
        else:
            entities.append(str(e))

    # --- 2. Build descriptions ---
    if approach == "rakg":
        # RAKG has rich entity descriptions in entities_raw
        raw_desc_map = {}
        for raw in data.get("entities_raw", []):
            if isinstance(raw, dict) and "name" in raw and "description" in raw:
                raw_desc_map[raw["name"].lower()] = raw["description"]

        # Collect triples for fallback
        triples = data.get("triples_final", [])
        fallback = _build_descriptions_from_triples(entities, triples)

        descriptions = []
        for i, e in enumerate(entities):
            desc = raw_desc_map.get(e.lower())
            if desc:
                # Take first description segment (before ;;;) to keep it concise
                descriptions.append(desc.split(";;;")[0].strip())
            else:
                descriptions.append(fallback[i])
    else:
        # KGGen / Our approach: build from triples
        if approach == "kggen":
            # triples live inside chunks[].relations
            triples = []
            for chunk in data.get("chunks", []):
                for rel in chunk.get("relations", []):
                    triples.append(rel)
        else:
            # our_approach: triples_final at top level
            triples = data.get("triples_final", [])

        descriptions = _build_descriptions_from_triples(entities, triples)

    logger.debug(f"  Loaded {len(entities)} entities, {len(descriptions)} descriptions")
    return entities, descriptions


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    experiment_dir: str,
    approach: str,
    output_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.75,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{approach}_uir.log")
    logger = setup_logging(log_path, approach)

    logger.info("=" * 70)
    logger.info(f"UIR (Entity Redundancy) Evaluation")
    logger.info(f"  Approach      : {approach}")
    logger.info(f"  Experiment dir: {experiment_dir}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info(f"  Model         : {model_name}")
    logger.info(f"  Threshold     : {threshold}")
    logger.info("=" * 70)

    # Pre-load the sentence transformer model once
    logger.info(f"Loading SentenceTransformer model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded.")

    # Discover doc_* folders
    doc_dirs = sorted(
        [
            d
            for d in os.listdir(experiment_dir)
            if d.startswith("doc_") and os.path.isdir(os.path.join(experiment_dir, d))
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    logger.info(f"Found {len(doc_dirs)} document folders.")

    per_doc_results = []
    all_redundancy_scores = []
    all_redundant_pair_counts = []
    all_entity_counts = []
    skipped = []
    failed = []

    wall_start = time.time()

    for idx, doc_name in enumerate(doc_dirs):
        doc_path = os.path.join(experiment_dir, doc_name)
        logger.info(f"[{idx + 1}/{len(doc_dirs)}] Processing {doc_name} ...")

        try:
            entities, descriptions = load_entities_and_descriptions(
                doc_path, approach, logger,
            )

            if len(entities) <= 1:
                logger.info(f"  Skipped (<=1 entity)")
                skipped.append(doc_name)
                per_doc_results.append({
                    "doc": doc_name,
                    "status": "skipped",
                    "reason": f"only {len(entities)} entity",
                    "entity_count": len(entities),
                })
                continue

            redundancy, details, pair_count = calculate_entity_redundancy(
                entities=entities,
                descriptions=descriptions,
                model_name=model_name,
                threshold=threshold,
                visualize=False,
                verbose=False,
                model=model,
            )

            # Extract top pairs for logging
            top_pairs = []
            for i, j, score in details.get("top_redundant_pairs", [])[:5]:
                top_pairs.append({
                    "entity_a": entities[i],
                    "entity_b": entities[j],
                    "score": round(float(score), 4),
                })

            doc_result = {
                "doc": doc_name,
                "status": "success",
                "entity_count": len(entities),
                "overall_redundancy": round(redundancy, 6),
                "redundant_pair_count": pair_count,
                "max_redundancy": round(details["max_redundancy"], 6),
                "min_redundancy": round(details["min_redundancy"], 6),
                "std_redundancy": round(details["std_redundancy"], 6),
                "top_redundant_pairs": top_pairs,
                "entities": entities,
            }
            per_doc_results.append(doc_result)

            all_redundancy_scores.append(redundancy)
            all_redundant_pair_counts.append(pair_count)
            all_entity_counts.append(len(entities))

            logger.info(
                f"  entities={len(entities):3d}  "
                f"redundancy={redundancy:.4f}  "
                f"redundant_pairs={pair_count}  "
                f"max={details['max_redundancy']:.4f}"
            )
            if top_pairs:
                best = top_pairs[0]
                logger.debug(
                    f"  Most redundant: '{best['entity_a']}' <-> "
                    f"'{best['entity_b']}' = {best['score']}"
                )

        except Exception as exc:
            logger.error(f"  FAILED: {exc}")
            logger.debug(traceback.format_exc())
            failed.append(doc_name)
            per_doc_results.append({
                "doc": doc_name,
                "status": "failed",
                "error": str(exc),
            })

    wall_elapsed = time.time() - wall_start

    # --- Aggregate ---
    if all_redundancy_scores:
        arr = np.array(all_redundancy_scores)
        pairs_arr = np.array(all_redundant_pair_counts)
        entity_arr = np.array(all_entity_counts)

        aggregate = {
            "docs_evaluated": len(all_redundancy_scores),
            "docs_skipped": len(skipped),
            "docs_failed": len(failed),
            "mean_redundancy": round(float(np.mean(arr)), 6),
            "median_redundancy": round(float(np.median(arr)), 6),
            "std_redundancy": round(float(np.std(arr)), 6),
            "min_redundancy": round(float(np.min(arr)), 6),
            "max_redundancy": round(float(np.max(arr)), 6),
            "total_redundant_pairs": int(np.sum(pairs_arr)),
            "mean_redundant_pairs_per_doc": round(float(np.mean(pairs_arr)), 4),
            "mean_entity_count": round(float(np.mean(entity_arr)), 2),
            "total_entities_evaluated": int(np.sum(entity_arr)),
            # UIR = 1 - mean_redundancy  (higher is better)
            "UIR_score": round(1.0 - float(np.mean(arr)), 6),
        }
    else:
        aggregate = {
            "docs_evaluated": 0,
            "docs_skipped": len(skipped),
            "docs_failed": len(failed),
            "error": "No documents were successfully evaluated",
        }

    # --- Log summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"AGGREGATE RESULTS — {approach}")
    logger.info("=" * 70)
    for k, v in aggregate.items():
        logger.info(f"  {k:35s}: {v}")
    logger.info(f"  wall_time_seconds              : {wall_elapsed:.2f}")
    if skipped:
        logger.info(f"  skipped_docs                   : {skipped}")
    if failed:
        logger.warning(f"  failed_docs                    : {failed}")
    logger.info("=" * 70)

    # --- Save ---
    output = {
        "metadata": {
            "approach": approach,
            "experiment_dir": experiment_dir,
            "model_name": model_name,
            "threshold": threshold,
            "wall_time_seconds": round(wall_elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "aggregate": aggregate,
        "per_doc": per_doc_results,
    }

    result_path = os.path.join(output_dir, f"{approach}_uir_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {result_path}")
    logger.info(f"Log saved to     {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UIR evaluation on an experiment directory.")
    parser.add_argument("experiment_dir", help="Path to the experiment directory (contains doc_* folders)")
    parser.add_argument("approach", help="Approach name (kggen, rakg, our_approach)")
    parser.add_argument("output_dir", help="Directory to write results and logs")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--threshold", type=float, default=0.75, help="Redundancy threshold")
    args = parser.parse_args()

    run_evaluation(
        experiment_dir=args.experiment_dir,
        approach=args.approach,
        output_dir=args.output_dir,
        model_name=args.model,
        threshold=args.threshold,
    )
