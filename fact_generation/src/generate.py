"""
Stage 1 — Generate candidate atomic facts using GPT-5.

For each document, calls GPT-5 to extract ~15 atomic facts with supporting spans.
Performs deterministic span-verification in code (not by trusting the LLM).
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_clients import OpenAIClient

logger = logging.getLogger("atomic_facts")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to a single space, strip, and lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()


def span_in_source(span: str, source_text: str) -> bool:
    """Check if span is a substring of source_text (whitespace-normalized, case-insensitive)."""
    return normalize_whitespace(span) in normalize_whitespace(source_text)


def generate_facts_for_doc(
    doc_id: str,
    source_text: str,
    client: OpenAIClient,
    generator_prompt: str,
    target_facts: int = 15,
) -> Dict[str, Any]:
    """
    Generate candidate atomic facts for a single document.

    Returns a dict with:
        - doc_id
        - facts: list of {fact, supporting_span, span_mismatch}
        - generator_model
        - error (if any)
    """
    user_prompt = f"SOURCE TEXT:\n\n{source_text}"

    try:
        parsed = client.generate_json(
            system_prompt=generator_prompt,
            user_prompt=user_prompt,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error(f"[Stage1] {doc_id}: API/parse error — {exc}")
        return {
            "doc_id": doc_id,
            "facts": [],
            "generator_model": client.model,
            "error": str(exc),
        }

    raw_facts = parsed.get("facts", [])

    # Cap at target_facts
    if len(raw_facts) > target_facts:
        logger.info(f"[Stage1] {doc_id}: capping {len(raw_facts)} facts to {target_facts}")
        raw_facts = raw_facts[:target_facts]

    # Validate spans deterministically
    validated = []
    for entry in raw_facts:
        fact_text = entry.get("fact", "")
        supporting_span = entry.get("supporting_span", "")
        mismatch = not span_in_source(supporting_span, source_text) if supporting_span else True
        if mismatch:
            logger.debug(f"[Stage1] {doc_id}: span mismatch for fact: {fact_text[:80]}...")
        validated.append({
            "fact": fact_text,
            "supporting_span": supporting_span,
            "span_mismatch": mismatch,
        })

    return {
        "doc_id": doc_id,
        "facts": validated,
        "generator_model": client.model,
        "error": None,
    }


def run_stage1(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    prompt_path: str,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Run Stage 1 across all documents in input_dir.

    Args:
        input_dir:   Directory containing {doc_id}.txt files
        output_dir:  Directory for stage1_candidates output
        config:      Parsed config.yaml dict
        prompt_path: Path to generator.txt prompt
        limit:       Optional limit on number of docs to process (for smoke tests)

    Returns:
        List of doc_ids that were processed.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gen_cfg = config["generator"]
    pipe_cfg = config["pipeline"]

    # Load prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        generator_prompt = f.read()

    # Discover documents
    doc_files = sorted(input_path.glob("*.txt"))
    if limit:
        doc_files = doc_files[:limit]

    # Skip already-processed docs (resumability) — but re-process error files
    to_process = []
    for doc_file in doc_files:
        doc_id = doc_file.stem
        out_file = output_path / f"{doc_id}.json"
        if out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("error"):
                    logger.info(f"[Stage1] {doc_id}: previous run had error, re-processing")
                else:
                    logger.info(f"[Stage1] {doc_id}: already exists, skipping")
                    continue
            except (json.JSONDecodeError, OSError):
                logger.info(f"[Stage1] {doc_id}: corrupt output file, re-processing")
        to_process.append((doc_id, doc_file))

    if not to_process:
        logger.info("[Stage1] All documents already processed.")
        return [f.stem for f in doc_files]

    logger.info(f"[Stage1] Processing {len(to_process)} documents (skipped {len(doc_files) - len(to_process)} existing)")

    # Build client
    client = OpenAIClient(
        model=gen_cfg["model"],
        temperature=gen_cfg["temperature"],
        max_output_tokens=gen_cfg["max_output_tokens"],
        max_retries=pipe_cfg["max_retries"],
        backoff_base=pipe_cfg["retry_backoff_base"],
    )

    def _process_one(doc_id: str, doc_file: Path) -> str:
        source_text = doc_file.read_text(encoding="utf-8")
        result = generate_facts_for_doc(
            doc_id=doc_id,
            source_text=source_text,
            client=client,
            generator_prompt=generator_prompt,
            target_facts=pipe_cfg["target_facts_per_doc"],
        )
        out_file = output_path / f"{doc_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        n_facts = len(result["facts"])
        n_mismatch = sum(1 for fact in result["facts"] if fact["span_mismatch"])
        logger.info(f"[Stage1] {doc_id}: {n_facts} facts, {n_mismatch} span mismatches")
        return doc_id

    processed = []
    max_workers = gen_cfg.get("max_workers", 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, doc_id, doc_file): doc_id
            for doc_id, doc_file in to_process
        }
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                processed.append(future.result())
            except Exception as exc:
                logger.error(f"[Stage1] {doc_id}: unhandled error — {exc}")
                # Write error record so we don't silently lose this doc
                error_result = {
                    "doc_id": doc_id,
                    "facts": [],
                    "generator_model": gen_cfg["model"],
                    "error": str(exc),
                }
                out_file = output_path / f"{doc_id}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                processed.append(doc_id)

    all_doc_ids = [f.stem for f in doc_files]
    logger.info(f"[Stage1] Complete. {len(processed)} new + {len(all_doc_ids) - len(processed)} cached.")
    return all_doc_ids
