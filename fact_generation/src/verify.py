"""
Stage 2 — Cross-family verification using Gemini.

For each candidate fact from Stage 1, independently verifies whether the fact
is supported by the source text using a different model family (Gemini).
One fact per API call — no batching to avoid order effects.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_clients import GeminiClient

logger = logging.getLogger("atomic_facts")


def verify_single_fact(
    source_text: str,
    fact: str,
    generator_span: str,
    client: GeminiClient,
    verifier_prompt: str,
) -> Dict[str, Any]:
    """
    Verify a single candidate fact against the source text.

    Returns a dict with:
        - verifier_supporting_span
        - reasoning
        - verdict (SUPPORTED / CONTRADICTED / NOT_STATED)
        - error (if any)
    """
    user_prompt = (
        f"SOURCE TEXT:\n\n{source_text}\n\n"
        f"---\n\n"
        f"CANDIDATE FACT:\n{fact}\n\n"
        f"PROPOSED SUPPORTING SPAN:\n{generator_span}"
    )

    try:
        parsed = client.generate_json(
            system_prompt=verifier_prompt,
            user_prompt=user_prompt,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error(f"[Stage2] Verification API/parse error for fact: {fact[:80]}... — {exc}")
        return {
            "verifier_supporting_span": None,
            "reasoning": None,
            "verdict": None,
            "error": str(exc),
        }

    verdict = parsed.get("verdict", "").upper().strip()
    if verdict not in ("SUPPORTED", "CONTRADICTED", "NOT_STATED"):
        logger.warning(f"[Stage2] Unexpected verdict '{verdict}' for fact: {fact[:80]}...")
        return {
            "verifier_supporting_span": parsed.get("verifier_supporting_span"),
            "reasoning": parsed.get("reasoning"),
            "verdict": None,
            "error": f"Invalid verdict: {verdict}",
        }

    return {
        "verifier_supporting_span": parsed.get("verifier_supporting_span"),
        "reasoning": parsed.get("reasoning"),
        "verdict": verdict,
        "error": None,
    }


def run_stage2(
    input_dir: str,
    stage1_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    prompt_path: str,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Run Stage 2 across all documents that have Stage 1 output.

    Args:
        input_dir:   Directory containing {doc_id}.txt source files
        stage1_dir:  Directory containing Stage 1 output JSONs
        output_dir:  Directory for stage2_verified output
        config:      Parsed config.yaml dict
        prompt_path: Path to verifier.txt prompt
        limit:       Optional limit on number of docs

    Returns:
        List of doc_ids that were processed.
    """
    input_path = Path(input_dir)
    stage1_path = Path(stage1_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ver_cfg = config["verifier"]
    pipe_cfg = config["pipeline"]

    # Load prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        verifier_prompt = f.read()

    # Discover Stage 1 outputs
    stage1_files = sorted(stage1_path.glob("*.json"))
    if limit:
        stage1_files = stage1_files[:limit]

    # Skip already-processed docs (resumability)
    to_process = []
    for s1_file in stage1_files:
        doc_id = s1_file.stem
        out_file = output_path / f"{doc_id}.json"
        if out_file.exists():
            logger.info(f"[Stage2] {doc_id}: already exists, skipping")
            continue
        to_process.append((doc_id, s1_file))

    if not to_process:
        logger.info("[Stage2] All documents already verified.")
        return [f.stem for f in stage1_files]

    logger.info(f"[Stage2] Verifying {len(to_process)} documents (skipped {len(stage1_files) - len(to_process)} existing)")

    # Build client
    client = GeminiClient(
        model=ver_cfg["model"],
        temperature=ver_cfg["temperature"],
        top_p=ver_cfg["top_p"],
        max_output_tokens=ver_cfg["max_output_tokens"],
        max_retries=pipe_cfg["max_retries"],
        backoff_base=pipe_cfg["retry_backoff_base"],
    )

    def _process_one_doc(doc_id: str, s1_file: Path) -> str:
        # Load Stage 1 output
        with open(s1_file, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)

        # Load source text
        source_file = input_path / f"{doc_id}.txt"
        if not source_file.exists():
            logger.error(f"[Stage2] {doc_id}: source file not found at {source_file}")
            return doc_id
        source_text = source_file.read_text(encoding="utf-8")

        facts = stage1_data.get("facts", [])
        if not facts:
            # Write empty output if Stage 1 produced no facts (or had an error)
            out_file = output_path / f"{doc_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "doc_id": doc_id,
                    "generator_model": stage1_data.get("generator_model"),
                    "verifier_model": client.model,
                    "verified_facts": [],
                    "stage1_error": stage1_data.get("error"),
                }, f, indent=2, ensure_ascii=False)
            return doc_id

        # Verify each fact — one per call (no batching)
        verified_facts = []
        for idx, fact_entry in enumerate(facts):
            result = verify_single_fact(
                source_text=source_text,
                fact=fact_entry["fact"],
                generator_span=fact_entry["supporting_span"],
                client=client,
                verifier_prompt=verifier_prompt,
            )
            verified_facts.append({
                "fact_idx": idx,
                "fact": fact_entry["fact"],
                "generator_supporting_span": fact_entry["supporting_span"],
                "span_mismatch": fact_entry["span_mismatch"],
                "verifier_supporting_span": result["verifier_supporting_span"],
                "verifier_reasoning": result["reasoning"],
                "verifier_verdict": result["verdict"],
                "verifier_error": result["error"],
            })
            logger.debug(
                f"[Stage2] {doc_id} fact {idx}: verdict={result['verdict']}"
            )

        # Write output
        output_data = {
            "doc_id": doc_id,
            "generator_model": stage1_data.get("generator_model"),
            "verifier_model": client.model,
            "verified_facts": verified_facts,
        }
        out_file = output_path / f"{doc_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        n_supported = sum(1 for v in verified_facts if v["verifier_verdict"] == "SUPPORTED")
        logger.info(f"[Stage2] {doc_id}: {n_supported}/{len(verified_facts)} SUPPORTED")
        return doc_id

    processed = []
    max_workers = ver_cfg.get("max_workers", 8)

    # Note: concurrency is at the document level. Each doc processes its facts
    # sequentially (one API call per fact), but multiple docs run in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_doc, doc_id, s1_file): doc_id
            for doc_id, s1_file in to_process
        }
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                processed.append(future.result())
            except Exception as exc:
                logger.error(f"[Stage2] {doc_id}: unhandled error — {exc}")
                processed.append(doc_id)

    all_doc_ids = [f.stem for f in stage1_files]
    logger.info(f"[Stage2] Complete. {len(processed)} new + {len(all_doc_ids) - len(processed)} cached.")
    return all_doc_ids
