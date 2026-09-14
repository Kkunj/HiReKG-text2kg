"""
Classify MINE dataset facts as single-entity or multiple-entity using GPT-5 Batch API.

The MINE dataset stores facts in a single answers.json with structure:
  { "features": [...], "rows": [ { "row_idx": N, "row": { "answers": [{"answer": "..."}] } } ] }

This script classifies each fact, then appends "single-entity" and "multiple-entity"
index lists to each row in-place.

Usage:
    python classify_mine_facts.py                        # full run
    python classify_mine_facts.py --dry-run               # write input.jsonl only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, OpenAIError, RateLimitError

_BATCH_DIR = Path(__file__).resolve().parent.parent / "our_approach" / "batch"
_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent / "our_approach"
for _p in (_BATCH_DIR, _OUR_APPROACH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from batch_client import BatchClient, BatchResultEntry, build_responses_request  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from classify_entity_facts import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    MAX_OUTPUT_TOKENS,
    sync_classify_fact,
)

load_dotenv()

# ---------------------------------------------------------------------------
ANSWERS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "MINE" / "answers.json"
BATCH_WORKDIR = Path(__file__).resolve().parent / "batch_workdir_classify"
DEFAULT_MODEL = "gpt-5"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("MineClassifier")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(BATCH_WORKDIR / "classify_mine.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify MINE dataset facts via GPT batch API")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--answers-path", default=str(ANSWERS_PATH))
    args = parser.parse_args()

    answers_path = Path(args.answers_path)
    BATCH_WORKDIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()

    # 1. Load
    with open(answers_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]
    total_facts = sum(len(r["row"]["answers"]) for r in rows)
    logger.info(f"Loaded {len(rows)} rows with {total_facts} total facts from {answers_path}")

    # 2. Build batch requests
    requests = []
    id_map: Dict[str, tuple] = {}  # custom_id -> (row_idx_in_list, fact_idx)

    for row_list_idx, row_entry in enumerate(rows):
        row_idx = row_entry["row_idx"]
        answers = row_entry["row"]["answers"]
        for fact_idx, ans in enumerate(answers):
            fact = ans["answer"]
            custom_id = f"classify|row{row_idx}|{fact_idx}"
            req = build_responses_request(
                custom_id=custom_id,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT_TEMPLATE.format(fact=fact),
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                json_mode=True,
            )
            req.body.pop("temperature", None)
            requests.append(req)
            id_map[custom_id] = (row_list_idx, fact_idx)

    logger.info(f"Built {len(requests)} batch requests")

    # 3. Run batch
    batch_client = BatchClient(workdir=BATCH_WORKDIR, poll_interval_seconds=60, logger=logger)
    job_name = "classify_mine"
    endpoint = "/v1/responses"

    if args.dry_run:
        batch_client.write_input_file(job_name, requests)
        logger.info(f"Dry run complete. Input file: {BATCH_WORKDIR / job_name / 'input.jsonl'}")
        logger.info(f"Total requests: {len(requests)}")
        return

    results: Dict[str, BatchResultEntry] = batch_client.run_job(
        job_name=job_name, endpoint=endpoint, requests=requests,
    )

    # 4. Parse results
    CATEGORY_MAP = {"SINGLE_ENTITY": "single-entity", "MULTI_ENTITY": "multiple-entity"}
    VALID_CATEGORIES = set(CATEGORY_MAP.keys()) | {"AMBIGUOUS"}

    row_classifications: Dict[int, Dict[str, List[int]]] = {
        i: {"single-entity": [], "multiple-entity": [], "ambiguous": []}
        for i in range(len(rows))
    }

    success_count = 0
    fail_count = 0
    ambiguous_count = 0
    sync_retry_count = 0

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _extract_category(parsed: Dict[str, Any]) -> Optional[str]:
        raw = parsed.get("category", "").strip().upper().replace("-", "_").replace(" ", "_")
        return raw if raw in VALID_CATEGORIES else None

    for custom_id, (row_list_idx, fact_idx) in id_map.items():
        entry = results.get(custom_id)
        category = None

        if entry and entry.success and entry.content:
            try:
                cleaned = LLMClient._clean_json_response(entry.content)
                parsed = json.loads(cleaned)
                category = _extract_category(parsed)
            except (json.JSONDecodeError, AttributeError):
                category = None

        if category is None:
            sync_retry_count += 1
            logger.info(f"Sync retry for {custom_id}")
            try:
                fact = rows[row_list_idx]["row"]["answers"][fact_idx]["answer"]
                parsed = sync_classify_fact(openai_client, args.model, fact)
                if parsed:
                    category = _extract_category(parsed)
            except Exception as exc:
                logger.error(f"Sync retry failed for {custom_id}: {exc}")

        if category in ("SINGLE_ENTITY", "MULTI_ENTITY"):
            row_classifications[row_list_idx][CATEGORY_MAP[category]].append(fact_idx)
            success_count += 1
        elif category == "AMBIGUOUS":
            row_classifications[row_list_idx]["ambiguous"].append(fact_idx)
            ambiguous_count += 1
        else:
            logger.warning(f"Could not classify {custom_id}, marking as ambiguous")
            row_classifications[row_list_idx]["ambiguous"].append(fact_idx)
            fail_count += 1

    logger.info(
        f"Classification complete: {success_count} succeeded, "
        f"{ambiguous_count} ambiguous, {fail_count} failed, "
        f"{sync_retry_count} sync retries"
    )

    # 5. Write results back into answers.json
    for row_list_idx, row_entry in enumerate(rows):
        cls = row_classifications[row_list_idx]
        row_entry["row"]["single-entity"] = sorted(cls["single-entity"])
        row_entry["row"]["multiple-entity"] = sorted(cls["multiple-entity"])
        if cls["ambiguous"]:
            row_entry["row"]["ambiguous"] = sorted(cls["ambiguous"])

    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.info(f"Updated {answers_path}")

    # Summary
    total_single = sum(len(row_classifications[i]["single-entity"]) for i in range(len(rows)))
    total_multi = sum(len(row_classifications[i]["multiple-entity"]) for i in range(len(rows)))
    total_ambiguous = sum(len(row_classifications[i]["ambiguous"]) for i in range(len(rows)))
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total single-entity facts: {total_single}")
    logger.info(f"Total multiple-entity facts: {total_multi}")
    logger.info(f"Total ambiguous facts: {total_ambiguous}")
    logger.info(f"Total facts classified: {total_single + total_multi + total_ambiguous}")


if __name__ == "__main__":
    main()
