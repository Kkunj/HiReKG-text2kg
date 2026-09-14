"""
Classify atomic facts as single-entity or multiple-entity using GPT-5 Batch API.

Reads all *.json files from a stage3_final directory, submits one batch
request per fact to classify it, then appends "single-entity" and
"multiple-entity" index lists back to each JSON file.

Usage:
    python classify_entity_facts.py                      # full run
    python classify_entity_facts.py --dry-run             # write input.jsonl only, no submission
    python classify_entity_facts.py --model gpt-4o        # use a different model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, OpenAIError, RateLimitError

# Add batch client to path
_BATCH_DIR = Path(__file__).resolve().parent.parent / "our_approach" / "batch"
_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent / "our_approach"
for _p in (_BATCH_DIR, _OUR_APPROACH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from batch_client import BatchClient, BatchResultEntry, build_responses_request  # noqa: E402
from llm_client import LLMClient  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
STAGE3_DIR = Path(__file__).resolve().parent.parent / "datasets" / "scierc" / "atomic_facts" / "stage3_final"
BATCH_WORKDIR = Path(__file__).resolve().parent / "batch_workdir_classify"
DEFAULT_MODEL = "gpt-5"
# GPT-5 uses ~128-192 reasoning tokens internally; 4096 gives very ample room
MAX_OUTPUT_TOKENS = 4096

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert linguistic annotator. Your task is to classify atomic \
facts into one of two categories based on their entity structure.

# CATEGORIES

## SINGLE_ENTITY
A fact whose semantic content is primarily about ONE named entity. The fact \
describes an attribute, property, action, or standalone descriptor of that \
single entity. The fact can be expressed as: <Entity> + <attribute/property/\
action> + <non-entity value such as a number, date, descriptor, category, \
or generic noun>.

Examples:
- "Marie Curie was a physicist."  \
  (Curie is the only named entity; "physicist" is a category, not an entity)
- "The Eiffel Tower is 330 meters tall."  \
  (Tower is the only named entity; "330 meters" is a measurement)
- "World War II began in 1939."  \
  (WWII is the only named entity; "1939" is a date)
- "Python is a programming language."  \
  (Python is the only named entity; "programming language" is a category)
- "The novel was published in 1925."  \
  (The novel — even if referenced obliquely — is the only entity; date is non-entity)

## MULTI_ENTITY
A fact whose semantic content requires AT LEAST TWO distinct named entities \
to be expressed. Removing either named entity would make the fact incomplete \
or change its core meaning.

Examples:
- "Paris is the capital of France."  \
  (Requires both "Paris" and "France")
- "Marie Curie won the Nobel Prize in Physics."  \
  (Requires "Marie Curie" and "Nobel Prize in Physics" as the awarded entity)
- "Einstein worked at Princeton University."  \
  (Requires "Einstein" and "Princeton University")
- "The Beatles released Abbey Road."  \
  (Requires "The Beatles" and "Abbey Road")
- "Python was created by Guido van Rossum."  \
  (Requires "Python" and "Guido van Rossum")

# CRITICAL RULES

1. A NAMED ENTITY is a proper noun referring to a specific, identifiable \
real-world or fictional entity: a person, organization, location, work \
(book/film/song), product, named event, scientific concept with a proper \
name, etc.

2. The following are NOT named entities (they are generic descriptors, \
even if they appear in the fact):
   - Common nouns: "physicist", "city", "company", "language"
   - Categories: "programming language", "novel", "award"
   - Dates and times: "1939", "May 2023", "the 19th century"
   - Numbers and measurements: "330 meters", "5 children", "42%"
   - Generic descriptors: "tall", "famous", "ancient"
   - Pronouns referring to entities ("he", "it", "they")

3. If a category-level noun (like "Nobel Prize") is presented as a SPECIFIC \
named award/entity in the fact, treat it as a named entity. Use context to \
decide. For example:
   - "She won an award" -> "award" is generic, NOT a named entity.
   - "She won the Nobel Prize" -> "Nobel Prize" is specific, IS a named entity.

4. If the fact contains a pronoun (he, she, it, they, this) that clearly \
refers to a named entity from prior context, COUNT that pronoun as referring \
to the named entity it represents.

5. If the fact is ambiguous, malformed, or you cannot confidently classify \
it, use AMBIGUOUS rather than guessing.

# DECISION HEURISTIC

Ask yourself: "If I remove all but one named entity from this fact, does the \
fact still convey its core meaning?"
- YES -> SINGLE_ENTITY
- NO -> MULTI_ENTITY
- Unclear -> AMBIGUOUS

# OUTPUT FORMAT

Return ONLY a valid JSON object with the following structure. Do not \
include any text before or after the JSON.

{
  "named_entities": ["<entity1>", "<entity2>", ...],
  "category": "SINGLE_ENTITY" | "MULTI_ENTITY" | "AMBIGUOUS",
  "reasoning": "<one-sentence explanation of why this category was chosen>"
}\
"""

USER_PROMPT_TEMPLATE = '# FACT TO CLASSIFY\n\n"{fact}"'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class DocEntry:
    """A loaded document together with its source path on disk."""
    path: Path
    data: Dict[str, Any]


def load_all_docs(stage3_dir: Path) -> List[DocEntry]:
    """Load all *.json files from stage3_dir, sorted by filename."""
    entries: List[DocEntry] = []
    for p in sorted(stage3_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            entries.append(DocEntry(path=p, data=json.load(f)))
    return entries


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("FactClassifier")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(BATCH_WORKDIR / "classify.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)

    return logger


def sync_classify_fact(
    client: OpenAI,
    model: str,
    fact: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Classify a single fact synchronously using the OpenAI responses API.
    Handles GPT-5 quirks: no temperature param, uses text.format for JSON mode.
    Returns parsed JSON dict or None on failure.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            request_payload: Dict[str, Any] = {
                "model": model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(fact=fact)},
                ],
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "text": {"format": {"type": "json_object"}},
            }
            response = client.responses.create(**request_payload)
            # Extract text from response (GPT-5 output includes reasoning items
            # with content=None, so we must guard against that)
            raw = getattr(response, "output_text", None) or ""
            if not raw:
                parts = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", None) or []:
                        text = getattr(content, "text", None)
                        if text:
                            parts.append(text)
                raw = "".join(parts).strip()
            if not raw:
                return None
            cleaned = LLMClient._clean_json_response(raw)
            return json.loads(cleaned)
        except (RateLimitError, APIConnectionError):
            attempt += 1
            time.sleep(2 ** attempt)
        except (OpenAIError, json.JSONDecodeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Classify atomic facts via GPT batch API")
    parser.add_argument("--dry-run", action="store_true", help="Write input.jsonl only, do not submit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--stage3-dir", default=str(STAGE3_DIR), help="Path to stage3_final directory")
    parser.add_argument("--job-name", default="classify_entity_facts", help="Batch job name (use distinct names for different datasets)")
    args = parser.parse_args()

    stage3_dir = Path(args.stage3_dir)
    BATCH_WORKDIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()

    # 1. Load all documents
    entries = load_all_docs(stage3_dir)
    total_facts = sum(len(e.data["facts"]) for e in entries)
    logger.info(f"Loaded {len(entries)} documents with {total_facts} total facts")

    # 2. Build batch requests — one per fact
    requests = []
    id_map: Dict[str, tuple] = {}

    for doc_idx, entry in enumerate(entries):
        doc_id = entry.data["doc_id"]
        for fact_idx, fact in enumerate(entry.data["facts"]):
            custom_id = f"classify|{doc_id}|{fact_idx}"
            req = build_responses_request(
                custom_id=custom_id,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT_TEMPLATE.format(fact=fact),
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                json_mode=True,
            )
            # GPT-5 does not support the temperature parameter
            req.body.pop("temperature", None)
            requests.append(req)
            id_map[custom_id] = (doc_idx, fact_idx)

    logger.info(f"Built {len(requests)} batch requests")

    # 3. Initialize batch client and run
    batch_client = BatchClient(
        workdir=BATCH_WORKDIR,
        poll_interval_seconds=60,
        logger=logger,
    )

    job_name = args.job_name
    endpoint = "/v1/responses"

    if args.dry_run:
        batch_client.write_input_file(job_name, requests)
        logger.info(f"Dry run complete. Input file written to {BATCH_WORKDIR / job_name / 'input.jsonl'}")
        logger.info(f"Total requests: {len(requests)}")
        return

    results: Dict[str, BatchResultEntry] = batch_client.run_job(
        job_name=job_name,
        endpoint=endpoint,
        requests=requests,
    )

    # 4. Parse results and build classification indices per doc
    CATEGORY_MAP = {
        "SINGLE_ENTITY": "single-entity",
        "MULTI_ENTITY": "multiple-entity",
    }
    VALID_CATEGORIES = set(CATEGORY_MAP.keys()) | {"AMBIGUOUS"}

    doc_classifications: Dict[int, Dict[str, List[int]]] = {
        i: {"single-entity": [], "multiple-entity": [], "ambiguous": []}
        for i in range(len(entries))
    }

    success_count = 0
    fail_count = 0
    ambiguous_count = 0
    sync_retry_count = 0

    # Direct OpenAI client for sync retries (avoids LLMClient quirks with GPT-5)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _extract_category(parsed: Dict[str, Any]) -> Optional[str]:
        raw = parsed.get("category", "").strip().upper()
        raw = raw.replace("-", "_").replace(" ", "_")
        if raw in VALID_CATEGORIES:
            return raw
        return None

    for custom_id, (doc_idx, fact_idx) in id_map.items():
        entry = results.get(custom_id)
        category = None

        # Try to parse batch result
        if entry and entry.success and entry.content:
            try:
                cleaned = LLMClient._clean_json_response(entry.content)
                parsed = json.loads(cleaned)
                category = _extract_category(parsed)
            except (json.JSONDecodeError, AttributeError):
                category = None

        # Sync retry if batch failed or returned incomplete/unparseable
        if category is None:
            sync_retry_count += 1
            logger.info(f"Sync retry for {custom_id}")
            try:
                fact = entries[doc_idx].data["facts"][fact_idx]
                parsed = sync_classify_fact(openai_client, args.model, fact)
                if parsed:
                    category = _extract_category(parsed)
            except Exception as exc:
                logger.error(f"Sync retry failed for {custom_id}: {exc}")
                category = None

        # Assign classification
        if category in ("SINGLE_ENTITY", "MULTI_ENTITY"):
            output_key = CATEGORY_MAP[category]
            doc_classifications[doc_idx][output_key].append(fact_idx)
            success_count += 1
        elif category == "AMBIGUOUS":
            doc_classifications[doc_idx]["ambiguous"].append(fact_idx)
            ambiguous_count += 1
        else:
            logger.warning(f"Could not classify {custom_id}, marking as ambiguous")
            doc_classifications[doc_idx]["ambiguous"].append(fact_idx)
            fail_count += 1

    logger.info(
        f"Classification complete: {success_count} succeeded, "
        f"{ambiguous_count} ambiguous, {fail_count} failed, "
        f"{sync_retry_count} sync retries"
    )

    # 5. Sort indices and write back to JSON files
    for doc_idx, entry in enumerate(entries):
        cls = doc_classifications[doc_idx]
        entry.data["single-entity"] = sorted(cls["single-entity"])
        entry.data["multiple-entity"] = sorted(cls["multiple-entity"])
        if cls["ambiguous"]:
            entry.data["ambiguous"] = sorted(cls["ambiguous"])

        with open(entry.path, "w", encoding="utf-8") as f:
            json.dump(entry.data, f, indent=2, ensure_ascii=False)

    logger.info(f"Updated {len(entries)} JSON files in {stage3_dir}")

    # Print summary
    total_single = sum(len(doc_classifications[i]["single-entity"]) for i in range(len(entries)))
    total_multi = sum(len(doc_classifications[i]["multiple-entity"]) for i in range(len(entries)))
    total_ambiguous = sum(len(doc_classifications[i]["ambiguous"]) for i in range(len(entries)))
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total single-entity facts: {total_single}")
    logger.info(f"Total multiple-entity facts: {total_multi}")
    logger.info(f"Total ambiguous facts: {total_ambiguous}")
    logger.info(f"Total facts classified: {total_single + total_multi + total_ambiguous}")


if __name__ == "__main__":
    main()
