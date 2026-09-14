"""
Stage 1 — Verbalize predicted triples into self-contained statements.

Why a separate verbalize step (and not feeding raw triples to the judge)?
A triple like (life_cycle, include, egg) is not a proposition: the predicate
"include" is ambiguous (set inclusion? ingredient?) and the subject/object
strings are often noun fragments. Forcing the judge to *interpret* AND *verify*
in one shot mixes two cognitive operations and adds noise. By verbalizing
first with strict rules ("preserve entity strings verbatim, no new info"), the
judge step receives a clean entailment task: "does the source state X?"

Verbalization is run as an OpenAI batch — one triple per request — for two
reasons:
  * cost: batch is half-priced and we have ~10k triples across 100 docs
  * audit: every (triple -> statement) row is logged, so verbalization
    drift can be inspected post-hoc
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from batch.batch_client import BatchClient, BatchRequest, build_responses_request


VERBALIZE_JOB = "verbalize_batch"

# Keep verbalize cheap — gpt-5-mini is more than enough to restate a triple,
# and using a different (smaller) model than the judge avoids any chance that
# the verbalizer pre-biases the judge's output toward SUPPORTED.
DEFAULT_VERBALIZE_MODEL = "gpt-5-mini"
DEFAULT_VERBALIZE_REASONING = "low"


@dataclass
class TripleRow:
    """One predicted triple together with its provenance."""
    doc_id: str
    triple_idx: int          # index inside the doc's triples_final list
    subject: str
    predicate: str
    object: str

    @property
    def custom_id(self) -> str:
        return f"verb|{self.doc_id}|{self.triple_idx}"


def load_verbalizer_prompt(prompts_dir: Path) -> str:
    return (prompts_dir / "verbalizer.txt").read_text(encoding="utf-8")


def _is_malformed(t: TripleRow) -> bool:
    return not (t.subject.strip() and t.predicate.strip() and t.object.strip())


def template_fallback(t: TripleRow) -> str:
    """Deterministic fallback used only when the LLM fails twice."""
    return f"{t.subject} {t.predicate} {t.object}."


def build_verbalize_requests(
    triples: List[TripleRow],
    prompts_dir: Path,
    model: str = DEFAULT_VERBALIZE_MODEL,
    reasoning_effort: str = DEFAULT_VERBALIZE_REASONING,
) -> Tuple[List[BatchRequest], List[TripleRow]]:
    """
    Build one /v1/responses request per well-formed triple.

    Returns (requests, malformed) — malformed triples are NOT submitted; their
    statements stay None and they will be reported under MALFORMED in the audit
    log (per the plan: "skip triples with empty/null entities, log them").
    """
    template = load_verbalizer_prompt(prompts_dir)
    system = (
        "You are a precise verbalizer. Follow the rules exactly. "
        "Output only the requested sentence."
    )

    requests: List[BatchRequest] = []
    malformed: List[TripleRow] = []
    for t in triples:
        if _is_malformed(t):
            malformed.append(t)
            continue
        user = (
            template
            .replace("{subject}", t.subject)
            .replace("{predicate}", t.predicate)
            .replace("{object}", t.object)
        )
        body = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": 4000,
        }
        requests.append(
            BatchRequest(
                custom_id=t.custom_id,
                method="POST",
                url="/v1/responses",
                body=body,
            )
        )
    return requests, malformed


def parse_statement(content: Optional[str]) -> Optional[str]:
    """
    Extract a single-sentence statement from the verbalizer's output.

    The verbalizer is asked to return raw text, but we defensively strip
    surrounding quotes / markdown / 'Statement:' prefixes that small models
    sometimes add.
    """
    if not content:
        return None
    s = content.strip()
    # Strip code fences
    if s.startswith("```"):
        s = s.split("```", 2)[-1]
        if s.startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Strip leading 'Statement:' label
    for prefix in ("Statement:", "statement:", "STATEMENT:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # Strip outer quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # Collapse multi-line — verbalizer must produce one sentence; keep first line
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    return s or None


_VERBALIZE_SYSTEM = (
    "You are a precise verbalizer. Follow the rules exactly. "
    "Output only the requested sentence."
)


def _user_prompt_for(t: TripleRow, template: str) -> str:
    return (
        template
        .replace("{subject}", t.subject)
        .replace("{predicate}", t.predicate)
        .replace("{object}", t.object)
    )


def run_verbalize_stage(
    triples: List[TripleRow],
    prompts_dir: Path,
    workdir: Path,
    logger: logging.Logger,
    model: str = DEFAULT_VERBALIZE_MODEL,
    reasoning_effort: str = DEFAULT_VERBALIZE_REASONING,
    backend: str = "openai",
    local_max_workers: int = 8,
) -> Tuple[Dict[str, str], List[TripleRow]]:
    """
    Run the verbalize stage end-to-end. Returns:
      statements: dict {custom_id -> statement}    (only successful rows)
      malformed:  list of triples skipped pre-flight

    `backend` selects the execution path:
      * "openai" — submit one OpenAI Batch job (resumable via batch output.jsonl)
      * "local"  — call a self-hosted LLM (LLMClient model_type="local") in
                   parallel; resumable via local_results.jsonl on disk.
    """
    if backend == "local":
        return _run_verbalize_local(
            triples=triples,
            prompts_dir=prompts_dir,
            workdir=workdir,
            logger=logger,
            model=model,
            max_workers=local_max_workers,
        )

    requests, malformed = build_verbalize_requests(
        triples, prompts_dir=prompts_dir, model=model, reasoning_effort=reasoning_effort
    )
    logger.info(
        f"[verbalize] requests={len(requests)} malformed_skipped={len(malformed)}"
    )

    if not requests:
        return {}, malformed

    batch = BatchClient(workdir=workdir, poll_interval_seconds=60, logger=logger)
    results = batch.run_job(
        job_name=VERBALIZE_JOB,
        endpoint="/v1/responses",
        requests=requests,
    )

    statements: Dict[str, str] = {}
    n_failed = 0
    for cid, entry in results.items():
        if not entry.success:
            n_failed += 1
            continue
        stmt = parse_statement(entry.content)
        if stmt is None:
            n_failed += 1
            continue
        statements[cid] = stmt
    logger.info(
        f"[verbalize] parsed_ok={len(statements)} parse_or_api_failed={n_failed}"
    )
    return statements, malformed


def _run_verbalize_local(
    triples: List[TripleRow],
    prompts_dir: Path,
    workdir: Path,
    logger: logging.Logger,
    model: str,
    max_workers: int,
) -> Tuple[Dict[str, str], List[TripleRow]]:
    from .local_runner import LocalJob, run_jobs_local

    template = load_verbalizer_prompt(prompts_dir)

    jobs: List[LocalJob] = []
    malformed: List[TripleRow] = []
    for t in triples:
        if _is_malformed(t):
            malformed.append(t)
            continue
        jobs.append(
            LocalJob(
                custom_id=t.custom_id,
                system=_VERBALIZE_SYSTEM,
                user=_user_prompt_for(t, template),
                mode="text",
                # Qwen3 burns most of its budget inside <think>...</think>
                # before emitting the answer (which strip_thinking removes
                # in llm_client). 512 was too tight; bump so reasoning + the
                # one-sentence answer both fit.
                max_output_tokens=4096,
            )
        )
    logger.info(
        f"[verbalize] requests={len(jobs)} malformed_skipped={len(malformed)} backend=local"
    )

    if not jobs:
        return {}, malformed

    results = run_jobs_local(
        job_name=VERBALIZE_JOB,
        jobs=jobs,
        workdir=workdir,
        logger=logger,
        model=model,
        max_workers=max_workers,
        temperature=0.0,
        max_output_tokens=4096,
    )

    statements: Dict[str, str] = {}
    n_failed = 0
    for cid, row in results.items():
        if not row.get("success"):
            n_failed += 1
            continue
        stmt = parse_statement(row.get("content"))
        if stmt is None:
            n_failed += 1
            continue
        statements[cid] = stmt
    logger.info(
        f"[verbalize] parsed_ok={len(statements)} parse_or_api_failed={n_failed}"
    )
    return statements, malformed
