"""
Stage 3 — Entailment judging.

For each (context, statement) pair we issue ONE batch /v1/responses request
to gpt-5 with reasoning_effort=high. The judge returns a binary verdict:
SUPPORTED or NOT_SUPPORTED. Contradictions and silence are merged into the
single NOT_SUPPORTED bucket because they're equivalent for the precision
question we're answering ("did the extractor invent this?").

The judge must return strict JSON with keys (supporting_span, reasoning,
verdict) — *in that order*. Putting span and reasoning before the verdict
forces the judge to ground its decision in the text rather than rationalize
a guessed label.

Why one statement per call (no batching multiple statements per prompt):
  * avoids order effects: judging statement 7 should not depend on what was
    judged for statements 1-6 in the same prompt.
  * avoids context contamination: the judge's reasoning for one statement
    could implicitly seed a verdict on another in the same call.
  * the OpenAI Batch API gives us the throughput we'd otherwise want from
    in-prompt batching, at half the cost.

Why a SEPARATE judge model (gpt-5) from the verbalizer (gpt-5-mini):
  * the verbalizer can only restate; it cannot bias the judge toward
    SUPPORTED, because the judge has access to the source text and is
    instructed to ignore world knowledge.

Failure handling: any request whose JSON cannot be parsed after one batch
attempt is retried synchronously once; if still unparseable it is logged as
JUDGE_ERROR and EXCLUDED from the faithfulness denominator (per the plan,
"don't silently dump them into NOT_STATED").
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from batch.batch_client import BatchClient, BatchRequest


JUDGE_JOB = "judge_batch"
DEFAULT_JUDGE_MODEL = "gpt-5"
DEFAULT_JUDGE_REASONING = "high"

VALID_VERDICTS = {"SUPPORTED", "NOT_SUPPORTED"}

# Legacy three-class labels we still tolerate from older judge runs / prompts.
# We collapse them to the binary scheme so a partial cache from before the
# binary switch can still be reused without re-running the judge.
LEGACY_VERDICT_MAP = {
    "CONTRADICTED": "NOT_SUPPORTED",
    "NOT_STATED": "NOT_SUPPORTED",
}


@dataclass
class JudgeOutcome:
    """Parsed judge result for a single (statement, context) pair."""
    custom_id: str
    supporting_span: Optional[str]
    reasoning: Optional[str]
    verdict: Optional[str]      # SUPPORTED | NOT_SUPPORTED, or None on parse fail
    raw_content: Optional[str] = None
    error: Optional[str] = None  # set iff parse / API failed (JUDGE_ERROR)


def load_judge_prompt(prompts_dir: Path) -> str:
    return (prompts_dir / "judge.txt").read_text(encoding="utf-8")


def build_judge_request(
    custom_id: str,
    context: str,
    statement: str,
    prompt_template: str,
    model: str = DEFAULT_JUDGE_MODEL,
    reasoning_effort: str = DEFAULT_JUDGE_REASONING,
    max_output_tokens: int = 16000,
) -> BatchRequest:
    user = (
        prompt_template
        .replace("{context}", context.strip())
        .replace("{statement}", statement.strip())
    )
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a strict, evidence-grounded entailment judge. "
                    "Output only the requested JSON object."
                ),
            },
            {"role": "user", "content": user},
        ],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    return BatchRequest(
        custom_id=custom_id, method="POST", url="/v1/responses", body=body
    )


def _parse_payload(content: Optional[str]) -> JudgeOutcome:
    """Parse the judge's JSON output into a JudgeOutcome."""
    if not content:
        return JudgeOutcome(
            custom_id="", supporting_span=None, reasoning=None,
            verdict=None, raw_content=content, error="empty_content",
        )
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[-1]
        if s.startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        obj = json.loads(s)
    except Exception as exc:
        return JudgeOutcome(
            custom_id="", supporting_span=None, reasoning=None,
            verdict=None, raw_content=content, error=f"json_parse: {exc}",
        )

    verdict = obj.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().upper()
    if verdict in LEGACY_VERDICT_MAP:
        verdict = LEGACY_VERDICT_MAP[verdict]
    if verdict not in VALID_VERDICTS:
        return JudgeOutcome(
            custom_id="", supporting_span=None, reasoning=None,
            verdict=None, raw_content=content, error=f"bad_verdict: {verdict}",
        )

    span = obj.get("supporting_span")
    if isinstance(span, str):
        span = span.strip() or None
    elif span is not None:
        span = str(span)

    reasoning = obj.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip() or None

    return JudgeOutcome(
        custom_id="",
        supporting_span=span,
        reasoning=reasoning,
        verdict=verdict,
        raw_content=content,
    )


def _sync_retry_judge(
    context: str,
    statement: str,
    prompt_template: str,
    model: str = DEFAULT_JUDGE_MODEL,
    reasoning_effort: str = DEFAULT_JUDGE_REASONING,
) -> JudgeOutcome:
    """Synchronous one-shot retry for batch requests that failed or returned junk."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user = (
        prompt_template
        .replace("{context}", context.strip())
        .replace("{statement}", statement.strip())
    )
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, evidence-grounded entailment judge. "
                        "Output only the requested JSON object."
                    ),
                },
                {"role": "user", "content": user},
            ],
            reasoning={"effort": reasoning_effort},
            max_output_tokens=16000,
            text={"format": {"type": "json_object"}},
        )
        parts = []
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                if getattr(c, "text", None):
                    parts.append(c.text)
        return _parse_payload("".join(parts))
    except Exception as exc:
        return JudgeOutcome(
            custom_id="", supporting_span=None, reasoning=None,
            verdict=None, raw_content=None, error=f"sync_retry_exception: {exc}",
        )


_JUDGE_SYSTEM = (
    "You are a strict, evidence-grounded entailment judge. "
    "Output only the requested JSON object."
)


def run_judge_stage(
    statement_to_context: Dict[str, str],   # {custom_id -> context_text}
    statement_to_text: Dict[str, str],      # {custom_id -> statement}
    prompts_dir: Path,
    workdir: Path,
    logger: logging.Logger,
    model: str = DEFAULT_JUDGE_MODEL,
    reasoning_effort: str = DEFAULT_JUDGE_REASONING,
    backend: str = "openai",
    local_max_workers: int = 8,
) -> Dict[str, JudgeOutcome]:
    """
    Run the judge stage end-to-end and return parsed outcomes per custom_id.
    Custom_id format: "judge|<doc_id>|<triple_idx>" — derived from each
    statement's verbalize id by replacing the "verb|" prefix.

    `backend` selects the execution path: "openai" (Batch API) or "local"
    (self-hosted LLM via LLMClient).
    """
    if backend == "local":
        return _run_judge_local(
            statement_to_context=statement_to_context,
            statement_to_text=statement_to_text,
            prompts_dir=prompts_dir,
            workdir=workdir,
            logger=logger,
            model=model,
            max_workers=local_max_workers,
        )

    template = load_judge_prompt(prompts_dir)

    requests: List[BatchRequest] = []
    judge_id_for: Dict[str, str] = {}        # verb_cid -> judge_cid (for reverse lookup)
    for verb_cid, ctx in statement_to_context.items():
        statement = statement_to_text[verb_cid]
        judge_cid = "judge|" + verb_cid.split("|", 1)[1]
        judge_id_for[verb_cid] = judge_cid
        requests.append(
            build_judge_request(
                custom_id=judge_cid,
                context=ctx,
                statement=statement,
                prompt_template=template,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        )
    logger.info(f"[judge] requests={len(requests)} model={model}")

    if not requests:
        return {}

    batch = BatchClient(workdir=workdir, poll_interval_seconds=60, logger=logger)
    results = batch.run_job(
        job_name=JUDGE_JOB,
        endpoint="/v1/responses",
        requests=requests,
    )

    outcomes: Dict[str, JudgeOutcome] = {}
    n_retried = 0
    n_retry_failed = 0
    for verb_cid, ctx in statement_to_context.items():
        judge_cid = judge_id_for[verb_cid]
        entry = results.get(judge_cid)
        outcome = JudgeOutcome(
            custom_id=judge_cid, supporting_span=None, reasoning=None, verdict=None,
        )
        if entry is None:
            outcome.error = "no_batch_result"
        elif not entry.success:
            outcome.error = f"batch_failure: {entry.error}"
        else:
            outcome = _parse_payload(entry.content or "")
            outcome.custom_id = judge_cid

        if outcome.verdict is None:
            n_retried += 1
            statement = statement_to_text[verb_cid]
            retry = _sync_retry_judge(
                context=ctx, statement=statement, prompt_template=template,
                model=model, reasoning_effort=reasoning_effort,
            )
            retry.custom_id = judge_cid
            if retry.verdict is not None:
                outcome = retry
            else:
                n_retry_failed += 1
                # Preserve the most informative error message
                outcome.error = (
                    outcome.error or ""
                ) + f" | retry_error: {retry.error}"

        outcomes[judge_cid] = outcome

    n_supported = sum(1 for o in outcomes.values() if o.verdict == "SUPPORTED")
    n_not_supported = sum(1 for o in outcomes.values() if o.verdict == "NOT_SUPPORTED")
    n_errored = sum(1 for o in outcomes.values() if o.verdict is None)
    logger.info(
        f"[judge] verdicts: SUPPORTED={n_supported} NOT_SUPPORTED={n_not_supported} "
        f"JUDGE_ERROR={n_errored} "
        f"(sync_retries={n_retried}, sync_retry_failures={n_retry_failed})"
    )
    return outcomes


def _run_judge_local(
    statement_to_context: Dict[str, str],
    statement_to_text: Dict[str, str],
    prompts_dir: Path,
    workdir: Path,
    logger: logging.Logger,
    model: str,
    max_workers: int,
) -> Dict[str, JudgeOutcome]:
    from .local_runner import LocalJob, run_jobs_local

    template = load_judge_prompt(prompts_dir)

    jobs: List[LocalJob] = []
    judge_id_for: Dict[str, str] = {}
    for verb_cid, ctx in statement_to_context.items():
        statement = statement_to_text[verb_cid]
        judge_cid = "judge|" + verb_cid.split("|", 1)[1]
        judge_id_for[verb_cid] = judge_cid
        user = (
            template
            .replace("{context}", ctx.strip())
            .replace("{statement}", statement.strip())
        )
        jobs.append(
            LocalJob(
                custom_id=judge_cid,
                system=_JUDGE_SYSTEM,
                user=user,
                mode="json",
                max_output_tokens=4000,
            )
        )
    logger.info(f"[judge] requests={len(jobs)} model={model} backend=local")

    if not jobs:
        return {}

    results = run_jobs_local(
        job_name=JUDGE_JOB,
        jobs=jobs,
        workdir=workdir,
        logger=logger,
        model=model,
        max_workers=max_workers,
        temperature=0.0,
        max_output_tokens=4000,
    )

    outcomes: Dict[str, JudgeOutcome] = {}
    for verb_cid, _ctx in statement_to_context.items():
        judge_cid = judge_id_for[verb_cid]
        row = results.get(judge_cid)
        if row is None or not row.get("success"):
            err = (row or {}).get("error") or "no_result"
            outcomes[judge_cid] = JudgeOutcome(
                custom_id=judge_cid, supporting_span=None,
                reasoning=None, verdict=None, error=err,
            )
            continue
        outcome = _parse_payload(row.get("content") or "")
        outcome.custom_id = judge_cid
        outcomes[judge_cid] = outcome

    n_supported = sum(1 for o in outcomes.values() if o.verdict == "SUPPORTED")
    n_not_supported = sum(1 for o in outcomes.values() if o.verdict == "NOT_SUPPORTED")
    n_errored = sum(1 for o in outcomes.values() if o.verdict is None)
    logger.info(
        f"[judge] verdicts: SUPPORTED={n_supported} NOT_SUPPORTED={n_not_supported} "
        f"JUDGE_ERROR={n_errored} (backend=local)"
    )
    return outcomes
