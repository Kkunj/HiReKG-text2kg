"""
Local-LLM execution backend for verbalize + judge stages.

Mirrors the OpenAI Batch path's contract — given a list of jobs, return a dict
{custom_id -> {"success": bool, "content": str|None, "error": str|None}} — but
runs synchronously against a self-hosted endpoint via the project's existing
LLMClient (model_type="local"). Concurrency is provided by ThreadPoolExecutor;
results are written to a JSONL file as they arrive so an interrupted run
resumes from disk on the next invocation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# our_approach.llm_client lives outside this package; the runner script adds
# our_approach/ to sys.path before importing this module.
_OUR_APPROACH = Path(__file__).resolve().parents[3] / "our_approach"
if str(_OUR_APPROACH) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH))

from llm_client import LLMClient, strip_thinking  # noqa: E402


JobMode = Literal["text", "json"]


@dataclass
class LocalJob:
    custom_id: str
    system: str
    user: str
    mode: JobMode                  # "text" -> raw string; "json" -> stringified JSON
    max_output_tokens: int = 4000


def _cache_path(workdir: Path, job_name: str) -> Path:
    return workdir / job_name / "local_results.jsonl"


def _load_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["custom_id"]] = row
            except Exception:
                continue
    return out


class _AppendingWriter:
    """Thread-safe append-only JSONL writer for incremental result caching."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = threading.Lock()

    def write(self, row: Dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _make_client(
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    logger: logging.Logger,
) -> LLMClient:
    return LLMClient(
        model=model,
        model_type="local",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        logger=logger,
    )


def _run_one(client: LLMClient, job: LocalJob) -> Dict:
    try:
        if job.mode == "json":
            # generate_json already calls _run_request internally, which
            # routes through _run_self_hosted_request → strip_thinking, so
            # the parsed JSON is built from the post-</think> portion.
            obj = client.generate_json(
                system_prompt=job.system,
                user_prompt=job.user,
                max_output_tokens=job.max_output_tokens,
                enable_thinking=False,
            )
            content = json.dumps(obj, ensure_ascii=False)
        else:
            # _run_self_hosted_request already strips <think>...</think>;
            # we strip again defensively so the contract is visible here:
            # downstream parsers (parse_statement) only see the answer text.
            raw = client._run_request(
                system_prompt=job.system,
                user_prompt=job.user,
                response_format=None,
                max_output_tokens=job.max_output_tokens,
                enable_thinking=False,
            )
            content = strip_thinking(raw)
        return {
            "custom_id": job.custom_id,
            "success": True,
            "content": content,
            "error": None,
        }
    except Exception as exc:
        return {
            "custom_id": job.custom_id,
            "success": False,
            "content": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_jobs_local(
    job_name: str,
    jobs: List[LocalJob],
    workdir: Path,
    logger: logging.Logger,
    model: str,
    max_workers: int = 8,
    temperature: float = 0.0,
    max_output_tokens: int = 4000,
    max_retries: int = 3,
    log_every: int = 100,
) -> Dict[str, Dict]:
    """
    Execute `jobs` against a local LLM with ThreadPoolExecutor concurrency.

    Resumable: appends each completed result to {workdir}/{job_name}/local_results.jsonl;
    on restart, jobs whose custom_id is already in that file are skipped.

    Returns {custom_id -> {"success": bool, "content": str|None, "error": str|None}}.
    """
    cache_file = _cache_path(workdir, job_name)
    existing = _load_cache(cache_file)
    todo = [j for j in jobs if j.custom_id not in existing]

    logger.info(
        f"[local:{job_name}] total={len(jobs)} cached={len(existing)} "
        f"todo={len(todo)} model={model} workers={max_workers}"
    )
    if not todo:
        logger.info(f"[local:{job_name}] all results already cached; skipping LLM calls.")
        return existing

    # One client is fine — its underlying requests.post is thread-safe.
    client = _make_client(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        logger=logger,
    )

    writer = _AppendingWriter(cache_file)
    results: Dict[str, Dict] = dict(existing)
    n_done = 0
    n_failed = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_cid = {
                pool.submit(_run_one, client, j): j.custom_id for j in todo
            }
            for fut in as_completed(future_to_cid):
                cid = future_to_cid[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "custom_id": cid,
                        "success": False,
                        "content": None,
                        "error": f"executor_exception: {exc}",
                    }
                writer.write(row)
                results[cid] = row
                n_done += 1
                if not row["success"]:
                    n_failed += 1
                if n_done % log_every == 0 or n_done == len(todo):
                    logger.info(
                        f"[local:{job_name}] progress {n_done}/{len(todo)} "
                        f"(failed_so_far={n_failed})"
                    )
    finally:
        writer.close()

    logger.info(
        f"[local:{job_name}] done. ran={n_done} failed={n_failed} "
        f"total_with_cache={len(results)}"
    )
    return results
