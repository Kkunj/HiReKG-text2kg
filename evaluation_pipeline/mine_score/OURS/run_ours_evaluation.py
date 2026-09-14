"""
Evaluate the KGs we generated against the MINE ground-truth answers.

Pipeline — 3 stages, each resumable:

    STAGE 1 (local, fast):  retrieve
        For each of the 100 docs:
            - load our KG → convert to kg-gen dict
            - build nx graph + MiniLM node embeddings
            - for each GT answer: kggen.retrieve(answer, ...) → context text
        Save all (context, correct_answer) pairs to
            OURS/batch_workdir/<run_tag>/retrieved.json

    STAGE 2 (judge):  evaluates each (context, correct_answer) pair.
        - BACKEND="openai" → OpenAI Batch API (gpt-5/high, /v1/responses).
        - BACKEND="local"  → self-hosted LLM via our_approach.llm_client
                             (model_type="local"), threaded synchronous calls,
                             results streamed to judge_local_results.jsonl
                             for resumability.

    STAGE 3 (local, fast):  assemble
        Per-doc results_<filtered_idx>.json + summary_ours.json.

Pick the experiment folder + backend in the MANUAL CONFIG block below.

Usage:
    python run_ours_evaluation.py retrieve
    python run_ours_evaluation.py submit
    python run_ours_evaluation.py assemble
    python run_ours_evaluation.py all                 # end-to-end
    python run_ours_evaluation.py all --limit 2       # smoke test: first 2 docs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing our_approach/batch/batch_client.py, llm_client.py, and the
# sibling OURS/ package whether invoked as `python -m OURS.run_ours_evaluation`
# or as `python run_ours_evaluation.py`.
_THIS_DIR = Path(__file__).resolve().parent
_MINE_SCORE_DIR = _THIS_DIR.parent               # evaluation_pipeline/mine_score/
_REPO_ROOT = _THIS_DIR.parent.parent.parent      # graph_rag/
_OUR_APPROACH_DIR = _REPO_ROOT / "our_approach"
for p in (str(_MINE_SCORE_DIR), str(_OUR_APPROACH_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from batch.batch_client import (  # noqa: E402
    BatchClient,
    BatchRequest,
)
from OURS.kg_adapter import load_kg_for_doc  # noqa: E402


# =============================================================================
# MANUAL CONFIG — edit these to choose what to evaluate.
# =============================================================================
# Experiment folder under our_approach/experiments/.
EXPERIMENT_FOLDER = "our_mine_qwen3_8b"

# "openai" → gpt-5/high via OpenAI Batch API.
# "local"  → self-hosted LLM via LLMClient(model_type="local").
BACKEND = "local"

# Used only when BACKEND == "local".
LOCAL_MODEL = "qwen3-14b"
LOCAL_MAX_WORKERS = 4
LOCAL_LLM_URL_OVERRIDE: Optional[str] = "http://<LOCAL_LLM_HOST>:8003/v1/chat/completions"
# =============================================================================


# -----------------------------------------------------------------------------
# Paths (derived from MANUAL CONFIG)
# -----------------------------------------------------------------------------
MINE_SCORE_DIR = _THIS_DIR.parent                           # evaluation_pipeline/mine_score
DATA_DIR = MINE_SCORE_DIR / "data"
ESSAYS_PATH = DATA_DIR / "essays_filtered.json"
ANSWERS_PATH = DATA_DIR / "answers_filtered.json"
INDEX_MAP_PATH = DATA_DIR / "filtered_index_to_mine_doc.json"

EXPERIMENTS_ROOT = _OUR_APPROACH_DIR / "experiments" / EXPERIMENT_FOLDER

# Tag the run so OpenAI vs local artifacts (and different KG sources) don't collide.
RUN_TAG = f"{EXPERIMENT_FOLDER}__{BACKEND}"

RESULTS_DIR = MINE_SCORE_DIR / "results" / f"OURS_{RUN_TAG}"
WORKDIR = _THIS_DIR / "batch_workdir" / RUN_TAG
RETRIEVED_PATH = WORKDIR / "retrieved.json"
JUDGE_BATCH_JOB = "judge_batch"           # OpenAI batch path
LOCAL_JUDGE_JSONL = WORKDIR / "judge_local_results.jsonl"


# -----------------------------------------------------------------------------
# Judge prompts
# -----------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator. Determine whether the provided context contains "
    "the information stated in the given correct answer.\n\n"
    "Think step-by-step internally, then respond with valid JSON of the form:\n"
    '{"reasoning": "<brief reasoning>", "evaluation": 0 | 1}\n\n'
    "Rules:\n"
    "- evaluation = 1  if the context contains (supports) the correct answer.\n"
    "- evaluation = 0  otherwise.\n"
    "- Evaluate on information content, not exact wording. Paraphrases are fine.\n"
    "- If the context is empty or irrelevant, return 0.\n"
    "- Output ONLY the JSON object — no code fences, no extra keys."
)


def build_judge_user_prompt(context: str, correct_answer: str) -> str:
    return (
        f"Context:\n---\n{context.strip()}\n---\n\n"
        f"Correct answer:\n---\n{correct_answer.strip()}\n---\n\n"
        "Does the context contain the information stated in the correct answer? "
        "Respond with JSON."
    )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _make_logger() -> logging.Logger:
    logger = logging.getLogger("OURS_eval")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        WORKDIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    logger.addHandler(ch)
    return logger


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_filtered_data() -> Tuple[List[Dict], List[Dict], List[str]]:
    essays = json.loads(ESSAYS_PATH.read_text(encoding="utf-8"))
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    idx_map = json.loads(INDEX_MAP_PATH.read_text(encoding="utf-8"))
    doc_ids: List[str] = idx_map["index_to_mine_doc"]
    assert len(essays) == len(answers) == len(doc_ids), "filtered files are misaligned"
    return essays, answers, doc_ids


def extract_queries(answer_group: Dict) -> List[str]:
    """answers_filtered.json rows look like {"answers": [{"answer": "..."}, ...]}"""
    return [a["answer"] for a in answer_group.get("answers", []) if a.get("answer")]


# -----------------------------------------------------------------------------
# STAGE 1 — retrieve (local kg-gen retrieval)
# -----------------------------------------------------------------------------
def _retrieve_single_doc(args) -> Tuple[int, str, List[Dict], Optional[str]]:
    filtered_idx, doc_id, queries = args
    try:
        from kg_gen.kg_gen import KGGen

        kg_dict = load_kg_for_doc(EXPERIMENTS_ROOT, doc_id)
        if not kg_dict["relations"]:
            return (filtered_idx, doc_id, [], "empty KG (no relations)")

        kggen = KGGen(
            retrieval_model="all-MiniLM-L6-v2",
            model="openai/gpt-5",           # unused on retrieve path, but required by init
            api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            temperature=1.0,
            max_tokens=16000,
        )
        graph = kggen.from_dict(kg_dict)
        nxGraph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nxGraph)

        items: List[Dict] = []
        for qi, q in enumerate(queries):
            *_, context_text = kggen.retrieve(q, node_embeddings, nxGraph)
            items.append(
                {
                    "query_idx": qi,
                    "correct_answer": q,
                    "retrieved_context": context_text,
                }
            )
        return (filtered_idx, doc_id, items, None)
    except Exception as exc:
        return (filtered_idx, doc_id, [], f"{type(exc).__name__}: {exc}")


def stage_retrieve(
    logger: logging.Logger,
    limit: Optional[int] = None,
    max_workers: int = 4,
) -> List[Dict]:
    if RETRIEVED_PATH.exists():
        logger.info(f"[retrieve] Using cached {RETRIEVED_PATH}")
        return json.loads(RETRIEVED_PATH.read_text(encoding="utf-8"))

    _, answers, doc_ids = load_filtered_data()
    if limit is not None:
        doc_ids = doc_ids[:limit]
        answers = answers[:limit]

    tasks = []
    for i, (ans_group, doc_id) in enumerate(zip(answers, doc_ids)):
        queries = extract_queries(ans_group)
        tasks.append((i, doc_id, queries))

    logger.info(f"[retrieve] Retrieving contexts for {len(tasks)} docs (max_workers={max_workers})")

    out: List[Dict] = [None] * len(tasks)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_retrieve_single_doc, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            filtered_idx, doc_id, items, err = fut.result()
            done += 1
            if err:
                logger.warning(f"[retrieve] [{done}/{len(tasks)}] {doc_id}: {err}")
            else:
                logger.info(
                    f"[retrieve] [{done}/{len(tasks)}] {doc_id}: {len(items)} queries retrieved"
                )
            out[filtered_idx] = {
                "filtered_idx": filtered_idx,
                "doc_id": doc_id,
                "items": items,
                "retrieval_error": err,
            }

    WORKDIR.mkdir(parents=True, exist_ok=True)
    RETRIEVED_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[retrieve] Wrote {RETRIEVED_PATH}")
    return out


# -----------------------------------------------------------------------------
# STAGE 2 — OpenAI Batch judge path
# -----------------------------------------------------------------------------
def build_judge_request(
    custom_id: str,
    context: str,
    correct_answer: str,
    model: str = "gpt-5",
    reasoning_effort: str = "high",
    max_output_tokens: int = 16000,
) -> BatchRequest:
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(context, correct_answer)},
        ],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    return BatchRequest(custom_id=custom_id, method="POST", url="/v1/responses", body=body)


def stage_submit_openai(logger: logging.Logger) -> Dict[str, Any]:
    retrieved = json.loads(RETRIEVED_PATH.read_text(encoding="utf-8"))

    requests: List[BatchRequest] = []
    for row in retrieved:
        fi = row["filtered_idx"]
        for item in row["items"]:
            qi = item["query_idx"]
            cid = f"judge|{fi}|{qi}"
            requests.append(
                build_judge_request(
                    custom_id=cid,
                    context=item["retrieved_context"],
                    correct_answer=item["correct_answer"],
                )
            )
    logger.info(f"[submit:openai] Total judge requests: {len(requests)}")

    batch = BatchClient(workdir=WORKDIR, poll_interval_seconds=60, logger=logger)
    results = batch.run_job(
        job_name=JUDGE_BATCH_JOB,
        endpoint="/v1/responses",
        requests=requests,
    )
    success = sum(1 for r in results.values() if r.success)
    failed = sum(1 for r in results.values() if not r.success)
    logger.info(f"[submit:openai] Batch parsed: success={success} failed={failed}")
    return results


# -----------------------------------------------------------------------------
# STAGE 2 — Local judge path (LLMClient model_type="local", threaded)
# -----------------------------------------------------------------------------
def _load_local_jsonl(path: Path) -> Dict[str, Dict]:
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
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8", buffering=1)
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


def _judge_one_local(client, context: str, correct_answer: str) -> Tuple[Optional[int], str, Optional[str]]:
    """Returns (evaluation 0/1 or None, raw_response_str, error_str)."""
    try:
        obj = client.generate_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=build_judge_user_prompt(context, correct_answer),
            max_output_tokens=4000,
        )
        ev = obj.get("evaluation")
        if isinstance(ev, bool):
            return int(ev), json.dumps(obj, ensure_ascii=False), None
        if isinstance(ev, int):
            return (1 if ev else 0), json.dumps(obj, ensure_ascii=False), None
        if isinstance(ev, str) and ev.strip() in {"0", "1"}:
            return int(ev.strip()), json.dumps(obj, ensure_ascii=False), None
        return None, json.dumps(obj, ensure_ascii=False), f"bad_evaluation: {ev!r}"
    except Exception as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def stage_submit_local(logger: logging.Logger) -> Dict[str, Dict]:
    """
    Run all judge calls against a local LLM with ThreadPoolExecutor concurrency.
    Streams each completed result to LOCAL_JUDGE_JSONL for resumability.
    Returns {custom_id -> row} where row has keys
        custom_id, evaluation (0/1 or None), raw_content, error.
    """
    if LOCAL_LLM_URL_OVERRIDE:
        os.environ["LOCAL_LLM_URL"] = LOCAL_LLM_URL_OVERRIDE
    from llm_client import LLMClient  # local import: env var must be set first

    retrieved = json.loads(RETRIEVED_PATH.read_text(encoding="utf-8"))

    # Build the full job list.
    jobs: List[Tuple[str, str, str]] = []        # (cid, context, correct_answer)
    for row in retrieved:
        fi = row["filtered_idx"]
        for item in row["items"]:
            qi = item["query_idx"]
            cid = f"judge|{fi}|{qi}"
            jobs.append((cid, item["retrieved_context"], item["correct_answer"]))

    cached = _load_local_jsonl(LOCAL_JUDGE_JSONL)
    todo = [j for j in jobs if j[0] not in cached]
    logger.info(
        f"[submit:local] total={len(jobs)} cached={len(cached)} todo={len(todo)} "
        f"workers={LOCAL_MAX_WORKERS} model={LOCAL_MODEL}"
    )

    client = LLMClient(
        model=LOCAL_MODEL,
        model_type="local",
        temperature=0.0,
        max_output_tokens=4000,
        max_retries=3,
        logger=logger,
    )

    writer = _AppendingWriter(LOCAL_JUDGE_JSONL)
    results: Dict[str, Dict] = dict(cached)
    n_done = 0
    n_failed = 0

    def _worker(job):
        cid, ctx, ans = job
        ev, raw, err = _judge_one_local(client, ctx, ans)
        return {
            "custom_id": cid,
            "evaluation": ev,
            "raw_content": raw,
            "error": err,
        }

    try:
        with ThreadPoolExecutor(max_workers=LOCAL_MAX_WORKERS) as pool:
            futures = {pool.submit(_worker, j): j[0] for j in todo}
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "custom_id": cid,
                        "evaluation": None,
                        "raw_content": "",
                        "error": f"executor_exception: {exc}",
                    }
                writer.write(row)
                results[cid] = row
                n_done += 1
                if row["evaluation"] is None:
                    n_failed += 1
                if n_done % 100 == 0 or n_done == len(todo):
                    logger.info(
                        f"[submit:local] progress {n_done}/{len(todo)} "
                        f"(failed_so_far={n_failed})"
                    )
    finally:
        writer.close()

    logger.info(
        f"[submit:local] done. ran={n_done} failed={n_failed} "
        f"total_with_cache={len(results)}"
    )
    return results


# -----------------------------------------------------------------------------
# STAGE 3 — assemble per-doc results
# -----------------------------------------------------------------------------
def _parse_judge_payload(content: str) -> Optional[int]:
    """Extract integer evaluation from a JSON-string content."""
    if not content:
        return None
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
    except Exception:
        return None
    ev = obj.get("evaluation")
    if isinstance(ev, bool):
        return int(ev)
    if isinstance(ev, int):
        return 1 if ev else 0
    if isinstance(ev, str) and ev.strip() in {"0", "1"}:
        return int(ev.strip())
    return None


def _sync_retry_judge_openai(
    context: str, correct_answer: str, model: str = "gpt-5"
) -> Optional[int]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_user_prompt(context, correct_answer)},
            ],
            reasoning={"effort": "high"},
            max_output_tokens=16000,
            text={"format": {"type": "json_object"}},
        )
        parts = []
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                if getattr(c, "text", None):
                    parts.append(c.text)
        return _parse_judge_payload("".join(parts))
    except Exception:
        return None


def _sync_retry_judge_local(context: str, correct_answer: str) -> Optional[int]:
    if LOCAL_LLM_URL_OVERRIDE:
        os.environ["LOCAL_LLM_URL"] = LOCAL_LLM_URL_OVERRIDE
    try:
        from llm_client import LLMClient

        client = LLMClient(
            model=LOCAL_MODEL, model_type="local",
            temperature=0.0, max_output_tokens=4000, max_retries=3,
        )
        ev, _raw, _err = _judge_one_local(client, context, correct_answer)
        return ev
    except Exception:
        return None


def _evaluation_for_cid_openai(cid: str, batch_results) -> Optional[int]:
    entry = batch_results.get(cid)
    if entry is None or not entry.success:
        return None
    return _parse_judge_payload(entry.content or "")


def _evaluation_for_cid_local(cid: str, local_rows: Dict[str, Dict]) -> Optional[int]:
    row = local_rows.get(cid)
    if row is None:
        return None
    return row.get("evaluation")


def stage_assemble(logger: logging.Logger) -> None:
    retrieved = json.loads(RETRIEVED_PATH.read_text(encoding="utf-8"))

    if BACKEND == "openai":
        batch = BatchClient(workdir=WORKDIR, poll_interval_seconds=60, logger=logger)
        batch_results = batch.parse_results(JUDGE_BATCH_JOB, endpoint="/v1/responses")
        local_rows: Dict[str, Dict] = {}
    else:
        batch_results = None
        local_rows = _load_local_jsonl(LOCAL_JUDGE_JSONL)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_correct = 0
    overall_total = 0
    retries = 0
    retry_failures = 0

    for row in retrieved:
        fi = row["filtered_idx"]
        doc_id = row["doc_id"]
        items = row["items"]

        per_query: List[Dict[str, Any]] = []
        correct = 0
        for item in items:
            qi = item["query_idx"]
            cid = f"judge|{fi}|{qi}"

            if BACKEND == "openai":
                evaluation = _evaluation_for_cid_openai(cid, batch_results)
            else:
                evaluation = _evaluation_for_cid_local(cid, local_rows)

            if evaluation is None:
                retries += 1
                logger.info(f"[assemble] Sync-retry judge for {cid}")
                if BACKEND == "openai":
                    evaluation = _sync_retry_judge_openai(
                        item["retrieved_context"], item["correct_answer"]
                    )
                else:
                    evaluation = _sync_retry_judge_local(
                        item["retrieved_context"], item["correct_answer"]
                    )
                if evaluation is None:
                    retry_failures += 1
                    evaluation = 0       # fail closed

            per_query.append(
                {
                    "correct_answer": item["correct_answer"],
                    "retrieved_context": item["retrieved_context"],
                    "evaluation": int(evaluation),
                }
            )
            correct += int(evaluation)

        total = len(items)
        accuracy_str = f"{(correct/total)*100:.2f}%" if total else "0.00%"
        per_query.append({"accuracy": accuracy_str})

        out_path = RESULTS_DIR / f"results_{fi}.json"
        out_path.write_text(json.dumps(per_query, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"[assemble] {out_path.name} ({doc_id}): {correct}/{total} = {accuracy_str}"
        )
        overall_correct += correct
        overall_total += total

    overall_acc = (overall_correct / overall_total * 100) if overall_total else 0.0
    summary = {
        "experiment_folder": EXPERIMENT_FOLDER,
        "backend": BACKEND,
        "judge_model": LOCAL_MODEL if BACKEND == "local" else "gpt-5",
        "total_essays": len(retrieved),
        "total_queries": overall_total,
        "correct": overall_correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "sync_retries": retries,
        "sync_retry_failures": retry_failures,
    }
    (RESULTS_DIR / "summary_ours.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("=" * 60)
    logger.info(f"[assemble] OVERALL ACCURACY: {summary['overall_accuracy_pct']}%")
    logger.info(
        f"[assemble] {overall_correct}/{overall_total} queries correct "
        f"across {len(retrieved)} essays"
    )
    logger.info(
        f"[assemble] Sync retries: {retries} (failures: {retry_failures})"
    )
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate our KGs against MINE GT (configurable backend in MANUAL CONFIG)."
    )
    parser.add_argument(
        "stage",
        choices=["retrieve", "submit", "assemble", "all"],
        help="Which stage to run. 'all' runs the three in sequence.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N docs (smoke testing).",
    )
    parser.add_argument(
        "--force-retrieve", action="store_true",
        help="Delete retrieved.json before running (forces re-retrieval).",
    )
    parser.add_argument(
        "--retrieve-workers", type=int, default=4,
        help="Thread workers for the local retrieve stage.",
    )
    args = parser.parse_args()

    # Backend-specific env setup. Set LOCAL_LLM_URL before LLMClient is imported
    # (it caches the URL at class-definition time).
    if BACKEND == "local":
        if LOCAL_LLM_URL_OVERRIDE:
            os.environ["LOCAL_LLM_URL"] = LOCAL_LLM_URL_OVERRIDE
    else:
        if os.getenv("OPENAI_API_KEY_2"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]

    logger = _make_logger()
    logger.info("=" * 70)
    logger.info(
        f"OURS MINE evaluation — stage={args.stage}, limit={args.limit}, "
        f"backend={BACKEND}, experiment={EXPERIMENT_FOLDER}"
    )
    logger.info(f"  experiments_root = {EXPERIMENTS_ROOT}")
    logger.info(f"  results_dir      = {RESULTS_DIR}")
    if BACKEND == "local":
        logger.info(
            f"  endpoint={os.environ.get('LOCAL_LLM_URL', '<default in llm_client>')} "
            f"model={LOCAL_MODEL} workers={LOCAL_MAX_WORKERS}"
        )
    logger.info("=" * 70)

    if args.force_retrieve and RETRIEVED_PATH.exists():
        logger.info(f"Removing {RETRIEVED_PATH}")
        RETRIEVED_PATH.unlink()

    if args.stage in ("retrieve", "all"):
        stage_retrieve(logger, limit=args.limit, max_workers=args.retrieve_workers)
    if args.stage in ("submit", "all"):
        if BACKEND == "local":
            stage_submit_local(logger)
        else:
            stage_submit_openai(logger)
    if args.stage in ("assemble", "all"):
        stage_assemble(logger)


if __name__ == "__main__":
    main()
