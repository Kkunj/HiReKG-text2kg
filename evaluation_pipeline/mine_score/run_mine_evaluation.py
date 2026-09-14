"""
Unified MINE evaluation script.

Evaluates any KG (OURS / KGGen / RAKG / ...) against ground-truth atomic
facts using the MINE score.

Pipeline — 3 stages, each resumable:

    STAGE 1 (local, fast):  retrieve
        For each doc in the ground-truth directory:
            - load predicted KG (final_output.json) → kg-gen dict
            - build nx graph + MiniLM node embeddings
            - for each GT fact: kggen.retrieve(fact, ...) → context text
        Save all (context, correct_answer) pairs to
            <workdir>/retrieved.json

    STAGE 2 (judge):  evaluates each (context, correct_answer) pair.
        - BACKEND="openai" → OpenAI Batch API (gpt-5/high, /v1/responses).
        - BACKEND="local"  → self-hosted LLM via our_approach.llm_client
                             (model_type="local"), threaded synchronous calls,
                             results streamed to judge_local_results.jsonl
                             for resumability.

    STAGE 3 (local, fast):  assemble
        Per-doc results_<idx>.json + summary.json.

Usage:
    python run_mine_evaluation.py all \\
        --gt-dir  datasets/scierc/atomic_facts/stage3_final \\
        --pred-dir baselines/kggen/experiments/kggen_mine_qwen3_8b \\
        --run-tag kggen_scierc

    python run_mine_evaluation.py retrieve --gt-dir ... --pred-dir ...
    python run_mine_evaluation.py submit   --gt-dir ... --pred-dir ...
    python run_mine_evaluation.py assemble --gt-dir ... --pred-dir ...
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

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent                    # graph_rag/
_OUR_APPROACH_DIR = _REPO_ROOT / "our_approach"
for p in (str(_THIS_DIR), str(_OUR_APPROACH_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from batch.batch_client import (  # noqa: E402
    BatchClient,
    BatchRequest,
)


# -----------------------------------------------------------------------------
# KG adapter (inlined — identical across OURS / KGGEN / RAKG)
# -----------------------------------------------------------------------------
def _clean(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _unique_preserve_order(items) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def triples_to_kggen_dict(triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    relations_tuples: List[List[str]] = []
    node_candidates: List[str] = []
    edge_candidates: List[str] = []
    for t in triples:
        subj = _clean(t.get("subject"))
        rel = _clean(t.get("relation"))
        obj = _clean(t.get("object"))
        if not (subj and rel and obj):
            continue
        relations_tuples.append([subj, rel, obj])
        node_candidates.append(subj)
        node_candidates.append(obj)
        edge_candidates.append(rel)
    return {
        "entities": _unique_preserve_order(node_candidates),
        "edges": _unique_preserve_order(edge_candidates),
        "relations": relations_tuples,
    }


def load_kg_for_doc(pred_dir: Path, doc_id: str) -> Dict[str, Any]:
    final_path = pred_dir / doc_id / "final_output.json"
    if not final_path.exists():
        raise FileNotFoundError(
            f"Missing {final_path} — was the pipeline run for {doc_id}?"
        )
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    triples = payload.get("triples_final", []) or []
    return triples_to_kggen_dict(triples)


# -----------------------------------------------------------------------------
# Ground-truth loading from stage3_final
# -----------------------------------------------------------------------------
def load_ground_truth(gt_dir: Path) -> List[Dict[str, Any]]:
    """
    Read stage3_final directory. Each JSON file has:
        {"doc_id": "...", "facts": ["...", ...], ...}

    Supports both naming conventions:
        - doc_0.json, doc_1.json, ...  (sorted numerically)
        - Asus_VivoTab.json, ...       (sorted alphabetically)
    """
    gt_dir = Path(gt_dir)
    all_json = sorted(gt_dir.glob("*.json"))
    if not all_json:
        raise FileNotFoundError(f"No .json files found in {gt_dir}")

    docs: List[Dict[str, Any]] = []
    for p in all_json:
        data = json.loads(p.read_text(encoding="utf-8"))
        docs.append({
            "doc_id": data.get("doc_id", p.stem),
            "facts": data.get("facts", []),
        })
    return docs


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
def _make_logger(workdir: Path) -> logging.Logger:
    logger = logging.getLogger("MINE_eval")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    workdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        workdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"
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
# STAGE 1 — retrieve
# -----------------------------------------------------------------------------
def _retrieve_single_doc(
    args: Tuple[int, str, List[str], Path],
) -> Tuple[int, str, List[Dict], Optional[str]]:
    idx, doc_id, facts, pred_dir = args
    try:
        from kg_gen.kg_gen import KGGen

        kg_dict = load_kg_for_doc(pred_dir, doc_id)
        if not kg_dict["relations"]:
            return (idx, doc_id, [], "empty KG (no relations)")

        kggen = KGGen(
            retrieval_model="all-MiniLM-L6-v2",
            model="openai/gpt-5",
            api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            temperature=1.0,
            max_tokens=16000,
        )
        graph = kggen.from_dict(kg_dict)
        nxGraph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nxGraph)

        items: List[Dict] = []
        for qi, fact in enumerate(facts):
            *_, context_text = kggen.retrieve(fact, node_embeddings, nxGraph)
            items.append({
                "query_idx": qi,
                "correct_answer": fact,
                "retrieved_context": context_text,
            })
        return (idx, doc_id, items, None)
    except Exception as exc:
        return (idx, doc_id, [], f"{type(exc).__name__}: {exc}")


def stage_retrieve(
    logger: logging.Logger,
    gt_dir: Path,
    pred_dir: Path,
    retrieved_path: Path,
    limit: Optional[int] = None,
    max_workers: int = 4,
) -> List[Dict]:
    if retrieved_path.exists():
        logger.info(f"[retrieve] Using cached {retrieved_path}")
        return json.loads(retrieved_path.read_text(encoding="utf-8"))

    gt_docs = load_ground_truth(gt_dir)
    if limit is not None:
        gt_docs = gt_docs[:limit]

    tasks = []
    for i, doc in enumerate(gt_docs):
        tasks.append((i, doc["doc_id"], doc["facts"], pred_dir))

    logger.info(
        f"[retrieve] Retrieving contexts for {len(tasks)} docs "
        f"(max_workers={max_workers})"
    )

    out: List[Dict] = [None] * len(tasks)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_retrieve_single_doc, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            idx, doc_id, items, err = fut.result()
            done += 1
            if err:
                logger.warning(
                    f"[retrieve] [{done}/{len(tasks)}] {doc_id}: {err}"
                )
            else:
                logger.info(
                    f"[retrieve] [{done}/{len(tasks)}] {doc_id}: "
                    f"{len(items)} queries retrieved"
                )
            out[idx] = {
                "filtered_idx": idx,
                "doc_id": doc_id,
                "items": items,
                "retrieval_error": err,
            }

    retrieved_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"[retrieve] Wrote {retrieved_path}")
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


def stage_submit_openai(
    logger: logging.Logger,
    workdir: Path,
    retrieved_path: Path,
) -> Dict[str, Any]:
    retrieved = json.loads(retrieved_path.read_text(encoding="utf-8"))

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

    batch = BatchClient(workdir=workdir, poll_interval_seconds=60, logger=logger)
    results = batch.run_job(
        job_name="judge_batch",
        endpoint="/v1/responses",
        requests=requests,
    )
    success = sum(1 for r in results.values() if r.success)
    failed = sum(1 for r in results.values() if not r.success)
    logger.info(f"[submit:openai] Batch parsed: success={success} failed={failed}")
    return results


# -----------------------------------------------------------------------------
# STAGE 2 — Local judge path
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


def _judge_one_local(
    client, context: str, correct_answer: str,
) -> Tuple[Optional[int], str, Optional[str]]:
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


def stage_submit_local(
    logger: logging.Logger,
    retrieved_path: Path,
    local_judge_jsonl: Path,
    local_model: str,
    local_max_workers: int,
    local_llm_url: Optional[str],
) -> Dict[str, Dict]:
    if local_llm_url:
        os.environ["LOCAL_LLM_URL"] = local_llm_url
    from llm_client import LLMClient

    retrieved = json.loads(retrieved_path.read_text(encoding="utf-8"))

    jobs: List[Tuple[str, str, str]] = []
    for row in retrieved:
        fi = row["filtered_idx"]
        for item in row["items"]:
            qi = item["query_idx"]
            cid = f"judge|{fi}|{qi}"
            jobs.append((cid, item["retrieved_context"], item["correct_answer"]))

    cached = _load_local_jsonl(local_judge_jsonl)
    todo = [j for j in jobs if j[0] not in cached]
    logger.info(
        f"[submit:local] total={len(jobs)} cached={len(cached)} todo={len(todo)} "
        f"workers={local_max_workers} model={local_model}"
    )

    client = LLMClient(
        model=local_model,
        model_type="local",
        temperature=0.0,
        max_output_tokens=4000,
        max_retries=3,
        logger=logger,
    )

    writer = _AppendingWriter(local_judge_jsonl)
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
        with ThreadPoolExecutor(max_workers=local_max_workers) as pool:
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
# STAGE 3 — assemble
# -----------------------------------------------------------------------------
def _parse_judge_payload(content: str) -> Optional[int]:
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
    context: str, correct_answer: str, model: str = "gpt-5",
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


def _sync_retry_judge_local(
    context: str,
    correct_answer: str,
    local_model: str,
    local_llm_url: Optional[str],
) -> Optional[int]:
    if local_llm_url:
        os.environ["LOCAL_LLM_URL"] = local_llm_url
    try:
        from llm_client import LLMClient

        client = LLMClient(
            model=local_model, model_type="local",
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


def stage_assemble(
    logger: logging.Logger,
    backend: str,
    retrieved_path: Path,
    workdir: Path,
    results_dir: Path,
    local_judge_jsonl: Path,
    run_tag: str,
    local_model: str = "",
    local_llm_url: Optional[str] = None,
) -> None:
    retrieved = json.loads(retrieved_path.read_text(encoding="utf-8"))

    if backend == "openai":
        batch = BatchClient(workdir=workdir, poll_interval_seconds=60, logger=logger)
        batch_results = batch.parse_results("judge_batch", endpoint="/v1/responses")
        local_rows: Dict[str, Dict] = {}
    else:
        batch_results = None
        local_rows = _load_local_jsonl(local_judge_jsonl)

    results_dir.mkdir(parents=True, exist_ok=True)
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

            if backend == "openai":
                evaluation = _evaluation_for_cid_openai(cid, batch_results)
            else:
                evaluation = _evaluation_for_cid_local(cid, local_rows)

            if evaluation is None:
                retries += 1
                logger.info(f"[assemble] Sync-retry judge for {cid}")
                if backend == "openai":
                    evaluation = _sync_retry_judge_openai(
                        item["retrieved_context"], item["correct_answer"]
                    )
                else:
                    evaluation = _sync_retry_judge_local(
                        item["retrieved_context"], item["correct_answer"],
                        local_model, local_llm_url,
                    )
                if evaluation is None:
                    retry_failures += 1
                    evaluation = 0

            per_query.append({
                "correct_answer": item["correct_answer"],
                "retrieved_context": item["retrieved_context"],
                "evaluation": int(evaluation),
            })
            correct += int(evaluation)

        total = len(items)
        accuracy_str = f"{(correct/total)*100:.2f}%" if total else "0.00%"
        per_query.append({"accuracy": accuracy_str})

        out_path = results_dir / f"results_{fi}.json"
        out_path.write_text(
            json.dumps(per_query, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            f"[assemble] {out_path.name} ({doc_id}): {correct}/{total} = {accuracy_str}"
        )
        overall_correct += correct
        overall_total += total

    overall_acc = (overall_correct / overall_total * 100) if overall_total else 0.0
    summary = {
        "run_tag": run_tag,
        "backend": backend,
        "judge_model": local_model if backend == "local" else "gpt-5",
        "total_docs": len(retrieved),
        "total_queries": overall_total,
        "correct": overall_correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "sync_retries": retries,
        "sync_retry_failures": retry_failures,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("=" * 60)
    logger.info(f"[assemble] OVERALL ACCURACY (MINE score): {summary['overall_accuracy_pct']}%")
    logger.info(
        f"[assemble] {overall_correct}/{overall_total} queries correct "
        f"across {len(retrieved)} docs"
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
        description="Unified MINE evaluation: any KG vs ground-truth atomic facts."
    )
    parser.add_argument(
        "stage",
        choices=["retrieve", "submit", "assemble", "all"],
        help="Which stage to run. 'all' runs all three in sequence.",
    )
    parser.add_argument(
        "--gt-dir", type=str, required=True,
        help="Path to ground-truth atomic facts directory "
             "(e.g. datasets/scierc/atomic_facts/stage3_final). "
             "Each doc_X.json must have {doc_id, facts: [...]}.",
    )
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Path to predicted KG experiment directory. "
             "Each doc_X/ subfolder must contain final_output.json "
             "with a triples_final field.",
    )
    parser.add_argument(
        "--run-tag", type=str, default=None,
        help="Tag for this evaluation run (used in output paths). "
             "Defaults to the pred-dir folder name.",
    )
    parser.add_argument(
        "--backend", type=str, choices=["local", "openai"], default="local",
        help="Judge backend (default: local).",
    )
    parser.add_argument(
        "--local-model", type=str, default="qwen3-14b",
        help="Model name for local backend (default: qwen3-14b).",
    )
    parser.add_argument(
        "--local-workers", type=int, default=32,
        help="Thread workers for local judge (default: 32).",
    )
    parser.add_argument(
        "--local-llm-url", type=str, default=None,
        help="Override URL for local LLM endpoint.",
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
        help="Thread workers for the retrieve stage (default: 4).",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    run_tag = args.run_tag or pred_dir.name
    backend = args.backend

    # Derived paths
    workdir = _THIS_DIR / "batch_workdir" / f"{run_tag}__{backend}"
    retrieved_path = workdir / "retrieved.json"
    results_dir = _THIS_DIR / "results" / f"{run_tag}__{backend}"
    local_judge_jsonl = workdir / "judge_local_results.jsonl"

    # Backend env setup
    if backend == "local":
        if args.local_llm_url:
            os.environ["LOCAL_LLM_URL"] = args.local_llm_url
    else:
        if os.getenv("OPENAI_API_KEY_2"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]

    logger = _make_logger(workdir)
    logger.info("=" * 70)
    logger.info(
        f"MINE evaluation — stage={args.stage}, limit={args.limit}, "
        f"backend={backend}, run_tag={run_tag}"
    )
    logger.info(f"  gt_dir     = {gt_dir}")
    logger.info(f"  pred_dir   = {pred_dir}")
    logger.info(f"  results    = {results_dir}")
    if backend == "local":
        logger.info(
            f"  endpoint={os.environ.get('LOCAL_LLM_URL', '<default in llm_client>')} "
            f"model={args.local_model} workers={args.local_workers}"
        )
    logger.info("=" * 70)

    if args.force_retrieve and retrieved_path.exists():
        logger.info(f"Removing {retrieved_path}")
        retrieved_path.unlink()

    if args.stage in ("retrieve", "all"):
        stage_retrieve(
            logger, gt_dir, pred_dir, retrieved_path,
            limit=args.limit, max_workers=args.retrieve_workers,
        )
    if args.stage in ("submit", "all"):
        if backend == "local":
            stage_submit_local(
                logger, retrieved_path, local_judge_jsonl,
                args.local_model, args.local_workers, args.local_llm_url,
            )
        else:
            stage_submit_openai(logger, workdir, retrieved_path)
    if args.stage in ("assemble", "all"):
        stage_assemble(
            logger, backend, retrieved_path, workdir, results_dir,
            local_judge_jsonl, run_tag,
            local_model=args.local_model,
            local_llm_url=args.local_llm_url,
        )


if __name__ == "__main__":
    main()
