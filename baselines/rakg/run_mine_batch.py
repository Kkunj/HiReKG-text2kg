"""
Resumable batch runner for the RAKG pipeline over the MINE dataset.

Mirrors baselines/kggen/run_mine_batch.py and our_approach/run_mine_batch.py:
same resumability semantics, same output layout, same atomic status writes —
only the pipeline import and config keys differ.

Per-doc output layout:
    experiments/<batch_name>/<doc_id>/            <- standard pipeline output
    experiments/<batch_name>/_batch_status.json   <- rewritten after every doc
    experiments/<batch_name>/_batch.log           <- batch-level events

A doc is considered "done" iff its `final_output.json` exists. Rerunning skips
done docs and retries any failed ones automatically.

Usage
-----
    python run_mine_batch.py                       # process every doc
    python run_mine_batch.py --only doc_0,doc_3    # only these
    python run_mine_batch.py --limit 5             # first 5 remaining docs
    python run_mine_batch.py --force               # reprocess everything
    python run_mine_batch.py --list                # report status, do not run

    # Override dataset, model, and batch name:
    python run_mine_batch.py --batch-name rakg_scierc_gpt4o \
        --texts-dir C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\texts \
        --model gpt-4o --model-type openai
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pre-parse --endpoint BEFORE importing pipeline. LLMClient.SELF_HOSTED_URL
# is read from LOCAL_LLM_URL exactly once at class-definition time, so the
# env var must be set before the import chain reaches llm_client.
for _i, _a in enumerate(sys.argv):
    if _a == "--endpoint" and _i + 1 < len(sys.argv):
        os.environ["LOCAL_LLM_URL"] = sys.argv[_i + 1]
        break
    if _a.startswith("--endpoint="):
        os.environ["LOCAL_LLM_URL"] = _a.split("=", 1)[1]
        break

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipeline_parallel import run_pipeline  # noqa: E402


# Defaults — overridable via CLI args --batch-name, --texts-dir, --model, --model-type
BATCH_NAME = "rakg_redocred_qwen3_8b"
TEXTS_DIR = Path(r"C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\texts")
EXPERIMENTS_ROOT = _THIS_DIR / "experiments"
BATCH_DIR = EXPERIMENTS_ROOT / BATCH_NAME

# These are set in main() based on --start/--end so parallel instances
# (one per vLLM port) do not race on a shared status file or log file.
STATUS_FILE: Path = BATCH_DIR / "_batch_status.json"
BATCH_LOG_FILE: Path = BATCH_DIR / "_batch.log"


BASE_CONFIG: Dict[str, Any] = {
    # LLM — locally hosted Qwen via vLLM (same as our_approach + kggen runners)
    "model": "qwen3-8b",
    "model_type": "local",
    "temperature": 0.0,
    "max_output_tokens": 13000,
    "max_retries": 3,

    # Similarity LLM — falls back to main LLM when None
    "similarity_model": None,
    "similarity_model_type": None,

    # Embeddings — local BGE-M3 (RAKG's natural default; avoids OpenAI round-trips)
    "embedding_backend": "local",
    "embedding_model": "BAAI/bge-m3",

    # RAKG pipeline knobs
    "similarity_threshold": 0.60,
    "retrieval_top_k": 5,

    # Output
    "save_experiments": True,

    # Per-doc pipeline logs go inside each doc's folder
    "log_to_console": False,
    "log_to_file": True,
}


def _apply_cli_overrides(args) -> None:
    """Apply --batch-name, --texts-dir, --model, --model-type to globals."""
    global BATCH_NAME, TEXTS_DIR, BATCH_DIR, STATUS_FILE, BATCH_LOG_FILE

    if args.batch_name:
        BATCH_NAME = args.batch_name
    if args.texts_dir:
        TEXTS_DIR = Path(args.texts_dir)
    if args.model:
        BASE_CONFIG["model"] = args.model
    if args.model_type:
        BASE_CONFIG["model_type"] = args.model_type

    BATCH_DIR = EXPERIMENTS_ROOT / BATCH_NAME
    STATUS_FILE = BATCH_DIR / "_batch_status.json"
    BATCH_LOG_FILE = BATCH_DIR / "_batch.log"


def _setup_batch_logger() -> logging.Logger:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RAKGMineBatch")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh = logging.FileHandler(BATCH_LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def _load_status() -> Dict[str, Any]:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {
        "batch_name": BATCH_NAME,
        "model": BASE_CONFIG["model"],
        "texts_dir": str(TEXTS_DIR),
        "started_at": datetime.now().isoformat(),
        "last_updated": None,
        "docs": {},
    }


def _save_status(status: Dict[str, Any]) -> None:
    status["last_updated"] = datetime.now().isoformat()
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATUS_FILE)  # atomic swap so crashes never leave partial status


def _is_done(doc_id: str) -> bool:
    return (BATCH_DIR / doc_id / "final_output.json").exists()


def _discover_docs() -> List[Path]:
    # Handle two naming conventions: MINE-style "doc_<N>.txt" (numeric sort)
    # and ReDocRED-style descriptive names like "Asus_VivoTab.txt" (lex sort).
    paths = list(TEXTS_DIR.glob("*.txt"))
    try:
        return sorted(paths, key=lambda p: int(p.stem.split("_")[1]))
    except (ValueError, IndexError):
        return sorted(paths)


def _process_doc(doc_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    text = doc_path.read_text(encoding="utf-8")

    cfg = dict(BASE_CONFIG)
    # Nested experiment_name → pipeline saves into experiments/<batch>/<doc_id>/
    cfg["experiment_name"] = f"{BATCH_NAME}/{doc_path.stem}"

    t0 = time.time()
    results = run_pipeline(text, cfg)
    elapsed = time.time() - t0

    return {
        "status": "ok",
        "chars": len(text),
        "entities": len(results["entities_refined"]),
        "triples": len(results["triples_final"]),
        "elapsed_seconds": round(elapsed, 2),
        "finished_at": datetime.now().isoformat(),
    }


def _parse_doc_index(s: Optional[str]) -> Optional[int]:
    """Accept either '5' or 'doc_5'. Returns None when input is None."""
    if s is None:
        return None
    s = str(s).strip()
    if s.lower().startswith("doc_"):
        s = s[4:]
    return int(s)


def _filter_docs(
    all_docs: List[Path],
    only: Optional[List[str]],
    limit: Optional[int],
    force: bool,
    start: Optional[int],
    end: Optional[int],
) -> List[Path]:
    if only:
        wanted = set(only)
        selected = [p for p in all_docs if p.stem in wanted]
        missing = wanted - {p.stem for p in selected}
        if missing:
            print(f"WARN: --only referenced unknown docs: {sorted(missing)}")
    else:
        selected = all_docs

    # Range filter — both bounds inclusive when provided.
    if start is not None or end is not None:
        lo = start if start is not None else -1
        hi = end if end is not None else 10**9
        selected = [
            p for p in selected
            if lo <= int(p.stem.split("_")[1]) <= hi
        ]

    if not force:
        selected = [p for p in selected if not _is_done(p.stem)]

    if limit is not None:
        selected = selected[:limit]
    return selected


def _print_status_report(all_docs: List[Path]) -> None:
    done, pending, failed = [], [], []

    # Aggregate failure records across every per-instance status file so
    # `--list` reports correctly even when 3 terminals split the work.
    failures: Dict[str, str] = {}
    for sf in BATCH_DIR.glob("_batch_status*.json"):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            for doc_id, rec in data.get("docs", {}).items():
                if rec.get("status") == "error":
                    failures[doc_id] = rec.get("error", "")[:80]
        except Exception:
            pass

    for p in all_docs:
        if _is_done(p.stem):
            done.append(p.stem)
        elif p.stem in failures:
            failed.append((p.stem, failures[p.stem]))
        else:
            pending.append(p.stem)

    print(f"Batch:   {BATCH_NAME}")
    print(f"Dir:     {BATCH_DIR}")
    print(f"Total:   {len(all_docs)}")
    print(f"Done:    {len(done)}")
    print(f"Pending: {len(pending)}")
    print(f"Failed:  {len(failed)}")
    if failed:
        print("  Failed docs (will be retried on next run):")
        for doc_id, err in failed[:10]:
            print(f"    {doc_id}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAKG pipeline over every MINE doc (resumable, parallel-safe)."
    )
    parser.add_argument("--batch-name", type=str, default=None,
                        help="Override batch/experiment name (e.g. rakg_scierc_gpt4o).")
    parser.add_argument("--texts-dir", type=str, default=None,
                        help="Override path to the texts directory.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model name (e.g. gpt-4o).")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Override LLM backend (openai, gemini, local, nim).")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated doc ids (e.g. doc_0,doc_5).")
    parser.add_argument("--start", type=str, default=None,
                        help="Start of doc range, inclusive (e.g. 0 or doc_0).")
    parser.add_argument("--end", type=str, default=None,
                        help="End of doc range, inclusive (e.g. 32 or doc_32).")
    parser.add_argument("--endpoint", type=str, default=None,
                        help=("Override LOCAL_LLM_URL for this run, e.g. "
                              "http://<LOCAL_LLM_HOST>:8084/v1/chat/completions. "
                              "Used by parallel runs targeting different vLLM ports."))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N docs from the remaining set.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess docs even if final_output.json exists.")
    parser.add_argument("--list", action="store_true",
                        help="Print status summary and exit.")
    args = parser.parse_args()

    _apply_cli_overrides(args)

    if not TEXTS_DIR.exists():
        sys.exit(f"ERROR: texts dir not found: {TEXTS_DIR}")

    start = _parse_doc_index(args.start)
    end = _parse_doc_index(args.end)
    if start is not None and end is not None and start > end:
        sys.exit(f"ERROR: --start ({start}) must be <= --end ({end})")

    # Per-instance status + log file when a range is given, so 3 parallel
    # terminals never overwrite each other's records.
    global STATUS_FILE, BATCH_LOG_FILE
    if start is not None or end is not None:
        s = start if start is not None else 0
        e = end if end is not None else 99
        STATUS_FILE = BATCH_DIR / f"_batch_status_{s}_{e}.json"
        BATCH_LOG_FILE = BATCH_DIR / f"_batch_{s}_{e}.log"

    all_docs = _discover_docs()
    if not all_docs:
        sys.exit(f"ERROR: no doc_*.txt files in {TEXTS_DIR}")

    if args.list:
        _print_status_report(all_docs)
        return

    only_list = [s.strip() for s in args.only.split(",")] if args.only else None
    todo = _filter_docs(all_docs, only_list, args.limit, args.force, start, end)

    logger = _setup_batch_logger()
    logger.info("=== RAKG MINE batch start ===")
    logger.info(f"Batch name:  {BATCH_NAME}")
    logger.info(f"Model:       {BASE_CONFIG['model']} ({BASE_CONFIG['model_type']})")
    logger.info(f"Embeddings:  {BASE_CONFIG['embedding_model']} ({BASE_CONFIG['embedding_backend']})")
    logger.info(f"Endpoint:    {os.environ.get('LOCAL_LLM_URL', '<default in llm_client>')}")
    if start is not None or end is not None:
        logger.info(f"Range:       start={start}  end={end}")
    logger.info(f"Total docs:  {len(all_docs)}   todo: {len(todo)}   "
                f"already done: {len(all_docs) - len(todo) if not args.force else 0}")

    if not todo:
        logger.info("Nothing to do. (Use --force to reprocess.)")
        return

    status = _load_status()
    batch_t0 = time.time()

    for i, doc_path in enumerate(todo, 1):
        doc_id = doc_path.stem
        logger.info(f"[{i}/{len(todo)}] {doc_id} ...")
        try:
            record = _process_doc(doc_path, logger)
            status["docs"][doc_id] = record
            logger.info(
                f"    OK  entities={record['entities']} "
                f"triples={record['triples']} "
                f"({record['elapsed_seconds']}s)"
            )
        except KeyboardInterrupt:
            status["docs"][doc_id] = {
                "status": "interrupted",
                "finished_at": datetime.now().isoformat(),
            }
            _save_status(status)
            logger.warning("Interrupted by user. Progress saved; rerun to resume.")
            sys.exit(130)
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            status["docs"][doc_id] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "finished_at": datetime.now().isoformat(),
            }
            logger.error(f"    FAIL  {type(exc).__name__}: {exc}")
        finally:
            _save_status(status)

    elapsed = time.time() - batch_t0
    ok = sum(1 for d in status["docs"].values() if d.get("status") == "ok")
    err = sum(1 for d in status["docs"].values() if d.get("status") == "error")
    logger.info(
        f"=== Batch done in {elapsed/60:.1f} min   ok={ok}  err={err} ==="
    )
    if err:
        logger.info("Rerun the script to retry failed docs.")


if __name__ == "__main__":
    main()
