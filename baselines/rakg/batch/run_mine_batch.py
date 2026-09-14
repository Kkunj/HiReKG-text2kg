"""
Entry point for the batched RAKG pipeline (OpenAI Batch API).

Picks N random docs from a texts directory (or all of them) and runs the
batched RAKG pipeline, writing per-doc folders under
    baselines/rakg/experiments/<experiment_name>/<doc_id>/.

Usage
-----
    python run_mine_batch.py --all                  # process every doc in --texts-dir
    python run_mine_batch.py --docs 3 --seed 7      # sample 3 random docs (deterministic)
    python run_mine_batch.py --all --dry-run        # write input .jsonl files only, do not submit

    # Override dataset and experiment name:
    python run_mine_batch.py --all \
        --texts-dir C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\texts \
        --experiment-name rakg_scierc_gpt4o \
        --model gpt-4o
"""

from __future__ import annotations

import argparse
import os
import random

import sys
from pathlib import Path
from typing import Optional

_BATCH_DIR = Path(__file__).resolve().parent
if str(_BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BATCH_DIR))

from batch_pipeline import BatchRAKGPipeline


CONFIG = {
    # --- LLM ---
    "model": "gpt-4o-2024-11-20",
    "model_type": "openai",
    "base_url": None,
    "temperature": 0.0,
    "max_output_tokens": 10000,
    "max_retries": 3,

    # --- Embeddings ---
    "embedding_model": "text-embedding-3-large",
    "embedding_backend": "openai",

    # --- RAKG knobs ---
    "similarity_threshold": 0.60,
    "retrieval_top_k": 5,

    # --- Experiment output ---
    "experiment_name": "RAKG_MINE_batch_results",

    # --- Logging ---
    "log_to_console": True,
    "log_to_file": True,
}


def select_docs(texts_dir: Path, n: Optional[int], seed: int) -> dict:
    all_files = sorted(texts_dir.glob("*.txt"))
    if not all_files:
        raise ValueError(f"No .txt files found in {texts_dir}")
    if n is None:
        picked = all_files
    else:
        if len(all_files) < n:
            raise ValueError(
                f"Requested {n} docs but only {len(all_files)} found in {texts_dir}"
            )
        rng = random.Random(seed)
        picked = rng.sample(all_files, n)
    return {p.stem: p.read_text(encoding="utf-8") for p in picked}


def main():
    parser = argparse.ArgumentParser(
        description="Batched RAKG pipeline over MINE docs"
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=None,
        help="Number of docs to sample. Omit (or use --all) to process every doc.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every doc in --texts-dir (equivalent to omitting --docs).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (ignored when processing all docs)",
    )
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=Path("<PROJECT_ROOT>/graph_rag/datasets/MINE/texts"),
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Override experiment/batch name (e.g. rakg_scierc_gpt4o).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override LLM model name (e.g. gpt-4o).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build input .jsonl files locally and report counts; do not submit.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    # Mirror our_approach: prefer the secondary key when present.
    if os.getenv("OPENAI_API_KEY_2"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]
        print("Using OPENAI_API_KEY_2 for this run.")
    else:
        print("WARNING: OPENAI_API_KEY_2 not found; using OPENAI_API_KEY.")

    # Apply CLI overrides to CONFIG
    if args.experiment_name:
        CONFIG["experiment_name"] = args.experiment_name
    if args.model:
        CONFIG["model"] = args.model

    n_docs: Optional[int] = None if args.all else args.docs
    docs = select_docs(args.texts_dir, n_docs, args.seed)
    print(f"Selected {len(docs)} docs: "
          f"{list(docs.keys())[:10]}{'...' if len(docs) > 10 else ''}")
    for doc_id, text in docs.items():
        print(f"  {doc_id}: {len(text)} chars, ~{len(text.split())} words")

    pipeline = BatchRAKGPipeline(config=CONFIG)
    pipeline.run(docs, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
