"""
Entry point for Gemini-batched KG pipeline.

Mirrors batch/run_mine_batch.py but targets the Gemini Batch API with
gemini-2.5-pro for generation and gemini-embedding-2 for embeddings.

Usage:
    python run_mine_batch.py --all                                          # MINE (default)
    python run_mine_batch.py --all --texts-dir C:/path/to/texts --name foo  # custom dataset
    python run_mine_batch.py --docs 3 --seed 7                              # sample 3 docs
    python run_mine_batch.py --all --dry-run                                # validate only
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent.parent
if str(_OUR_APPROACH_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH_DIR))

from batch.gemini.gemini_batch_pipeline import GeminiBatchKGPipeline  # noqa: E402


CONFIG = {
    # --- LLM ---
    "model": "gemini-2.5-pro",
    "model_type": "gemini",
    "base_url": None,
    "temperature": 0.2,
    "max_output_tokens": 10000,
    "max_retries": 3,

    # --- Text chunking ---
    "sentences_per_chunk": 3,

    # --- Summarization ---
    "summarize_text": True,

    # --- Entity refinement ---
    "entity_refinement_mode": "llm",

    # --- Semantic linking ---
    "enable_semantic_linking": True,
    "embedding_backend": "gemini",
    "embedding_model": "gemini-embedding-2",
    "cluster_distance_threshold": 0.99,
    "min_cluster_size": 2,

    # --- Storage ---
    "store_to_neo4j": False,

    # --- Experiment output ---
    "save_experiments": True,
    "experiment_name": "MINE_gemini_batch_results",

    # --- Logging ---
    "log_to_console": True,
    "log_to_file": True,
}


def select_docs(texts_dir: Path, n: Optional[int], seed: int) -> dict:
    """Select documents from the texts directory."""
    all_files = sorted(texts_dir.glob("doc_*.txt"))
    if not all_files:
        # Fallback: try any .txt file (e.g. ReDocRED-style names)
        all_files = sorted(texts_dir.glob("*.txt"))
    if n is None:
        picked = all_files
    else:
        if len(all_files) < n:
            raise ValueError(
                f"Requested {n} docs but only {len(all_files)} found in {texts_dir}"
            )
        rng = random.Random(seed)
        picked = rng.sample(all_files, n)
    docs = {}
    for p in picked:
        docs[p.stem] = p.read_text(encoding="utf-8")
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Gemini-batched KG pipeline over MINE docs"
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=None,
        help="Number of docs to sample. Omit (or use --all) to process all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every doc in --texts-dir.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (ignored when --all)"
    )
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=Path("<PROJECT_ROOT>/graph_rag/datasets/MINE/texts"),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: derived from --texts-dir parent folder).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build input .jsonl files locally and report counts; do not submit.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit(
            "ERROR: GEMINI_API_KEY not found in environment. "
            "Set it via .env or export."
        )

    n_docs: Optional[int] = None if args.all else args.docs
    if n_docs is None and not args.all:
        parser.error("Specify --docs N or --all")

    # Resolve experiment name
    if args.name:
        experiment_name = args.name
    else:
        # Derive from the texts directory's parent folder name
        dataset_name = args.texts_dir.resolve().parent.name
        experiment_name = f"{dataset_name}_gemini_batch"

    config = dict(CONFIG)
    config["experiment_name"] = experiment_name

    docs = select_docs(args.texts_dir, n_docs, args.seed)
    print(f"Dataset:     {args.texts_dir}")
    print(f"Experiment:  {experiment_name}")
    print(
        f"Selected {len(docs)} docs: "
        f"{list(docs.keys())[:10]}{'...' if len(docs) > 10 else ''}"
    )
    for doc_id, text in docs.items():
        print(f"  {doc_id}: {len(text)} chars, ~{len(text.split())} words")

    pipeline = GeminiBatchKGPipeline(config=config)
    pipeline.run(docs, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
