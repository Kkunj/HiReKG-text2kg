"""
Entry point for batched KG pipeline over the MINE dataset.

Picks N random docs from datasets/MINE/texts, runs the batched pipeline,
and writes per-doc experiment folders under experiments/<experiment_name>/<doc_id>/.

Usage:
    python run_mine_batch.py --all                  # process every doc in --texts-dir
    python run_mine_batch.py --docs 3 --seed 7      # sample 3 random docs (deterministic)
    python run_mine_batch.py --all --dry-run        # validate without submitting
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent
if str(_OUR_APPROACH_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH_DIR))

from batch.batch_pipeline import BatchKGPipeline


CONFIG = {
    # --- LLM ---
    "model": "gpt-4o-2024-11-20",
    "model_type": "openai",
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
    "embedding_backend": "openai",
    "embedding_model": "text-embedding-3-large",
    "cluster_distance_threshold": 0.99,
    "min_cluster_size": 2,

    # --- Storage ---
    "store_to_neo4j": False,

    # --- Experiment output ---
    "save_experiments": True,
    "experiment_name": "MINE_batch_results",

    # --- Logging ---
    "log_to_console": True,
    "log_to_file": True,
}


def select_docs(texts_dir: Path, n: Optional[int], seed: int) -> dict:
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
    parser = argparse.ArgumentParser(description="Batched KG pipeline over MINE docs")
    parser.add_argument(
        "--docs",
        type=int,
        default=None,
        help="Number of docs to sample. Omit (or use --all) to process every doc in --texts-dir.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every doc in --texts-dir (equivalent to omitting --docs).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (ignored when processing all docs)")
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=Path("<PROJECT_ROOT>/graph_rag/datasets/MINE/texts"),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override CONFIG['experiment_name'] (output folder name under experiments/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override CONFIG['model'] (e.g. gpt-4o).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build input .jsonl files locally and report counts; do not submit.",
    )
    args = parser.parse_args()

    if args.experiment_name:
        CONFIG["experiment_name"] = args.experiment_name
    if args.model:
        CONFIG["model"] = args.model

    import os
    from dotenv import load_dotenv
    load_dotenv()

    # This run uses the secondary API key. Promote OPENAI_API_KEY_2 → OPENAI_API_KEY
    # so every downstream client (batch, sync LLM, embeddings, object resolution)
    # picks it up without modifying any sequential-pipeline code.
    if os.getenv("OPENAI_API_KEY_2"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]
        print("Using OPENAI_API_KEY_2 for this run.")
    else:
        print("WARNING: OPENAI_API_KEY_2 not found in environment; using OPENAI_API_KEY.")

    n_docs: Optional[int] = None if args.all else args.docs
    docs = select_docs(args.texts_dir, n_docs, args.seed)
    print(f"Selected {len(docs)} docs: {list(docs.keys())[:10]}{'...' if len(docs) > 10 else ''}")
    for doc_id, text in docs.items():
        print(f"  {doc_id}: {len(text)} chars, ~{len(text.split())} words")

    pipeline = BatchKGPipeline(config=CONFIG)
    pipeline.run(docs, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
