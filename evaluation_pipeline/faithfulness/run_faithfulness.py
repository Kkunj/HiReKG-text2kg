"""
Generalized faithfulness runner.

Computes per-doc and per-dataset faithfulness scores for ANY combination of
(approach, dataset). Configure the run by editing the constants in the
"MANUAL CONFIG" block below — no CLI args required for the selection itself.

Layout assumptions:
    Approach experiment folders (each contains `<doc_id>/final_output.json`):
        OURS  : graph_rag/our_approach/experiments/<experiment_folder>/
        RAKG  : graph_rag/baselines/rakg/experiments/<experiment_folder>/
        KGGEN : graph_rag/baselines/kggen/experiments/<experiment_folder>/

    Dataset source texts (one .txt per doc_id):
        MINE     : graph_rag/datasets/MINE/texts/<doc_id>.txt
        SCIERC   : graph_rag/datasets/scierc/texts/<doc_id>.txt
        REDOCRED : graph_rag/datasets/windows_redocred/texts/<doc_id>.txt

The doc_id used to read the source text is taken verbatim from the experiment
folder's subdirectory name (e.g. "doc_42" or "Asus_VivoTab"), and must match a
"<doc_id>.txt" file under the dataset's texts directory.

All three approaches store extracted triples under `triples_final` in their
final_output.json with the same {subject, relation, object[, evidence]}
schema, so a single loader works across approaches.

Outputs go to:
    results/<approach>_<dataset>_<experiment_folder>/
        audit_<doc_id>.json
        per_doc_summary.json
        dataset_summary.json

Resumable: the OpenAI batches and embedding cache are scoped per
(approach, experiment_folder) — re-running picks up where it left off. Pass
--force-rerun to wipe both.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — make src/ + the project's batch client importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent                 # graph_rag/
_OUR_APPROACH_DIR = _REPO_ROOT / "our_approach"

for p in (str(_THIS_DIR), str(_OUR_APPROACH_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.pipeline import run_pipeline, make_logger  # noqa: E402


# ===========================================================================
# MANUAL CONFIG — edit these to choose what to evaluate.
# ===========================================================================
APPROACH = "KGGEN"          # one of: "OURS", "RAKG", "KGGEN"
DATASET = "MINE"            # one of: "MINE", "SCIERC", "REDOCRED"

# Tag used to build the experiment folder name when EXPERIMENT_FOLDER is None.
# Examples that exist in this repo:
#   "qwen3_14b", "qwen3_14b_batch", "qwen36"
MODEL_TAG = "qwen3_14b"

# Optional explicit override of the experiment folder name. If None, computed
# as f"{prefix}_{dataset_name}_{MODEL_TAG}" (e.g. "our_mine_qwen3_14b").
# Set this when the auto-derived name doesn't match (e.g. "_batch" suffix).
EXPERIMENT_FOLDER: Optional[str] = "kggen_mine_temp_faithfulness"

# --- Execution backend ----------------------------------------------------
# "openai" → OpenAI Batch API (gpt-5-mini verbalize + gpt-5/high judge).
# "local"  → self-hosted LLM via our_approach.llm_client.LLMClient
#            (model_type="local"). Reads endpoint from LOCAL_LLM_URL env var
#            (default http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions). Both
#            verbalize and judge run synchronously through this endpoint
#            with ThreadPoolExecutor concurrency.
BACKEND = "local"

# Used only when BACKEND == "local". Mirrors our_approach/run_mine_batch.py.
LOCAL_MODEL = "qwen3-14b"
LOCAL_MAX_WORKERS = 3

# Optional override of the local endpoint for this run (parallel vLLM ports).
# When None, LLMClient uses LOCAL_LLM_URL from env.
LOCAL_LLM_URL_OVERRIDE: Optional[str] = "http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions"
# ===========================================================================


APPROACH_REGISTRY: Dict[str, Dict[str, object]] = {
    "OURS":  {
        "root": _REPO_ROOT / "our_approach" / "experiments",
        "prefix": "our",
    },
    "RAKG":  {
        "root": _REPO_ROOT / "baselines" / "rakg" / "experiments",
        "prefix": "rakg",
    },
    "KGGEN": {
        "root": _REPO_ROOT / "baselines" / "kggen" / "experiments",
        "prefix": "kggen",
    },
}

DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "MINE":     {
        "name": "mine",
        "texts_dir": _REPO_ROOT / "datasets" / "MINE" / "texts",
    },
    "SCIERC":   {
        "name": "scierc",
        "texts_dir": _REPO_ROOT / "datasets" / "scierc" / "texts",
    },
    "REDOCRED": {
        "name": "redocred",
        "texts_dir": _REPO_ROOT / "datasets" / "windows_redocred" / "texts",
    },
}


def resolve_paths(
    approach: str = APPROACH,
    dataset: str = DATASET,
    experiment_folder: Optional[str] = EXPERIMENT_FOLDER,
    model_tag: str = MODEL_TAG,
) -> Tuple[Path, Path, str]:
    """Return (experiments_root_for_this_run, texts_dir, run_tag)."""
    if approach not in APPROACH_REGISTRY:
        raise ValueError(
            f"Unknown approach={approach!r}. "
            f"Choose from {sorted(APPROACH_REGISTRY)}."
        )
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset={dataset!r}. "
            f"Choose from {sorted(DATASET_REGISTRY)}."
        )

    a = APPROACH_REGISTRY[approach]
    d = DATASET_REGISTRY[dataset]

    folder = experiment_folder or (
        f"{a['prefix']}_{d['name']}_{model_tag}"
    )
    experiments_root = a["root"] / folder              # type: ignore[operator]
    texts_dir = d["texts_dir"]                         # type: ignore[assignment]
    run_tag = f"{approach.lower()}_{dataset.lower()}_{folder}"
    return experiments_root, texts_dir, run_tag        # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Loaders — single adapter works for all three approaches because they all
# emit `triples_final` in final_output.json with the same schema.
# ---------------------------------------------------------------------------
def discover_doc_ids(experiments_root: Path) -> List[str]:
    """Every subdirectory of experiments_root that has a final_output.json."""
    if not experiments_root.exists():
        raise FileNotFoundError(
            f"Experiments root does not exist: {experiments_root}\n"
            f"Check APPROACH / DATASET / MODEL_TAG / EXPERIMENT_FOLDER."
        )
    out: List[str] = []
    for child in sorted(experiments_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "final_output.json").exists():
            out.append(child.name)
    return out


def load_triples_for_doc(experiments_root: Path, doc_id: str) -> List[Dict]:
    final_path = experiments_root / doc_id / "final_output.json"
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    return payload.get("triples_final", []) or []


def load_source_text(texts_dir: Path, doc_id: str) -> Optional[str]:
    txt_path = texts_dir / f"{doc_id}.txt"
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding="utf-8")


def load_inputs(
    experiments_root: Path,
    texts_dir: Path,
    limit: Optional[int],
    logger: logging.Logger,
) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    doc_ids = discover_doc_ids(experiments_root)
    if limit is not None:
        doc_ids = doc_ids[:limit]
    logger.info(f"[load] discovered {len(doc_ids)} docs in {experiments_root.name}")

    triples_per_doc: Dict[str, List[Dict]] = {}
    source_texts: Dict[str, str] = {}
    n_skipped = 0
    for doc_id in doc_ids:
        triples = load_triples_for_doc(experiments_root, doc_id)
        if not triples:
            logger.warning(f"[load] {doc_id}: empty triples_final, skipping")
            n_skipped += 1
            continue
        text = load_source_text(texts_dir, doc_id)
        if text is None:
            logger.warning(
                f"[load] {doc_id}: missing {texts_dir / (doc_id + '.txt')}, skipping"
            )
            n_skipped += 1
            continue
        triples_per_doc[doc_id] = triples
        source_texts[doc_id] = text

    logger.info(
        f"[load] loaded={len(triples_per_doc)} docs with triples, skipped={n_skipped}"
    )
    return triples_per_doc, source_texts


# ---------------------------------------------------------------------------
# CLI — only knobs are operational ones (limit, force-rerun, model overrides).
# Approach / dataset selection lives in the MANUAL CONFIG block above.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute faithfulness for an (approach, dataset) pair. "
            "Defaults come from the MANUAL CONFIG block at the top of this file; "
            "any of them can be overridden on the command line."
        )
    )
    parser.add_argument(
        "--approach", default=APPROACH, choices=sorted(APPROACH_REGISTRY),
        help=f"Which extractor's KGs to evaluate. Default: {APPROACH}.",
    )
    parser.add_argument(
        "--dataset", default=DATASET, choices=sorted(DATASET_REGISTRY),
        help=f"Source-text dataset. Default: {DATASET}.",
    )
    parser.add_argument(
        "--experiment-folder", default=EXPERIMENT_FOLDER,
        help="Override experiment folder name under the approach's experiments/ root. "
             f"Default: {EXPERIMENT_FOLDER!r}.",
    )
    parser.add_argument(
        "--model-tag", default=MODEL_TAG,
        help="Tag used to auto-derive the experiment folder name when "
             "--experiment-folder is not set.",
    )
    parser.add_argument(
        "--run-tag-suffix", default=None,
        help="Optional extra suffix appended to the run tag (e.g. 'v2'). "
             "Used to keep a re-run's results/workdir/cache separate from a "
             "previous run with otherwise identical config.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N docs (smoke test).",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Delete this run's batch_workdir + embeddings_cache before running.",
    )
    parser.add_argument(
        "--backend", default=BACKEND, choices=["local", "openai"],
        help=f"Execution backend. Default: {BACKEND}.",
    )
    parser.add_argument("--verbalize-model", default=None,
                        help="Override verbalize model. Defaults: openai=gpt-5-mini, local=LOCAL_MODEL.")
    parser.add_argument("--verbalize-reasoning", default="low")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model. Defaults: openai=gpt-5, local=LOCAL_MODEL.")
    parser.add_argument("--judge-reasoning", default="high")
    parser.add_argument("--retrieval-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=LOCAL_MAX_WORKERS,
                        help="Concurrent worker threads when BACKEND=local.")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Override LOCAL_LLM_URL for this run "
                             "(e.g. http://<LOCAL_LLM_HOST>:8084/v1/chat/completions).")
    args = parser.parse_args()

    backend = args.backend

    # Backend-specific env setup. For local, set LOCAL_LLM_URL before any
    # code path imports llm_client (which reads it at class-definition time).
    if backend == "local":
        endpoint = args.endpoint or LOCAL_LLM_URL_OVERRIDE
        if endpoint:
            os.environ["LOCAL_LLM_URL"] = endpoint
    else:
        if os.getenv("OPENAI_API_KEY_2"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]

    if backend == "local":
        verbalize_model = args.verbalize_model or LOCAL_MODEL
        judge_model = args.judge_model or LOCAL_MODEL
    else:
        verbalize_model = args.verbalize_model or "gpt-5-mini"
        judge_model = args.judge_model or "gpt-5"

    experiments_root, texts_dir, run_tag = resolve_paths(
        approach=args.approach,
        dataset=args.dataset,
        experiment_folder=args.experiment_folder,
        model_tag=args.model_tag,
    )
    run_tag = f"{run_tag}__{backend}"
    if args.run_tag_suffix:
        run_tag = f"{run_tag}__{args.run_tag_suffix}"

    prompts_dir = _THIS_DIR / "prompts"
    workdir = _THIS_DIR / "batch_workdir" / run_tag
    results_dir = _THIS_DIR / "results" / run_tag
    cache_dir = _THIS_DIR / "embeddings_cache" / args.dataset.lower()

    if args.force_rerun:
        import shutil
        for d in (workdir, cache_dir):
            if d.exists():
                shutil.rmtree(d)

    workdir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(workdir, name=f"Faithfulness[{run_tag}]")
    logger.info("=" * 70)
    logger.info(f"Faithfulness run: APPROACH={args.approach} DATASET={args.dataset} BACKEND={backend}")
    logger.info(f"  experiments_root = {experiments_root}")
    logger.info(f"  texts_dir        = {texts_dir}")
    logger.info(f"  results_dir      = {results_dir}")
    if backend == "local":
        logger.info(
            f"  endpoint={os.environ.get('LOCAL_LLM_URL', '<default in llm_client>')} "
            f"workers={args.workers}"
        )
        logger.info(f"  verbalize={verbalize_model} judge={judge_model}")
    else:
        logger.info(
            f"  verbalize={verbalize_model}/{args.verbalize_reasoning} "
            f"judge={judge_model}/{args.judge_reasoning}"
        )
    logger.info("=" * 70)

    triples_per_doc, source_texts = load_inputs(
        experiments_root, texts_dir, args.limit, logger,
    )
    if not triples_per_doc:
        logger.error("No documents to evaluate. Exiting.")
        sys.exit(1)

    run_pipeline(
        triples_per_doc=triples_per_doc,
        source_texts=source_texts,
        prompts_dir=prompts_dir,
        workdir=workdir,
        results_dir=results_dir,
        cache_dir=cache_dir,
        logger=logger,
        verbalize_model=verbalize_model,
        verbalize_reasoning=args.verbalize_reasoning,
        judge_model=judge_model,
        judge_reasoning=args.judge_reasoning,
        retrieval_model=args.retrieval_model,
        top_k=args.top_k,
        backend=backend,
        local_max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
