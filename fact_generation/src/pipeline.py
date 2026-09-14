"""
Atomic Fact Generation Pipeline — Orchestrator.

Usage:
    python -m src.pipeline --stage all                  # Run all stages
    python -m src.pipeline --stage 1                    # Run Stage 1 only
    python -m src.pipeline --stage 2                    # Run Stage 2 only
    python -m src.pipeline --stage 3                    # Run Stage 3 only
    python -m src.pipeline --stage 4                    # Run Stage 4 only
    python -m src.pipeline --stage all --limit 5        # Smoke test (5 docs)
    python -m src.pipeline --stage all --dataset MINE   # Name the dataset
    python -m src.pipeline --stage all --input-dir ../datasets/MINE/texts  # Custom input
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger("atomic_facts")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_config_hash(config: Dict[str, Any], generator_prompt: str, verifier_prompt: str) -> str:
    """Compute a deterministic hash of the config + prompts for reproducibility."""
    blob = json.dumps(config, sort_keys=True) + generator_prompt + verifier_prompt
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def get_git_commit() -> str:
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_run_config(config: Dict[str, Any], config_hash: str, output_path: str):
    """Save the full run config for reproducibility."""
    run_config = {
        "generator_model": config["generator"]["model_version"],
        "generator_temperature": config["generator"]["temperature"],
        "generator_reasoning_effort": config["generator"].get("reasoning_effort", "high"),
        "verifier_model": config["verifier"]["model_version"],
        "verifier_temperature": config["verifier"]["temperature"],
        "target_facts_per_doc": config["pipeline"]["target_facts_per_doc"],
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    logger.info(f"Run config saved: {output_path} (hash={config_hash})")


def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Atomic Fact Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["1", "2", "3", "4", "all"],
        help="Which stage(s) to run: 1, 2, 3, 4, or all",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override input directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N documents (for smoke tests)",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name for reporting (e.g., MINE, scierc)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Resolve paths relative to the script's directory (fact_generation/)
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / args.config
    config = load_config(str(config_path))
    paths = config["paths"]

    input_dir = args.input_dir or str(base_dir / paths["inputs"])
    stage1_dir = str(base_dir / paths["stage1_candidates"])
    stage2_dir = str(base_dir / paths["stage2_verified"])
    final_dir = str(base_dir / paths["stage3_final"])
    audit_dir = str(base_dir / paths["stage3_audit"])
    stats_path = str(base_dir / paths["dataset_stats"])
    run_config_path = str(base_dir / paths["run_config"])

    generator_prompt_path = str(base_dir / "prompts" / "generator.txt")
    verifier_prompt_path = str(base_dir / "prompts" / "verifier.txt")

    # Load prompts for config hash
    with open(generator_prompt_path, "r", encoding="utf-8") as f:
        gen_prompt = f.read()
    with open(verifier_prompt_path, "r", encoding="utf-8") as f:
        ver_prompt = f.read()
    config_hash = compute_config_hash(config, gen_prompt, ver_prompt)

    # Save run config
    save_run_config(config, config_hash, run_config_path)

    stages_to_run = (
        ["1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    )

    logger.info(f"Pipeline starting: stages={stages_to_run}, limit={args.limit}, dataset={args.dataset or '(unnamed)'}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Config hash: {config_hash}")

    # ── Stage 1 ──
    if "1" in stages_to_run:
        logger.info("=" * 60)
        logger.info("STAGE 1 — Generate candidate atomic facts (GPT-5)")
        logger.info("=" * 60)
        from .generate import run_stage1
        run_stage1(
            input_dir=input_dir,
            output_dir=stage1_dir,
            config=config,
            prompt_path=generator_prompt_path,
            limit=args.limit,
        )

    # ── Stage 2 ──
    if "2" in stages_to_run:
        logger.info("=" * 60)
        logger.info("STAGE 2 — Cross-family verification (Gemini)")
        logger.info("=" * 60)
        from .verify import run_stage2
        run_stage2(
            input_dir=input_dir,
            stage1_dir=stage1_dir,
            output_dir=stage2_dir,
            config=config,
            prompt_path=verifier_prompt_path,
            limit=args.limit,
        )

    # ── Stage 3 ──
    if "3" in stages_to_run:
        logger.info("=" * 60)
        logger.info("STAGE 3 — Filtering & final fact list")
        logger.info("=" * 60)
        from .filter import run_stage3
        run_stage3(
            stage2_dir=stage2_dir,
            final_dir=final_dir,
            audit_dir=audit_dir,
            config=config,
            limit=args.limit,
        )

    # ── Stage 4 ──
    if "4" in stages_to_run:
        logger.info("=" * 60)
        logger.info("STAGE 4 — Aggregate dataset-level statistics")
        logger.info("=" * 60)
        from .aggregate import run_stage4
        stats = run_stage4(
            stage1_dir=stage1_dir,
            audit_dir=audit_dir,
            final_dir=final_dir,
            stats_output_path=stats_path,
            config=config,
            dataset_name=args.dataset,
        )
        if stats:
            logger.info(f"Dataset stats:\n{json.dumps(stats, indent=2)}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
