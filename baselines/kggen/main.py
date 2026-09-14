"""
KGGen Baseline — Knowledge Graph Construction from Text

All configuration is passed as a single dict to run_pipeline().
See pipeline.DEFAULT_CONFIG for all available keys and their defaults.
"""

from pipeline import run_pipeline


if __name__ == "__main__":

    from pathlib import Path

    text =  "PASTE TEXT HERE"
    config = {
        # LLM
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "model_type": "local",

        # Pipeline
        "chunk_size": 4000,
        "context": "Technology company and products",
        "deduplication": "semhash",

        # Output
        "experiment_name": "kggen_exp_local_qwen",
        "save_experiments": True,

        # Logging
        "log_to_console": True,
        "log_to_file": True,
    }

    results = run_pipeline(text, config)

    # Quick summary
    print(f"\nEntities: {len(results['entities_refined'])}")
    print(f"Triples:  {len(results['triples_final'])}")
    for t in results["triples_final"][:10]:
        print(f"  ({t['subject']}) -[{t['relation']}]-> ({t['object']})")
