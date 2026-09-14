"""
RAKG Baseline — Retrieval Augmented Knowledge Graph Construction

All configuration is passed as a single dict to run_pipeline().
See pipeline.DEFAULT_CONFIG for all available keys and their defaults.
"""

from pipeline import run_pipeline


if __name__ == "__main__":

    from pathlib import Path

    text = "PASTE TEXT HERE"
    config = {
        # LLM
        "model": "gemini-3.1-pro-preview",
        "model_type": "gemini",

        # Embeddings (local BGE-M3 by default, no API needed)
        "embedding_model": "BAAI/bge-m3",
        "embedding_backend": "local",

        # Pipeline
        "similarity_threshold": 0.60,
        "retrieval_top_k": 5,

        # Output
        "experiment_name": "rakg_exp_1",
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
