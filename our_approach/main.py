"""
Knowledge Graph creation from raw text — example usage.

All configuration is passed as a single dict to run_pipeline().
See pipeline.DEFAULT_CONFIG for all available keys and their defaults.
"""

from pipeline import run_pipeline


if __name__ == "__main__":

    from pathlib import Path

    text = "A butterfly's life cycle has four stages: egg, larva, pupa, and adult. Female butterflies lay eggs on host plants, usually on the underside of leaves. After hatching, the larva, called a caterpillar, feeds on leaves and grows rapidly, shedding its skin through a process known as molting. The caterpillar then attaches to a stem and forms a chrysalis, inside which it undergoes metamorphosis into a butterfly. The adult butterfly feeds on nectar through a tubular mouthpart called a proboscis, and pollinates flowers by transferring pollen between them. Adult butterflies typically live a few days to several weeks. Females then lay eggs on host plants, beginning the cycle again."
    config = {
        # LLM
        "model": "gemini-3.1-pro-preview",
        "model_type": "gemini",

        # Pipeline
        "sentences_per_chunk": 4,
        "summarize_text": True,
        "entity_refinement_mode": "deterministic",

        # Output
        "experiment_name": "walkthrough_example",
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
