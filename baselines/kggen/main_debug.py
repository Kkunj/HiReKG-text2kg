"""
Debug runner for KGGen + self-hosted Nemotron.

Uses `llm_client_debug.py` (not `llm_client.py`) so the main client stays clean.
The debug client dumps the raw pre-strip LLM response and `finish_reason` so we
can see whether Nemotron is truncating, looping, or emitting runaway reasoning.

Run:
    cd baselines/kggen
    python main_debug.py
"""

import sys
from pathlib import Path

_KGGEN_DIR = Path(__file__).resolve().parent
_OUR_APPROACH_DIR = _KGGEN_DIR.parent.parent / "our_approach"

# kggen dir first so `from pipeline import ...` resolves to kggen/pipeline.py
# (our_approach/ also has a pipeline.py — without this we'd import the wrong one).
sys.path.insert(0, str(_KGGEN_DIR))
sys.path.append(str(_OUR_APPROACH_DIR))

import llm_client_debug
sys.modules["llm_client"] = llm_client_debug

from pipeline import run_pipeline


if __name__ == "__main__":
    text = Path(r"C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\texts\Asus_VivoTab.txt").read_text(encoding="utf-8")

    config = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "model_type": "local",
        "max_output_tokens": 10000,
        "temperature": 0.0,

        "chunk_size": 5000,
        "context": "Technology company and products",
        "deduplication": "semhash",

        "experiment_name": "kggen_exp_local_debug",
        "save_experiments": True,

        "log_to_console": True,
        "log_to_file": True,
    }

    results = run_pipeline(text, config)
    print(f"\nEntities: {len(results['entities_refined'])}")
    print(f"Triples:  {len(results['triples_final'])}")
