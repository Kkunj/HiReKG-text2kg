"""
A/B test: does Nemotron `/think` mode actually help KG extraction?

Runs the kggen pipeline N times with thinking ON and N times with thinking OFF
(controlled by KG_DISABLE_THINKING env var -> `/no_think` system prefix).
Collects per-call and per-run metrics and prints a verdict.

Run:
    cd baselines/kggen
    python ab_test_thinking.py
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

_KGGEN_DIR = Path(__file__).resolve().parent
_OUR_APPROACH_DIR = _KGGEN_DIR.parent.parent / "our_approach"
sys.path.insert(0, str(_KGGEN_DIR))
sys.path.append(str(_OUR_APPROACH_DIR))

import llm_client_abtest
sys.modules["llm_client"] = llm_client_abtest

from pipeline import run_pipeline

TEXT = Path(r"C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\texts\Asus_VivoTab.txt").read_text(encoding="utf-8")
N_PER_ARM = 3
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MAX_OUTPUT_TOKENS = 10000


def run_once(arm: str, run_idx: int, disable_thinking: bool) -> dict:
    os.environ["KG_DISABLE_THINKING"] = "1" if disable_thinking else "0"
    llm_client_abtest.CALL_METRICS.clear()

    t0 = time.time()
    error = None
    entities_refined = 0
    triples_final = []
    triples_raw = 0
    try:
        result = run_pipeline(TEXT, {
            "model": MODEL,
            "model_type": "local",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "chunk_size": 5000,
            "context": "Technology company and products",
            "deduplication": "none",
            "experiment_name": f"abtest_{arm}_run{run_idx}",
            "save_experiments": True,
            "log_to_console": False,
            "log_to_file": True,
        })
        entities_refined = len(result["entities_refined"])
        triples_final = result["triples_final"]
        triples_raw = len(result["triples_final"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    wall = time.time() - t0
    calls = list(llm_client_abtest.CALL_METRICS)

    # Repetition heuristic: how many final triples are unique?
    triple_keys = [(t["subject"], t["relation"], t["object"]) for t in triples_final]
    unique_triples = len(set(triple_keys))

    return {
        "arm": arm,
        "run_idx": run_idx,
        "wall_s": round(wall, 1),
        "error": error,
        "entities": entities_refined,
        "triples": triples_raw,
        "unique_triples": unique_triples,
        "n_llm_calls": len(calls),
        "truncated_calls": sum(1 for c in calls if c["finish_reason"] == "length"),
        "total_completion_tokens": sum((c.get("completion_tokens") or 0) for c in calls),
        "calls": calls,
    }


def summarize(arm_name: str, runs: list[dict]) -> dict:
    wall = [r["wall_s"] for r in runs]
    ents = [r["entities"] for r in runs]
    trips = [r["triples"] for r in runs]
    uniq = [r["unique_triples"] for r in runs]
    trunc = [r["truncated_calls"] for r in runs]
    tot_tokens = [r["total_completion_tokens"] for r in runs]
    n_errors = sum(1 for r in runs if r["error"])

    def mstd(xs):
        if not xs:
            return "-"
        m = statistics.mean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return f"{m:.1f} ± {s:.1f}"

    return {
        "arm": arm_name,
        "runs": len(runs),
        "errors": n_errors,
        "wall_s": mstd(wall),
        "entities": mstd(ents),
        "triples": mstd(trips),
        "unique_triples": mstd(uniq),
        "truncated_calls_per_run": mstd(trunc),
        "total_completion_tokens": mstd(tot_tokens),
        "raw_wall": wall,
        "raw_triples": trips,
    }


def main() -> None:
    print("=" * 70)
    print(f"A/B TEST: Nemotron thinking ON vs OFF ({N_PER_ARM} runs each)")
    print(f"Model: {MODEL}")
    print(f"Input: {len(TEXT)} chars (Asus_VivoTab.txt)")
    print("=" * 70)

    results = {"think_on": [], "think_off": []}

    for arm, disable in [("think_on", False), ("think_off", True)]:
        print(f"\n--- ARM: {arm} (disable_thinking={disable}) ---")
        for i in range(N_PER_ARM):
            print(f"  Run {i+1}/{N_PER_ARM}... ", end="", flush=True)
            r = run_once(arm, i, disable)
            results[arm].append(r)
            status = "ERROR" if r["error"] else "ok"
            print(
                f"{status}  wall={r['wall_s']}s  ents={r['entities']}  "
                f"trips={r['triples']}  trunc_calls={r['truncated_calls']}/{r['n_llm_calls']}"
            )
            if r["error"]:
                print(f"      error: {r['error']}")

    # Dump raw results to a JSON file for inspection
    out_path = _KGGEN_DIR / "experiments" / "abtest_thinking_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    # Summaries
    sum_on = summarize("think_on", results["think_on"])
    sum_off = summarize("think_off", results["think_off"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'metric':<30} | {'think_on':>20} | {'think_off':>20}"
    print(header)
    print("-" * len(header))
    for key in ["runs", "errors", "wall_s", "entities", "triples",
                "unique_triples", "truncated_calls_per_run", "total_completion_tokens"]:
        print(f"{key:<30} | {str(sum_on[key]):>20} | {str(sum_off[key]):>20}")

    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
