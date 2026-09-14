"""
Aggregate LLM token usage from pipeline logs across all KGgen / RAKG /
our_approach experiment directories.

For each experiment directory, this script:
    1. Recursively walks for `*.log` files (handles both layouts found in
       this codebase: per-doc logs inside doc_* / Wikipedia-named folders,
       and top-level batch logs).
    2. Extracts every line matching the token-usage pattern
            ... | DEBUG | [<Provider>] Tokens - Input: <N>, Output: <M>
       (Provider is `SelfHosted` for Qwen runs, `OpenAI` for GPT-4o.)
    3. Sums input tokens, output tokens, and call count, broken down by
       provider where present.

Note:
    The `kggen-graphs-gpt-4o` experiment is intentionally excluded -- it
    used the standalone KGgen library and isn't part of this token audit.

    Some GPT-4o experiment runs only emit INFO-level logs at the top
    level, with no DEBUG token lines available. Those are reported as
    `status: "no_token_logs"` with zero counts so they're visible in the
    output rather than silently dropped.

Output:
    structural/llm_token_usage_<timestamp>.json
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


STRUCTURAL_DIR = Path(__file__).resolve().parent


# Pattern: capture provider, input tokens, output tokens
# Examples it must match:
#   2026-05-06 13:47:29 | DEBUG    | [SelfHosted] Tokens - Input: 671, Output: 1315
#   2026-05-16 22:13:08 | DEBUG    | [OpenAI] Tokens - Input: 255, Output: 179
TOKEN_LINE_RE = re.compile(
    r"\[(?P<provider>[A-Za-z0-9_\-]+)\]\s+Tokens\s*[-:]\s*"
    r"Input[:\s]+(?P<inp>\d+)[,\s]+Output[:\s]+(?P<out>\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Experiment directory groups (kggen-graphs-gpt-4o intentionally omitted)
# ---------------------------------------------------------------------------

APPROACH_DIRS: Dict[str, List[str]] = {
    "kggen": [
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_mine_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_mine_qwen3_14b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_qwen3_14b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_scierc_qwen3_14b",
    ],
    "rakg": [
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\RAKG_MINE_gpt-4o",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_mine_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_mine_qwen3_14b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_redocred_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_redocred_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_redocred_qwen3_14b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_scierc_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_scierc_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_scierc_qwen3_14b",
    ],
    "our_approach": [
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\MINE_batch_results",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_14b_batch",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_redocred_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_14b",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_scierc_gpt4o",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_8b",
        r"C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_scierc_qwen3_14b",
    ],
}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def scan_experiment(experiment_dir: str) -> Dict[str, Any]:
    """
    Recursively scan all .log files under `experiment_dir` and aggregate
    LLM token usage.

    Returns a dict with totals and per-provider breakdown. If no .log files
    are found OR no token lines match in any of them, the returned dict
    carries an explanatory `status` field so the caller can flag it.
    """
    experiment_name = os.path.basename(experiment_dir.rstrip("\\/"))

    if not os.path.isdir(experiment_dir):
        return {
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "status": "missing_directory",
            "log_files_scanned": 0,
            "llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "by_provider": {},
        }

    log_paths: List[str] = []
    for root, _dirs, files in os.walk(experiment_dir):
        for f in files:
            if f.endswith(".log"):
                log_paths.append(os.path.join(root, f))

    if not log_paths:
        return {
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "status": "no_log_files",
            "log_files_scanned": 0,
            "llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "by_provider": {},
        }

    by_provider: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    total_calls = 0
    total_in = 0
    total_out = 0
    files_with_matches = 0
    bad_files: List[Dict[str, str]] = []

    for lp in log_paths:
        try:
            # errors="replace" because some pipeline logs contain non-utf8
            # bytes from model outputs; we don't want a single bad byte to
            # abort an otherwise-valid scan.
            with open(lp, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception as exc:
            bad_files.append({"path": lp, "error": str(exc)})
            continue

        matched_in_file = 0
        for m in TOKEN_LINE_RE.finditer(text):
            provider = m.group("provider")
            inp = int(m.group("inp"))
            out = int(m.group("out"))
            by_provider[provider]["calls"] += 1
            by_provider[provider]["input_tokens"] += inp
            by_provider[provider]["output_tokens"] += out
            total_calls += 1
            total_in += inp
            total_out += out
            matched_in_file += 1
        if matched_in_file > 0:
            files_with_matches += 1

    status = "ok" if total_calls > 0 else "no_token_logs"
    return {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "status": status,
        "log_files_scanned": len(log_paths),
        "log_files_with_token_lines": files_with_matches,
        "llm_calls": total_calls,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "by_provider": {
            p: {**v, "total_tokens": v["input_tokens"] + v["output_tokens"]}
            for p, v in by_provider.items()
        },
        "bad_files": bad_files,
    }


# ---------------------------------------------------------------------------
# Approach + grand aggregation
# ---------------------------------------------------------------------------

def aggregate_experiments(exp_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sum totals across all experiments in one approach."""
    total = {
        "experiments_count": len(exp_results),
        "experiments_with_data": 0,
        "experiments_missing": [],
        "experiments_no_token_logs": [],
        "llm_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "by_provider": defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        ),
    }
    for name, res in exp_results.items():
        if res["status"] == "missing_directory":
            total["experiments_missing"].append(name)
            continue
        if res["status"] == "no_token_logs":
            total["experiments_no_token_logs"].append(name)
            continue
        if res["status"] != "ok":
            continue
        total["experiments_with_data"] += 1
        total["llm_calls"] += res["llm_calls"]
        total["total_input_tokens"] += res["total_input_tokens"]
        total["total_output_tokens"] += res["total_output_tokens"]
        total["total_tokens"] += res["total_tokens"]
        for prov, counts in res["by_provider"].items():
            total["by_provider"][prov]["calls"] += counts["calls"]
            total["by_provider"][prov]["input_tokens"] += counts["input_tokens"]
            total["by_provider"][prov]["output_tokens"] += counts["output_tokens"]

    total["by_provider"] = {
        p: {**v, "total_tokens": v["input_tokens"] + v["output_tokens"]}
        for p, v in total["by_provider"].items()
    }
    return total


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(n: int) -> str:
    return f"{n:,}"


def print_table(approach_results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("PER-EXPERIMENT TOKEN USAGE")
    print("=" * 92)
    header = f"  {'Experiment':<34} {'Calls':>10} {'InTokens':>14} {'OutTokens':>14} {'Total':>14}"
    for approach, block in approach_results.items():
        print(f"\n[{approach}]")
        print(header)
        print("  " + "-" * 88)
        for name, res in block["experiments"].items():
            if res["status"] != "ok":
                print(f"  {name:<34} {'--':>10} {res['status']:>14}")
                continue
            print(
                f"  {name:<34} "
                f"{_fmt(res['llm_calls']):>10} "
                f"{_fmt(res['total_input_tokens']):>14} "
                f"{_fmt(res['total_output_tokens']):>14} "
                f"{_fmt(res['total_tokens']):>14}"
            )
        agg = block["aggregate"]
        print("  " + "-" * 88)
        print(
            f"  {'TOTAL ('+approach+')':<34} "
            f"{_fmt(agg['llm_calls']):>10} "
            f"{_fmt(agg['total_input_tokens']):>14} "
            f"{_fmt(agg['total_output_tokens']):>14} "
            f"{_fmt(agg['total_tokens']):>14}"
        )
        if agg["experiments_missing"]:
            print(f"  Missing dirs           : {agg['experiments_missing']}")
        if agg["experiments_no_token_logs"]:
            print(f"  No token-level logs in : {agg['experiments_no_token_logs']}")


def print_summary(approach_results: Dict[str, Dict[str, Any]], grand: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("CROSS-APPROACH SUMMARY")
    print("=" * 92)
    header = f"  {'Approach':<14} {'Exps':>5} {'WithData':>10} {'Calls':>10} {'InTokens':>14} {'OutTokens':>14} {'Total':>14}"
    print(header)
    print("  " + "-" * 88)
    for approach, block in approach_results.items():
        agg = block["aggregate"]
        print(
            f"  {approach:<14} "
            f"{agg['experiments_count']:>5} "
            f"{agg['experiments_with_data']:>10} "
            f"{_fmt(agg['llm_calls']):>10} "
            f"{_fmt(agg['total_input_tokens']):>14} "
            f"{_fmt(agg['total_output_tokens']):>14} "
            f"{_fmt(agg['total_tokens']):>14}"
        )
    print("  " + "-" * 88)
    print(
        f"  {'GRAND TOTAL':<14} {'-':>5} {'-':>10} "
        f"{_fmt(grand['llm_calls']):>10} "
        f"{_fmt(grand['total_input_tokens']):>14} "
        f"{_fmt(grand['total_output_tokens']):>14} "
        f"{_fmt(grand['total_tokens']):>14}"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 92)
    print("LLM token-usage scan")
    print("=" * 92)
    print(f"  Scanning {sum(len(v) for v in APPROACH_DIRS.values())} experiment directories "
          f"across {len(APPROACH_DIRS)} approaches.")
    print(f"  (kggen-graphs-gpt-4o intentionally excluded.)")

    approach_results: Dict[str, Dict[str, Any]] = {}
    grand = {
        "llm_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "by_provider": defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        ),
    }

    wall_start = time.time()
    for approach, dirs in APPROACH_DIRS.items():
        print(f"\n  [{approach}] scanning {len(dirs)} dirs ...")
        exp_results: Dict[str, Dict[str, Any]] = {}
        for d in dirs:
            r = scan_experiment(d)
            exp_results[r["experiment_name"]] = r
            mark = (
                f"{r['llm_calls']:>7,} calls" if r["status"] == "ok"
                else f"[{r['status']}]"
            )
            print(f"    {r['experiment_name']:<34} {mark}")
        aggregate = aggregate_experiments(exp_results)
        approach_results[approach] = {
            "experiments": exp_results,
            "aggregate": aggregate,
        }
        # Roll into grand total
        grand["llm_calls"] += aggregate["llm_calls"]
        grand["total_input_tokens"] += aggregate["total_input_tokens"]
        grand["total_output_tokens"] += aggregate["total_output_tokens"]
        grand["total_tokens"] += aggregate["total_tokens"]
        for prov, counts in aggregate["by_provider"].items():
            grand["by_provider"][prov]["calls"] += counts["calls"]
            grand["by_provider"][prov]["input_tokens"] += counts["input_tokens"]
            grand["by_provider"][prov]["output_tokens"] += counts["output_tokens"]
    elapsed = time.time() - wall_start

    grand["by_provider"] = {
        p: {**v, "total_tokens": v["input_tokens"] + v["output_tokens"]}
        for p, v in grand["by_provider"].items()
    }

    print_table(approach_results)
    print_summary(approach_results, grand)

    # --- Save JSON ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = STRUCTURAL_DIR / f"llm_token_usage_{timestamp}.json"
    payload = {
        "metadata": {
            "timestamp": timestamp,
            "scan_wall_time_seconds": round(elapsed, 2),
            "excluded_experiments": ["kggen-graphs-gpt-4o"],
            "token_line_pattern": TOKEN_LINE_RE.pattern,
            "log_file_glob": "*.log (recursive)",
        },
        "approaches": approach_results,
        "grand_total": grand,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 92)
    print(f"Saved: {output_path}")
    print(f"Wall time: {elapsed:.2f}s")
    print("=" * 92)


if __name__ == "__main__":
    main()
