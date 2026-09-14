"""
Compute per-(approach x dataset) LLM token-usage averages.

Joins:
    - llm_token_usage_<latest>.json   (per-experiment token totals, from
      llm_token_usage.py)
    - uir_<approach>_experiments_<latest>.json  (per-experiment doc counts)

For each of the 9 (approach, dataset) cells -- approaches: kggen / rakg /
our_approach; datasets: mine / redocred / scierc -- it produces:

    experiments_with_tokens      : how many experiments contributed
    experiments_missing_tokens   : names of experiments excluded because
                                   their logs had no token lines
    docs                         : sum of docs_total across contributing
                                   experiments
    llm_calls / input / output / total
                                 : raw sums across contributing experiments
    avg_calls_per_doc            : doc-weighted average
    avg_input_tokens_per_doc     : doc-weighted average
    avg_output_tokens_per_doc    : doc-weighted average
    avg_total_tokens_per_doc     : doc-weighted average

Doc-weighted (sum/sum) is preferred over experiment-mean (mean of per-exp
ratios) because experiments in this set have very uneven doc counts
(100 vs 15), and we care about "per-document cost", not "per-experiment-
config cost".

The result block is APPENDED to the latest llm_token_usage_*.json under
`averages_by_approach_dataset` (a .bak.<timestamp> is created first), and
a 3x3 comparison table is printed.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STRUCTURAL_DIR = Path(__file__).resolve().parent

UIR_GLOBS = {
    "kggen": "uir_kggen_experiments_*.json",
    "rakg": "uir_rakg_experiments_*.json",
    "our_approach": "uir_ours_experiments_*.json",
}
TOKEN_GLOB = "llm_token_usage_*.json"

DATASETS = ("mine", "redocred", "scierc")


def latest(glob_pattern: str) -> Optional[Path]:
    candidates = [
        p for p in STRUCTURAL_DIR.glob(glob_pattern)
        if p.suffix == ".json" and ".bak." not in p.name
    ]
    return sorted(candidates)[-1] if candidates else None


def classify_dataset(name: str) -> str:
    n = name.lower().replace("-", "_")
    for ds in DATASETS:
        if ds in n:
            return ds
    return "other"


def load_uir_doc_counts() -> Dict[str, Dict[str, int]]:
    """
    Return {approach: {experiment_name: docs_total}}.
    `docs_total` is the count of source doc folders, regardless of whether
    extraction produced any triples -- the right denominator for per-doc
    LLM cost averaging.
    """
    out: Dict[str, Dict[str, int]] = {}
    for approach, glob in UIR_GLOBS.items():
        path = latest(glob)
        if path is None:
            print(f"[WARN] no UIR JSON found for {approach} (glob={glob})")
            out[approach] = {}
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out[approach] = {
            name: int(rec.get("docs_total", 0) or 0)
            for name, rec in payload.get("experiments", {}).items()
        }
    return out


def load_token_results() -> Tuple[Path, Dict[str, Any]]:
    path = latest(TOKEN_GLOB)
    if path is None:
        raise FileNotFoundError("No llm_token_usage_*.json found in structural/")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return path, payload


def compute_slices(
    token_payload: Dict[str, Any],
    uir_doc_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns nested dict: {approach: {dataset: aggregate_dict}}.
    """
    slices: Dict[str, Dict[str, Any]] = {}

    for approach, approach_block in token_payload.get("approaches", {}).items():
        approach_docs = uir_doc_counts.get(approach, {})
        # Initialise empty slice cells for every dataset so the JSON output
        # is rectangular (3 x 3) even when a slice has no contributors.
        per_dataset: Dict[str, Dict[str, Any]] = {
            ds: {
                "experiments_with_tokens": 0,
                "experiments_missing_tokens": [],
                "experiments_missing_docs": [],
                "experiment_names": [],
                "docs": 0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "avg_calls_per_doc": None,
                "avg_input_tokens_per_doc": None,
                "avg_output_tokens_per_doc": None,
                "avg_total_tokens_per_doc": None,
            }
            for ds in DATASETS
        }

        for exp_name, exp_rec in approach_block.get("experiments", {}).items():
            ds = classify_dataset(exp_name)
            if ds not in DATASETS:
                continue  # skip 'other' for the 3x3 view
            cell = per_dataset[ds]
            docs = approach_docs.get(exp_name)
            if exp_rec.get("status") != "ok":
                cell["experiments_missing_tokens"].append(exp_name)
                continue
            if not docs:
                cell["experiments_missing_docs"].append(exp_name)
                continue
            cell["experiments_with_tokens"] += 1
            cell["experiment_names"].append(exp_name)
            cell["docs"] += docs
            cell["llm_calls"] += exp_rec.get("llm_calls", 0)
            cell["input_tokens"] += exp_rec.get("total_input_tokens", 0)
            cell["output_tokens"] += exp_rec.get("total_output_tokens", 0)
            cell["total_tokens"] += exp_rec.get("total_tokens", 0)

        # Compute doc-weighted averages.
        for ds, cell in per_dataset.items():
            d = cell["docs"]
            if d > 0:
                cell["avg_calls_per_doc"] = round(cell["llm_calls"] / d, 2)
                cell["avg_input_tokens_per_doc"] = round(cell["input_tokens"] / d, 2)
                cell["avg_output_tokens_per_doc"] = round(cell["output_tokens"] / d, 2)
                cell["avg_total_tokens_per_doc"] = round(cell["total_tokens"] / d, 2)

        slices[approach] = per_dataset
    return slices


def _fmt(n) -> str:
    if n is None:
        return "    N/A"
    if isinstance(n, float):
        return f"{n:>10,.1f}"
    return f"{n:>10,}"


def print_tables(slices: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("PER (approach x dataset) -- doc-weighted averages")
    print("=" * 100)

    for metric_key, metric_label in [
        ("avg_calls_per_doc",        "Avg LLM calls per doc"),
        ("avg_input_tokens_per_doc", "Avg input tokens per doc"),
        ("avg_output_tokens_per_doc","Avg output tokens per doc"),
        ("avg_total_tokens_per_doc", "Avg total tokens per doc"),
    ]:
        print(f"\n  {metric_label}")
        header = f"    {'Approach':<14}" + "".join(f"{ds:>14}" for ds in DATASETS)
        print(header)
        print("    " + "-" * (14 + 14 * len(DATASETS)))
        for approach, per_ds in slices.items():
            row = f"    {approach:<14}"
            for ds in DATASETS:
                v = per_ds[ds].get(metric_key)
                row += _fmt(v) if isinstance(v, (int, float)) else f"{'N/A':>14}"
            print(row)

    print("\n" + "=" * 100)
    print("SLICE COMPOSITION (which experiments fed each cell)")
    print("=" * 100)
    for approach, per_ds in slices.items():
        print(f"\n  [{approach}]")
        for ds in DATASETS:
            cell = per_ds[ds]
            line = (
                f"    {ds:<10} "
                f"exps_with_tokens={cell['experiments_with_tokens']:<3} "
                f"docs={cell['docs']:<5} "
                f"calls={cell['llm_calls']:<8,} "
                f"in={cell['input_tokens']:<12,} "
                f"out={cell['output_tokens']:<12,}"
            )
            print(line)
            if cell["experiments_missing_tokens"]:
                print(f"      (missing tokens: {cell['experiments_missing_tokens']})")
            if cell["experiments_missing_docs"]:
                print(f"      (missing doc count: {cell['experiments_missing_docs']})")


def write_back(token_path: Path, token_payload: Dict[str, Any]) -> None:
    backup = token_path.with_suffix(
        token_path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(token_path, backup)
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Updated {token_path.name}")
    print(f"  Backup : {backup.name}")


def main() -> None:
    print("=" * 100)
    print("Token usage averaging by (approach x dataset)")
    print("=" * 100)

    uir_doc_counts = load_uir_doc_counts()
    for approach, m in uir_doc_counts.items():
        print(f"  {approach}: doc counts loaded for {len(m)} experiments")

    token_path, token_payload = load_token_results()
    print(f"  Token source: {token_path.name}")

    slices = compute_slices(token_payload, uir_doc_counts)

    token_payload["averages_by_approach_dataset"] = {
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "averaging": "doc_weighted (sum(tokens or calls) / sum(docs_total))",
        "denominator_field": "docs_total (from UIR JSONs)",
        "excluded_when": (
            "experiments with status != 'ok' in the token scan, or with no "
            "matching doc-count entry in the UIR JSON"
        ),
        "slices": slices,
    }
    write_back(token_path, token_payload)

    print_tables(slices)
    print()


if __name__ == "__main__":
    main()
