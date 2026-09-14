"""
Plot MINE-score distribution across 27 experiment directories
(3 approaches x 3 datasets x 3 LLMs) as 3 side-by-side panels
-- one panel per dataset.

Within each panel:
    X-axis : approach   (KGGen / RAKG / Ours)
    Hue    : LLM model  (GPT-4o / Qwen3-8B / Qwen3-14B)
    Y-axis : per-document MINE accuracy (%)
    Each cell shows the full distribution as a box-and-whisker, with the
    mean marked as a diamond, and outlying docs as small dots.

The per-doc accuracy is the `accuracy` field in the LAST element of each
results_*.json (matches what `_2_compare_results.py` reads).

Outputs (next to the results folder):
    mine_score_distribution.png      -- the 3-panel figure
    mine_score_distribution.csv      -- per-(approach,dataset,model) summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path(
    r"C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\results"
)

# Canonical ordering for the plot.
APPROACHES = ("kggen", "rakg", "our_approach")
DATASETS = ("mine", "redocred", "scierc")
MODELS = ("gpt4o", "qwen3_8b", "qwen3_14b")

APPROACH_LABEL = {"kggen": "KGGen", "rakg": "RAKG", "our_approach": "Ours"}
DATASET_LABEL = {"mine": "MINE", "redocred": "ReDocRED", "scierc": "SciERC"}
MODEL_LABEL = {"gpt4o": "GPT-4o", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B"}

MODEL_PALETTE = {
    "gpt4o":     "#2E86AB",   # blue
    "qwen3_8b":  "#A23B72",   # magenta
    "qwen3_14b": "#F18F01",   # orange
}


# ---------------------------------------------------------------------------
# Directory -> (approach, dataset, model) classification
# ---------------------------------------------------------------------------

def classify_directory(name: str) -> Optional[Tuple[str, str, str]]:
    """
    Map a result-directory name to (approach, dataset, model).
    Returns None for non-experiment folders (e.g. `comparisons`).
    """
    n = name.lower()
    # token-based check avoids the 'OURS_...' vs 'kggen' ambiguity
    tokens = set(n.replace("__", "_").split("_"))

    # Approach
    if "ours" in tokens or "our" in tokens:
        approach = "our_approach"
    elif "kggen" in tokens:
        approach = "kggen"
    elif "rakg" in tokens:
        approach = "rakg"
    else:
        return None

    # Dataset
    if "mine" in tokens:
        dataset = "mine"
    elif "redocred" in tokens:
        dataset = "redocred"
    elif "scierc" in tokens:
        dataset = "scierc"
    else:
        return None

    # Model (the `MINE_batch_results` runs predate the qwen sweep --
    # they were all GPT-4o.)
    if "qwen3_14b" in n:
        model = "qwen3_14b"
    elif "qwen3_8b" in n:
        model = "qwen3_8b"
    elif "gpt4o" in n or "gpt-4o" in n:
        model = "gpt4o"
    elif "batch_results" in n:
        model = "gpt4o"
    else:
        model = "gpt4o"  # safe default; will be flagged if it's wrong

    return approach, dataset, model


# ---------------------------------------------------------------------------
# Read per-doc accuracies from a result folder
# ---------------------------------------------------------------------------

def read_doc_accuracies(folder: Path) -> List[float]:
    """
    For each results_*.json in `folder`, return the per-doc accuracy
    (parsed from the trailing {"accuracy": "<XX.XX%>"} record).
    """
    accs: List[float] = []
    for jp in sorted(folder.glob("results_*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list) or not data:
            continue
        last = data[-1]
        if not isinstance(last, dict) or "accuracy" not in last:
            continue
        try:
            accs.append(float(str(last["accuracy"]).replace("%", "")))
        except ValueError:
            continue
    return accs


# ---------------------------------------------------------------------------
# Build long-form DataFrame
# ---------------------------------------------------------------------------

def build_dataframe() -> Tuple[pd.DataFrame, List[str], List[str]]:
    records: List[Dict] = []
    unclassified: List[str] = []
    empty: List[str] = []

    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name == "comparisons":
            continue
        cls = classify_directory(d.name)
        if cls is None:
            unclassified.append(d.name)
            continue
        approach, dataset, model = cls
        accs = read_doc_accuracies(d)
        if not accs:
            empty.append(d.name)
            continue
        for a in accs:
            records.append({
                "approach": approach,
                "dataset": dataset,
                "model": model,
                "accuracy": a,
                "source_dir": d.name,
            })

    df = pd.DataFrame(records)
    return df, unclassified, empty


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, output_png: Path) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        if sub.empty:
            ax.set_title(f"{DATASET_LABEL[ds]} (no data)", fontsize=14)
            continue

        sns.boxplot(
            data=sub,
            x="approach",
            y="accuracy",
            hue="model",
            order=APPROACHES,
            hue_order=MODELS,
            palette=MODEL_PALETTE,
            ax=ax,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 6,
            },
            fliersize=2.5,
            linewidth=1.0,
            width=0.7,
        )

        ax.set_xticklabels([APPROACH_LABEL[a] for a in APPROACHES], fontsize=11)
        ax.set_title(f"Dataset: {DATASET_LABEL[ds]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("MINE accuracy (%)" if ax is axes[0] else "", fontsize=12)
        ax.set_ylim(-2, 105)
        ax.grid(True, axis="y", alpha=0.35)

        # Per-panel legend with friendly labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            [MODEL_LABEL[l] for l in labels],
            title="LLM",
            loc="lower right",
            frameon=True,
            fontsize=10,
            title_fontsize=10,
        )

    fig.suptitle(
        "MINE score distribution  ·  3 approaches × 3 LLMs per dataset",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {output_png}")


def write_summary_csv(df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    """Cell-level mean/median/std/min/max/n, sorted by (dataset, approach, model)."""
    summary = (
        df.groupby(["dataset", "approach", "model"])["accuracy"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    # Force canonical order
    summary["dataset"] = pd.Categorical(summary["dataset"], DATASETS)
    summary["approach"] = pd.Categorical(summary["approach"], APPROACHES)
    summary["model"] = pd.Categorical(summary["model"], MODELS)
    summary = summary.sort_values(["dataset", "approach", "model"]).reset_index(drop=True)
    summary.to_csv(output_csv, index=False)
    print(f"Summary CSV saved: {output_csv}")
    return summary


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading from: {RESULTS_DIR}")
    df, unclassified, empty = build_dataframe()

    if unclassified:
        print(f"\n[unclassified folders skipped] {unclassified}")
    if empty:
        print(f"[no results_*.json found in] {empty}")

    # Sanity: 27 cells, each with N>=1
    cell_counts = (
        df.groupby(["dataset", "approach", "model"]).size().rename("n_docs").reset_index()
    )
    print(f"\nCells filled: {len(cell_counts)} / 27")
    if len(cell_counts) < 27:
        # Which cells are missing?
        existing = set(map(tuple, cell_counts[["dataset", "approach", "model"]].values))
        for ds in DATASETS:
            for ap in APPROACHES:
                for mo in MODELS:
                    if (ds, ap, mo) not in existing:
                        print(f"  MISSING: dataset={ds} approach={ap} model={mo}")

    out_png = RESULTS_DIR / "mine_score_distribution.png"
    out_csv = RESULTS_DIR / "mine_score_distribution.csv"
    plot(df, out_png)
    summary = write_summary_csv(df, out_csv)

    # Pretty-print the means table for the terminal.
    print("\n" + "=" * 78)
    print("Per-cell means (% accuracy)")
    print("=" * 78)
    pivot = (
        summary.pivot_table(
            index=["dataset", "approach"],
            columns="model",
            values="mean",
            aggfunc="first",
        )
        .reindex(MODELS, axis=1)
    )
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
