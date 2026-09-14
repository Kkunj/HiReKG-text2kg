"""
Plot MINE-score distribution across 27 experiment cells in 3 panels
(one per dataset), styled after results/plot_scripts/result_plot.py.

Within each panel:
    9 KDE curves     -- 3 approaches x 3 LLMs
    9 mean lines     -- vertical dashed, color-matched to each curve
    Y-axis           -- "frequency (articles)" (KDE x N x bin_width)
    X-axis           -- MINE accuracy %

Encoding:
    color     = approach   (KGGen blue, RAKG red, Ours green)
    linestyle = LLM model  (solid = GPT-4o, dashed = Qwen3-14B, dotted = Qwen3-8B)

Outputs:
    results/mine_score_distribution_kde.png
    results/mine_score_distribution.csv   (per-cell summary stats)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter1d


RESULTS_DIR = Path(
    r"C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\results"
)

APPROACHES = ("kggen", "rakg", "our_approach")
DATASETS = ("mine", "redocred", "scierc")
MODELS = ("gpt4o", "qwen3_14b", "qwen3_8b")  # plot order: gpt4o first

APPROACH_LABEL = {"kggen": "KGgen", "rakg": "RAKG", "our_approach": "Ours"}
DATASET_LABEL = {"mine": "MINE", "redocred": "ReDocRED", "scierc": "SciERC"}
MODEL_LABEL = {"gpt4o": "GPT-4o", "qwen3_14b": "Qwen3-14B", "qwen3_8b": "Qwen3-8B"}

# Color = approach. Hex picked to match the reference script's palette
# (Ours = green, KGgen = blue, RAKG = red) but slightly muted for KDE lines.
APPROACH_COLOR = {
    "our_approach": "#27AE60",   # green
    "kggen":        "#3498DB",   # blue
    "rakg":         "#E74C3C",   # red
}

# Linestyle = model. Solid line reserved for the strongest LLM (GPT-4o) so
# the eye lands on it first.
MODEL_STYLE = {
    "gpt4o":     ("solid",  2.6),
    "qwen3_14b": ("dashed", 2.0),
    "qwen3_8b":  ("dotted", 2.0),
}


# ---------------------------------------------------------------------------
# Classification + loading (same logic as the boxplot script)
# ---------------------------------------------------------------------------

def classify_directory(name: str) -> Optional[Tuple[str, str, str]]:
    n = name.lower()
    tokens = set(n.replace("__", "_").split("_"))

    if "ours" in tokens or "our" in tokens:
        approach = "our_approach"
    elif "kggen" in tokens:
        approach = "kggen"
    elif "rakg" in tokens:
        approach = "rakg"
    else:
        return None

    if "mine" in tokens:
        dataset = "mine"
    elif "redocred" in tokens:
        dataset = "redocred"
    elif "scierc" in tokens:
        dataset = "scierc"
    else:
        return None

    if "qwen3_14b" in n:
        model = "qwen3_14b"
    elif "qwen3_8b" in n:
        model = "qwen3_8b"
    elif "gpt4o" in n or "gpt-4o" in n:
        model = "gpt4o"
    elif "batch_results" in n:
        model = "gpt4o"
    else:
        model = "gpt4o"

    return approach, dataset, model


def read_doc_accuracies(folder: Path) -> List[float]:
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


def collect() -> Dict[Tuple[str, str, str], List[float]]:
    """Return {(dataset, approach, model): [accuracies]}."""
    cells: Dict[Tuple[str, str, str], List[float]] = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name == "comparisons":
            continue
        cls = classify_directory(d.name)
        if cls is None:
            continue
        approach, dataset, model = cls
        accs = read_doc_accuracies(d)
        if not accs:
            continue
        cells.setdefault((dataset, approach, model), []).extend(accs)
    return cells


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def kde_curve(scores: List[float], x: np.ndarray, bin_width: float) -> np.ndarray:
    """KDE scaled to histogram-like 'frequency' y units, then smoothed.

    Matches the reference script's scaling (density * N * bin_width) and
    gaussian_filter1d(sigma=2) post-smoothing.
    """
    kde = stats.gaussian_kde(scores)
    density = kde(x) * len(scores) * bin_width
    return gaussian_filter1d(density, sigma=2)


def plot(cells: Dict[Tuple[str, str, str], List[float]], output_png: Path) -> None:
    bin_width = 5.0
    x_range = np.linspace(0, 100, 300)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        max_y = 0.0
        for approach in APPROACHES:
            color = APPROACH_COLOR[approach]
            for model in MODELS:
                scores = cells.get((ds, approach, model), [])
                if len(scores) < 2:
                    continue
                ls, lw = MODEL_STYLE[model]
                curve = kde_curve(scores, x_range, bin_width)
                max_y = max(max_y, curve.max())
                mean_val = float(np.mean(scores))

                label = (
                    f"{APPROACH_LABEL[approach]} · {MODEL_LABEL[model]}  "
                    f"(μ={mean_val:.1f}%, n={len(scores)})"
                )
                ax.plot(
                    x_range, curve,
                    color=color, linestyle=ls, linewidth=lw,
                    alpha=0.92, label=label,
                )
                # Faint mean line, color-matched, dashed/dotted matching model.
                ax.axvline(
                    mean_val,
                    color=color, linestyle=ls, linewidth=1.0, alpha=0.45,
                )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1)
        ax.set_xlabel("Facts captured / MINE accuracy (%)", fontsize=12, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Frequency (articles)", fontsize=12, fontweight="bold")
        ax.set_title(f"Dataset: {DATASET_LABEL[ds]}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            ncol=1,
        )

    # Single shared encoding legend at the bottom so readers know what
    # color and linestyle mean without parsing 9 per-panel labels.
    color_handles = [
        Line2D([0], [0], color=APPROACH_COLOR[a], linewidth=3.2, label=APPROACH_LABEL[a])
        for a in APPROACHES
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle=MODEL_STYLE[m][0],
               linewidth=2.0, label=MODEL_LABEL[m])
        for m in MODELS
    ]
    fig.legend(
        handles=color_handles + style_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "MINE score distribution  ·  KDE per (approach × LLM), one panel per dataset",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    print(f"Figure saved: {output_png}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary(cells: Dict[Tuple[str, str, str], List[float]], output_csv: Path) -> None:
    rows = []
    for (ds, ap, mo), accs in cells.items():
        rows.append({
            "dataset": ds, "approach": ap, "model": mo,
            "n": len(accs),
            "mean": float(np.mean(accs)),
            "median": float(np.median(accs)),
            "std": float(np.std(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
        })
    df = pd.DataFrame(rows)
    df["dataset"]  = pd.Categorical(df["dataset"],  DATASETS)
    df["approach"] = pd.Categorical(df["approach"], APPROACHES)
    df["model"]    = pd.Categorical(df["model"],    MODELS)
    df = df.sort_values(["dataset", "approach", "model"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Summary CSV saved: {output_csv}")


def main() -> None:
    print(f"Reading from: {RESULTS_DIR}")
    cells = collect()
    print(f"Cells with data: {len(cells)} / 27")
    if len(cells) < 27:
        existing = set(cells.keys())
        for ds in DATASETS:
            for ap in APPROACHES:
                for mo in MODELS:
                    if (ds, ap, mo) not in existing:
                        print(f"  MISSING: dataset={ds} approach={ap} model={mo}")

    out_png = RESULTS_DIR / "mine_score_distribution_kde.png"
    out_csv = RESULTS_DIR / "mine_score_distribution.csv"
    plot(cells, out_png)
    write_summary(cells, out_csv)


if __name__ == "__main__":
    main()
