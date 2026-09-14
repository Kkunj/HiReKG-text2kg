"""
Plot MINE-score distribution -- ONE chart per PNG, ONE PNG per dataset.

Style matches the screenshot the user approved:
    * 9 KDE curves per chart (3 approaches x 3 LLMs)
    * color    = approach   (KGgen blue, RAKG red, Ours green)
    * linestyle = LLM model (solid = GPT-4o, dashed = Qwen3-14B, dotted = Qwen3-8B)
    * faint dashed vertical line at each curve's mean, color-matched
    * legend entry per curve includes mean, std, and n

Outputs (in mine_score/results/):
    mine_distribution_MINE.png
    mine_distribution_ReDocRED.png
    mine_distribution_SciERC.png
    mine_score_distribution.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d


RESULTS_DIR = Path(
    r"C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\results"
)

APPROACHES = ("kggen", "rakg", "our_approach")
DATASETS = ("mine", "redocred", "scierc")
MODELS = ("gpt4o", "qwen3_14b", "qwen3_8b")

APPROACH_LABEL = {"kggen": "KGgen", "rakg": "RAKG", "our_approach": "Ours"}
DATASET_LABEL = {"mine": "MINE", "redocred": "ReDocRED", "scierc": "SciERC"}
MODEL_LABEL = {"gpt4o": "GPT-4o", "qwen3_14b": "Qwen3-14B", "qwen3_8b": "Qwen3-8B"}

APPROACH_COLOR = {
    "kggen":        "#3498DB",   # blue
    "rakg":         "#E74C3C",   # red
    "our_approach": "#27AE60",   # green
}

# (linestyle, linewidth). Solid + thick reserved for GPT-4o so it draws
# the eye first.
MODEL_STYLE = {
    "gpt4o":     ("solid",  3.0),
    "qwen3_14b": ("dashed", 2.4),
    "qwen3_8b":  ("dotted", 2.4),
}


# ---------------------------------------------------------------------------
# Loaders + classification
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
# Plot one dataset -> one figure
# ---------------------------------------------------------------------------

BIN_WIDTH = 5.0
X_RANGE = np.linspace(0, 100, 300)


def plot_one_dataset(
    cells: Dict[Tuple[str, str, str], List[float]],
    dataset: str,
    output_png: Path,
) -> None:
    """Single chart, 9 KDE curves, color = approach, linestyle = model."""
    fig, ax = plt.subplots(figsize=(15, 9))

    max_y = 0.0
    # Plot order: by approach then by model -- keeps the legend logically
    # grouped by approach so the eye reads "KGgen block, RAKG block, ...".
    for approach in APPROACHES:
        color = APPROACH_COLOR[approach]
        for model in MODELS:
            scores = cells.get((dataset, approach, model), [])
            if len(scores) < 2:
                continue
            ls, lw = MODEL_STYLE[model]
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))

            # KDE scaled to article-frequency units, then smoothed.
            kde = stats.gaussian_kde(scores)
            density = kde(X_RANGE) * len(scores) * BIN_WIDTH
            density_smoothed = gaussian_filter1d(density, sigma=2)
            max_y = max(max_y, density_smoothed.max())

            ax.plot(
                X_RANGE, density_smoothed,
                color=color, linestyle=ls, linewidth=lw,
                alpha=0.95,
                label=(
                    f"{APPROACH_LABEL[approach]} · {MODEL_LABEL[model]}  "
                    f"(μ={mean_val:.1f}%  σ={std_val:.1f}%,  n={len(scores)})"
                ),
            )
            # Mean line, color & style matched to the curve
            ax.axvline(
                mean_val,
                color=color, linestyle=ls, linewidth=2.4, alpha=0.85,
            )

    # Styling for readability
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1.0)
    ax.set_xlabel("Facts captured  /  MINE accuracy (%)",
                  fontsize=15, fontweight="bold")
    ax.set_ylabel("Frequency (Articles)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.30, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title(f"Dataset: {DATASET_LABEL[dataset]}",
                 fontsize=18, fontweight="bold", pad=12)

    ax.legend(
        loc="upper left",
        fontsize=14,
        framealpha=0.92,
        title="Approach · LLM   (μ = mean,  σ = std dev)",
        title_fontsize=14,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_png.name}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary(cells, output_csv: Path) -> None:
    rows = []
    for (ds, ap, mo), accs in cells.items():
        rows.append({
            "dataset": ds, "approach": ap, "model": mo,
            "n": len(accs),
            "mean":   float(np.mean(accs)),
            "median": float(np.median(accs)),
            "std":    float(np.std(accs)),
            "min":    float(np.min(accs)),
            "max":    float(np.max(accs)),
        })
    df = pd.DataFrame(rows)
    df["dataset"]  = pd.Categorical(df["dataset"],  DATASETS)
    df["approach"] = pd.Categorical(df["approach"], APPROACHES)
    df["model"]    = pd.Categorical(df["model"],    MODELS)
    df = df.sort_values(["dataset", "approach", "model"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}")


def main() -> None:
    print(f"Reading from: {RESULTS_DIR}")
    cells = collect()
    print(f"Cells with data: {len(cells)} / 27")
    missing = [
        (ds, ap, mo)
        for ds in DATASETS for ap in APPROACHES for mo in MODELS
        if (ds, ap, mo) not in cells
    ]
    for ds, ap, mo in missing:
        print(f"  MISSING: dataset={ds} approach={ap} model={mo}")

    print("\nWriting outputs:")
    for ds in DATASETS:
        out_png = RESULTS_DIR / f"mine_distribution_{DATASET_LABEL[ds]}.png"
        plot_one_dataset(cells, ds, out_png)

    write_summary(cells, RESULTS_DIR / "mine_score_distribution.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
