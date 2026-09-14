import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import os
import glob
from pathlib import Path

# Directories for each approach (resolved relative to the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MINE_RESULTS = REPO_ROOT / "evaluation_pipeline" / "mine_score" / "results"

ours_dir = MINE_RESULTS / "OURS_MINE_batch_results"
kggen_dir = (REPO_ROOT / "baselines" / "kggen" / "experiments"
             / "kggen_mine_gpt4o_graphs" / "mine_results")
rakg_dir = MINE_RESULTS / "RAKG_MINE_batch_results"

def load_scores_from_json_dir(directory, pattern='results_*.json', limit=None):
    """Load accuracy scores from results JSON files in a directory."""
    json_files = sorted(glob.glob(os.path.join(directory, pattern)))
    if limit:
        json_files = json_files[:limit]
    scores = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                last_item = data[-1]
                if isinstance(last_item, dict) and 'accuracy' in last_item:
                    accuracy_value = float(last_item['accuracy'].rstrip('%'))
                    scores.append(accuracy_value)
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")
    return scores

# Load scores for each approach
ours_scores = load_scores_from_json_dir(ours_dir)
kggen_scores = load_scores_from_json_dir(kggen_dir, pattern='*_results.json', limit=100)
rakg_scores = load_scores_from_json_dir(rakg_dir)

print(f"Our Approach: {len(ours_scores)} files, mean={np.mean(ours_scores):.2f}%")
print(f"KGgen: {len(kggen_scores)} files, mean={np.mean(kggen_scores):.2f}%")
print(f"RAKG: {len(rakg_scores)} files, mean={np.mean(rakg_scores):.2f}%")

ours_mean = np.mean(ours_scores)
kggen_mean = np.mean(kggen_scores)
rakg_mean = np.mean(rakg_scores)

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

bins = np.arange(0, 105, 5)
x_range = np.linspace(0, 100, 300)

approaches = [
    ('Our Approach', ours_scores, ours_mean, 'green', 'lightgreen', 'darkgreen'),
    ('KGgen',        kggen_scores, kggen_mean, '#3498DB', '#AED6F1', '#2471A3'),
    ('RAKG',         rakg_scores,  rakg_mean,  '#E74C3C', '#F5B7B1', '#C0392B'),
]

max_y = 0
for name, scores, mean_val, color, hist_color, edge_color in approaches:
    # Histogram
    counts, _, _ = ax.hist(scores, bins=bins, alpha=0.4, color=hist_color,
                           edgecolor=edge_color, linewidth=0.5)
    max_y = max(max_y, counts.max())

    # KDE curve
    kde = stats.gaussian_kde(scores)
    density = kde(x_range)
    density_scaled = density * len(scores) * (bins[1] - bins[0])
    density_smoothed = gaussian_filter1d(density_scaled, sigma=2)
    ax.plot(x_range, density_smoothed, color=color, linewidth=2.5, label=name)
    max_y = max(max_y, density_smoothed.max())

    # Mean vertical line
    ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.7,
               label=f'{name} Mean ({mean_val:.1f}%)')

# Styling
ax.set_xlabel('Facts captured (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (Articles)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, max_y * 1.15)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

# Title
title = f'Comparison of Knowledge Graph Extraction Approaches\n'
title += f'MINE Score distribution across {len(ours_scores)} articles'
plt.title(title, fontsize=12, pad=20, loc='left')

plt.tight_layout()

plt.savefig(Path(__file__).resolve().parent / 'kg_extraction_comparison.png', dpi=300, bbox_inches='tight')
print(f"Plot saved!")

plt.show()
