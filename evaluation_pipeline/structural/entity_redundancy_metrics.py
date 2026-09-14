"""
Entity redundancy metrics for knowledge graph evaluation.

Measures how redundant (duplicated) the entities in a generated knowledge graph
are, using a combination of fuzzy string matching and semantic similarity.

Pipeline:
  1. Compute pairwise fuzzy scores  (token sort ratio + Levenshtein).
  2. Compute pairwise semantic scores (cosine similarity of sentence embeddings).
  3. Combine via adaptive weighted harmonic mean (smooth sigmoid blending).
  4. Derive an overall redundancy score and count of redundant pairs.
"""

import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pairwise scoring helpers
# ---------------------------------------------------------------------------

def calculate_fuzzy_score(entity1: str, entity2: str) -> float:
    """
    Geometric mean of token-sort ratio and Levenshtein ratio.

    Both metrics must be high for the combined score to be high,
    penalising cases where only one metric is elevated.
    """
    token_sort = fuzz.token_sort_ratio(entity1, entity2) / 100.0
    levenshtein = fuzz.ratio(entity1, entity2) / 100.0
    return np.sqrt(token_sort * levenshtein)


def calculate_semantic_score(embeddings: np.ndarray, idx1: int, idx2: int) -> float:
    """Cosine similarity between two entity embeddings, clamped to [0, 1]."""
    emb1 = embeddings[idx1].reshape(1, -1)
    emb2 = embeddings[idx2].reshape(1, -1)
    return float(np.clip(cosine_similarity(emb1, emb2)[0][0], 0.0, 1.0))


def combine_scores(fuzzy_score: float, semantic_score: float) -> float:
    """
    Adaptive weighted harmonic mean of fuzzy and semantic scores.

    The fuzzy weight increases smoothly around fuzzy_score ≈ 0.8 via a
    sigmoid, avoiding a hard discontinuity.  When the fuzzy score is high
    (likely a linguistic duplicate) fuzzy dominates (up to 0.7); when it is
    low, semantic dominates (fuzzy weight drops to 0.4).
    """
    epsilon = 1e-10

    # Smooth transition centred at 0.8 (steepness k=20 gives a ~0.1-wide ramp)
    blend = 1.0 / (1.0 + np.exp(-20 * (fuzzy_score - 0.8)))
    w_fuzzy = 0.4 + 0.3 * blend      # ranges from 0.4 to 0.7
    w_semantic = 1.0 - w_fuzzy        # ranges from 0.6 to 0.3

    combined = (w_fuzzy + w_semantic) / (
        (w_fuzzy / (fuzzy_score + epsilon)) + (w_semantic / (semantic_score + epsilon))
    )
    return combined


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def calculate_entity_redundancy(
    entities: List[str],
    descriptions: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.75,
    visualize: bool = False,
    verbose: bool = True,
    model: Optional[SentenceTransformer] = None,
) -> Tuple[float, dict, int]:
    """
    Calculate overall redundancy score for a list of entities with descriptions.

    Parameters
    ----------
    entities : List[str]
        Entity name strings.
    descriptions : List[str]
        Context / description for each entity (same length as *entities*).
    model_name : str
        Sentence-transformer model used for embeddings.
    threshold : float
        Minimum combined score to consider a pair redundant.
    visualize : bool
        Whether to show heatmap visualisation.
    verbose : bool
        Whether to print progress and results.
    model : SentenceTransformer, optional
        Pre-loaded sentence-transformer model. If provided, *model_name* is
        ignored and the model is not reloaded.

    Returns
    -------
    overall_redundancy : float
        Mean pairwise redundancy across all unique pairs [0, 1].
    details : dict
        Intermediate matrices and statistics.
    redundant_pair_count : int
        Number of unique pairs with combined score >= *threshold*.
    """
    n = len(entities)
    if len(descriptions) != n:
        raise ValueError("Number of descriptions must match number of entities")

    # --- Edge case: 0 or 1 entities have no pairs to compare ---
    if n <= 1:
        return 0.0, {}, 0

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"Calculating redundancy for {n} entities...")

    # --- Step 1: contextualised embeddings ---
    contextualized_texts = [
        f"{entity}: {desc}" for entity, desc in zip(entities, descriptions)
    ]
    if model is None:
        model = SentenceTransformer(model_name)
    embeddings = model.encode(contextualized_texts, show_progress_bar=False)

    # --- Step 2: pairwise similarity matrices ---
    fuzzy_matrix = np.eye(n)
    semantic_matrix = np.eye(n)
    combined_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            f_score = calculate_fuzzy_score(entities[i], entities[j])
            s_score = calculate_semantic_score(embeddings, i, j)
            c_score = combine_scores(f_score, s_score)

            fuzzy_matrix[i, j] = fuzzy_matrix[j, i] = f_score
            semantic_matrix[i, j] = semantic_matrix[j, i] = s_score
            combined_matrix[i, j] = combined_matrix[j, i] = c_score

    # --- Step 3: aggregate ---
    upper_tri = combined_matrix[np.triu_indices(n, k=1)]
    overall_redundancy = float(np.mean(upper_tri))
    redundant_pair_count = int(np.sum(upper_tri >= threshold))

    # Statistics
    max_redundancy = float(np.max(upper_tri))
    min_redundancy = float(np.min(upper_tri))
    std_redundancy = float(np.std(upper_tri))

    # Top redundant pairs
    pair_scores = sorted(
        [
            (i, j, combined_matrix[i, j])
            for i in range(n)
            for j in range(i + 1, n)
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    if verbose:
        print(f"{'=' * 60}")
        print(f"Overall Redundancy Score: {overall_redundancy:.4f}")
        print(f"Max Pairwise Redundancy:  {max_redundancy:.4f}")
        print(f"Min Pairwise Redundancy:  {min_redundancy:.4f}")
        print(f"Std Dev:                  {std_redundancy:.4f}")
        print(f"Redundant pairs (>= {threshold}): {redundant_pair_count}")
        print(f"{'=' * 60}")
        print("\nTop 5 Most Redundant Pairs:")
        for rank, (i, j, score) in enumerate(pair_scores[:5], 1):
            print(f"  {rank}. '{entities[i]}' <-> '{entities[j]}': {score:.4f}")

    if visualize:
        visualize_redundancy_analysis(
            entities, fuzzy_matrix, semantic_matrix, combined_matrix, overall_redundancy
        )

    details = {
        "fuzzy_matrix": fuzzy_matrix,
        "semantic_matrix": semantic_matrix,
        "combined_matrix": combined_matrix,
        "overall_redundancy": overall_redundancy,
        "max_redundancy": max_redundancy,
        "min_redundancy": min_redundancy,
        "std_redundancy": std_redundancy,
        "top_redundant_pairs": pair_scores[:10],
        "embeddings": embeddings,
    }

    return overall_redundancy, details, redundant_pair_count


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_redundancy_analysis(
    entities: List[str],
    fuzzy_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
    combined_matrix: np.ndarray,
    overall_score: float,
) -> None:
    """Render four-panel heatmap of the redundancy analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    labels = [e[:20] + "..." if len(e) > 20 else e for e in entities]

    heatmap_configs = [
        (axes[0, 0], fuzzy_matrix, "YlOrRd", "Fuzzy Score",
         "Fuzzy Similarity Matrix\n(Token Sort + Levenshtein)"),
        (axes[0, 1], semantic_matrix, "YlGnBu", "Semantic Score",
         "Semantic Similarity Matrix\n(Contextualised Embeddings)"),
        (axes[1, 0], combined_matrix, "RdPu", "Combined Score",
         "Combined Redundancy Matrix (FINAL)\n(Symmetric)"),
    ]

    for ax, matrix, cmap, cbar_label, title in heatmap_configs:
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap=cmap,
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar_kws={"label": cbar_label}, vmin=0, vmax=1,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Entities")
        ax.set_ylabel("Entities")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Distribution histogram
    ax4 = axes[1, 1]
    n = len(entities)
    redundancy_scores = combined_matrix[np.triu_indices(n, k=1)]
    ax4.hist(redundancy_scores, bins=20, color="purple", alpha=0.7, edgecolor="black")
    ax4.axvline(
        overall_score, color="red", linestyle="--", linewidth=2,
        label=f"Overall Score: {overall_score:.4f}",
    )
    ax4.set_xlabel("Redundancy Score", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_title("Distribution of Pairwise Redundancy Scores", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# # ---------------------------------------------------------------------------
# # Example usage
# # ---------------------------------------------------------------------------
#
# nodes = [
#     {
#         "name": "Rocket",
#         "type": "Object",
#         "description": "A rocket is a vehicle designed to propel itself...",
#     },
#     {
#         "name": "Physics",
#         "type": "Scientific Discipline",
#         "description": "Physics is the branch of science concerned with...",
#     },
# ]
#
# entities = [n["name"] for n in nodes]
# descriptions = [n["description"] for n in nodes]
#
# score, details, redundant_pairs = calculate_entity_redundancy(
#     entities=entities,
#     descriptions=descriptions,
#     threshold=0.75,
#     visualize=False,
# )
#
# print(f"Redundancy Score: {score:.4f}")
# print(f"Redundant pairs:  {redundant_pairs}")
# print(f"Ratio:            {redundant_pairs / len(entities):.4f}")
