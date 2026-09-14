"""
BERTScore-based evaluation metrics for knowledge graph triplet matching.

Provides three matching strategies to compare predicted vs gold triplets:
  - One-to-one:   Hungarian algorithm for optimal bipartite matching.
  - Many-to-many: Each triplet matched to its best counterpart (no exclusivity).
  - Threshold:    Like many-to-many, but only pairs above a similarity threshold count.

Each strategy returns (precision, recall, f1).
"""

import numpy as np
from bert_score import score as score_bert
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_to_edge_strings(graph: List[List[str]]) -> List[str]:
    """Convert a single graph's triplets to semicolon-joined lowercase strings."""
    return [";".join(str(x) for x in triple).lower().strip() for triple in graph]


def _compute_f1(precision: float, recall: float) -> float:
    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


# ---------------------------------------------------------------------------
# Score matrix computation
# ---------------------------------------------------------------------------

def compute_bert_score_matrix(
    pred_edges: List[str],
    gold_edges: List[str],
    model_type: str = "bert-base-uncased",
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute BERTScore F1 matrix between predicted and gold edge strings.

    Returns:
        np.ndarray of shape (len(gold_edges), len(pred_edges)).
    """
    references = []
    candidates = []
    ref_cand_index = {}

    for j, pred_edge in enumerate(pred_edges):
        for i, gold_edge in enumerate(gold_edges):
            references.append(gold_edge)
            candidates.append(pred_edge)
            ref_cand_index[(i, j)] = len(references) - 1

    _, _, bs_f1 = score_bert(
        cands=candidates,
        refs=references,
        model_type=model_type,
        lang="en",
        idf=False,
        device=device,
    )

    score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
    for i in range(len(gold_edges)):
        for j in range(len(pred_edges)):
            score_matrix[i, j] = bs_f1[ref_cand_index[(i, j)]].item()

    return score_matrix


# ---------------------------------------------------------------------------
# Internal scoring from a precomputed matrix
# ---------------------------------------------------------------------------

def _one_to_one_from_matrix(
    score_matrix: np.ndarray, n_pred: int, n_gold: int,
) -> Tuple[float, float, float]:
    """One-to-one matching via Hungarian algorithm."""
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
    matched_sum = score_matrix[row_ind, col_ind].sum()

    precision = matched_sum / n_pred
    recall = matched_sum / n_gold
    return precision, recall, _compute_f1(precision, recall)


def _many_to_many_from_matrix(
    score_matrix: np.ndarray, n_pred: int, n_gold: int,
) -> Tuple[float, float, float]:
    """Many-to-many matching: best score per triplet, no exclusivity."""
    precision = np.max(score_matrix, axis=0).sum() / n_pred
    recall = np.max(score_matrix, axis=1).sum() / n_gold
    return precision, recall, _compute_f1(precision, recall)


def _threshold_from_matrix(
    score_matrix: np.ndarray, n_pred: int, n_gold: int, threshold: float,
) -> Tuple[float, float, float]:
    """Threshold matching: best score per triplet, only if >= threshold."""
    # Precision: for each predicted edge, best gold score above threshold
    precision_scores = []
    for j in range(n_pred):
        col = score_matrix[:, j]
        above = col[col >= threshold]
        precision_scores.append(above.max() if above.size > 0 else 0.0)

    # Recall: for each gold edge, best predicted score above threshold
    recall_scores = []
    for i in range(n_gold):
        row = score_matrix[i, :]
        above = row[row >= threshold]
        recall_scores.append(above.max() if above.size > 0 else 0.0)

    precision = sum(precision_scores) / n_pred
    recall = sum(recall_scores) / n_gold
    return precision, recall, _compute_f1(precision, recall)


# ---------------------------------------------------------------------------
# Public API — individual metrics
# ---------------------------------------------------------------------------

def bert_score_one_to_one(
    pred_graph: List[List[str]],
    gold_graph: List[List[str]],
    model_type: str = "bert-base-uncased",
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    BERTScore with one-to-one matching (Hungarian algorithm).

    Each predicted triplet is matched to at most one gold triplet and vice versa.

    Returns:
        (precision, recall, f1)
    """
    if not pred_graph or not gold_graph:
        return 0.0, 0.0, 0.0

    pred_edges = _graph_to_edge_strings(pred_graph)
    gold_edges = _graph_to_edge_strings(gold_graph)
    score_matrix = compute_bert_score_matrix(pred_edges, gold_edges, model_type, device)
    return _one_to_one_from_matrix(score_matrix, len(pred_edges), len(gold_edges))


def bert_score_many_to_many(
    pred_graph: List[List[str]],
    gold_graph: List[List[str]],
    model_type: str = "bert-base-uncased",
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    BERTScore with many-to-many matching.

    Each triplet is matched to its single best counterpart; the same gold
    triplet may be matched by multiple predictions and vice versa.

    Returns:
        (precision, recall, f1)
    """
    if not pred_graph or not gold_graph:
        return 0.0, 0.0, 0.0

    pred_edges = _graph_to_edge_strings(pred_graph)
    gold_edges = _graph_to_edge_strings(gold_graph)
    score_matrix = compute_bert_score_matrix(pred_edges, gold_edges, model_type, device)
    return _many_to_many_from_matrix(score_matrix, len(pred_edges), len(gold_edges))


def bert_score_threshold(
    pred_graph: List[List[str]],
    gold_graph: List[List[str]],
    threshold: float = 0.5,
    model_type: str = "bert-base-uncased",
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    BERTScore with threshold-based matching.

    Like many-to-many, but only pairs with similarity >= threshold are considered.

    Returns:
        (precision, recall, f1)
    """
    if not pred_graph or not gold_graph:
        return 0.0, 0.0, 0.0

    pred_edges = _graph_to_edge_strings(pred_graph)
    gold_edges = _graph_to_edge_strings(gold_graph)
    score_matrix = compute_bert_score_matrix(pred_edges, gold_edges, model_type, device)
    return _threshold_from_matrix(score_matrix, len(pred_edges), len(gold_edges), threshold)


# ---------------------------------------------------------------------------
# Public API — all metrics at once (computes score matrix only once)
# ---------------------------------------------------------------------------

def calculate_all_bert_scores(
    pred_graph: List[List[str]],
    gold_graph: List[List[str]],
    threshold: float = 0.5,
    model_type: str = "bert-base-uncased",
    device: str = "cpu",
) -> dict:
    """
    Compute all three BERTScore matching strategies in a single pass.

    Returns:
        {
            'one_to_one':   {'precision': float, 'recall': float, 'f1': float},
            'many_to_many': {'precision': float, 'recall': float, 'f1': float},
            'threshold':    {'precision': float, 'recall': float, 'f1': float},
        }
    """
    empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not pred_graph or not gold_graph:
        return {k: empty.copy() for k in ("one_to_one", "many_to_many", "threshold")}

    pred_edges = _graph_to_edge_strings(pred_graph)
    gold_edges = _graph_to_edge_strings(gold_graph)
    n_pred, n_gold = len(pred_edges), len(gold_edges)

    score_matrix = compute_bert_score_matrix(pred_edges, gold_edges, model_type, device)

    def _to_dict(p, r, f1):
        return {"precision": p, "recall": r, "f1": f1}

    return {
        "one_to_one":   _to_dict(*_one_to_one_from_matrix(score_matrix, n_pred, n_gold)),
        "many_to_many": _to_dict(*_many_to_many_from_matrix(score_matrix, n_pred, n_gold)),
        "threshold":    _to_dict(*_threshold_from_matrix(score_matrix, n_pred, n_gold, threshold)),
    }


# # ---------------------------------------------------------------------------
# # Example usage
# # ---------------------------------------------------------------------------
#
# predicted = [
#     ["mathematical formalism", "controls", "the saturation"],
#     ["entity A", "relates to", "entity B"],
# ]
#
# gold = [
#     ["formalism", "used for", "grammar formalisms"],
#     ["entity A", "relates to", "entity B"],
# ]
#
# # Individual metrics
# p, r, f1 = bert_score_one_to_one(predicted, gold)
# p, r, f1 = bert_score_many_to_many(predicted, gold)
# p, r, f1 = bert_score_threshold(predicted, gold, threshold=0.5)
#
# # All at once (single matrix computation)
# results = calculate_all_bert_scores(predicted, gold)
# print(results["one_to_one"])    # {'precision': ..., 'recall': ..., 'f1': ...}
# print(results["many_to_many"])  # {'precision': ..., 'recall': ..., 'f1': ...}
# print(results["threshold"])     # {'precision': ..., 'recall': ..., 'f1': ...}
