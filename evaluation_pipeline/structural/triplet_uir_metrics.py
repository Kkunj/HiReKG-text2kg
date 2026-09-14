"""
Triplet-level UIR (Unique Information Ratio) for knowledge graph evaluation.

Metric definition:

        UIR = Number of Clusters / Number of Triples

Triples that express the same fact in different surface forms
(e.g. "Eiffel Tower is located in Paris",
      "Eiffel Tower stands in the city of Paris",
      "Paris is the location of Eiffel Tower")
are clustered together when their pairwise BERTScore F1 exceeds a
threshold (default 0.8). Each cluster contributes a single unit of
"unique information"; UIR is the ratio of unique information to the
total triple count.

Interpretation
--------------
    UIR = 1.0  -> every triple is semantically unique (no redundancy)
    UIR < 1.0  -> some triples are semantic duplicates
    UIR -> 0   -> all triples collapse to one fact (maximum redundancy)

NOTE on wording: the original metric write-up reads "high UIR -> high
redundancy",
which contradicts the formula. Using the formula clusters/triples, HIGH UIR
means LOW redundancy. This module follows the formula (which matches the
worked example: 4 triples, 2 clusters -> UIR = 0.5 indicates
*more* redundancy than the ideal 1.0). This convention also matches the
existing run_uir_evaluation.py treatment of UIR as the "good" direction.

Pipeline
--------
    1. Verbalize each triple as "subject relation object".
    2. Compute pairwise similarity between verbalized triples.
       - Default: BERTScore F1 (Zhang et al. 2019), as specified.
       - Optional: SentenceTransformer cosine - faster for large graphs.
    3. Build a similarity graph; place an edge between (i, j) iff
       sim(i, j) >= threshold.
    4. Find connected components (union-find). Each component = a cluster.
    5. UIR = num_clusters / num_triples.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# A triple may be supplied as a 3-tuple/list (s, r, o) or as a dict with
# (subject|head, relation|predicate, object|tail) keys.
Triple = Union[Tuple[str, str, str], List[str], Dict[str, str]]


# ---------------------------------------------------------------------------
# Step 1: verbalisation
# ---------------------------------------------------------------------------

def verbalize_triple(triple: Triple) -> str:
    """
    Convert a triple into a single sentence "subject relation object".

    Accepts:
        - 3-tuple / 3-list: (subject, relation, object)
        - dict with keys {'subject'|'head', 'relation'|'predicate', 'object'|'tail'}
    """
    if isinstance(triple, dict):
        s = triple.get("subject", triple.get("head", ""))
        r = triple.get("relation", triple.get("predicate", ""))
        o = triple.get("object", triple.get("tail", ""))
    elif isinstance(triple, (list, tuple)) and len(triple) == 3:
        s, r, o = triple
    else:
        raise ValueError(f"Unsupported triple format: {triple!r}")

    parts = [str(s).strip(), str(r).strip(), str(o).strip()]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Step 2a: pairwise BERTScore (preferred, as specified)
# ---------------------------------------------------------------------------

def compute_pairwise_bertscore(
    verbalized: Sequence[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    batch_size: int = 64,
    device: Optional[str] = None,
    rescale_with_baseline: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute the pairwise BERTScore F1 matrix for a list of verbalized triples.

    BERTScore (Zhang et al. 2019) aligns tokens between two sentences using
    contextual embeddings and reports precision/recall/F1. We use F1 as the
    similarity score, which is the standard choice in the literature.

    Parameters
    ----------
    verbalized : list of str
        Verbalized triples (one sentence per triple).
    lang : str
        Language code passed to bert_score. Selects a default model_type
        (roberta-large for English).
    model_type : str, optional
        Explicit HF model identifier. Overrides `lang`. Useful for picking
        a lighter encoder (e.g. "distilbert-base-uncased") to speed up
        very large graphs at some cost to fidelity.
    batch_size : int
        Batch size for the underlying encoder.
    device : str, optional
        "cuda", "cpu", or None (auto-detect).
    rescale_with_baseline : bool
        Apply BERTScore's baseline rescaling. Off by default because the
        specified "> 0.8" threshold is stated in the un-rescaled scale.
    verbose : bool
        Forwarded to bert_score for progress reporting.

    Returns
    -------
    matrix : np.ndarray of shape (n, n)
        Symmetric matrix; diagonal is 1.0.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise ImportError(
            "bert_score is required for method='bertscore'. "
            "Install with `pip install bert-score`."
        ) from exc

    n = len(verbalized)
    if n <= 1:
        return np.eye(max(n, 1))

    # Enumerate the upper-triangular pairs and batch them through BERTScore.
    cands: List[str] = []
    refs: List[str] = []
    idx_pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cands.append(verbalized[i])
            refs.append(verbalized[j])
            idx_pairs.append((i, j))

    kwargs: Dict[str, Any] = {
        "lang": lang,
        "batch_size": batch_size,
        "verbose": verbose,
        "rescale_with_baseline": rescale_with_baseline,
    }
    if model_type is not None:
        kwargs["model_type"] = model_type
    if device is not None:
        kwargs["device"] = device

    _, _, f1 = bert_score_fn(cands, refs, **kwargs)
    f1_np = f1.detach().cpu().numpy() if hasattr(f1, "detach") else np.asarray(f1)

    matrix = np.eye(n)
    for (i, j), score in zip(idx_pairs, f1_np):
        matrix[i, j] = matrix[j, i] = float(score)
    return matrix


# ---------------------------------------------------------------------------
# Step 2b: pairwise sentence-transformer cosine (fast alternative)
# ---------------------------------------------------------------------------

def compute_pairwise_cosine(
    verbalized: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
    model: Optional[Any] = None,
) -> np.ndarray:
    """
    Pairwise cosine similarity of SentenceTransformer embeddings.

    Much faster than BERTScore (O(n) encodes + one matmul), and empirically
    correlates well with BERTScore F1 for short, factual sentences. Use this
    when the graph is large (n > ~500) and BERTScore is prohibitive.

    Note: thresholds calibrated for BERTScore F1 may not transfer directly
    to cosine similarity. Re-tune the threshold if you switch methods.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers and scikit-learn are required for "
            "method='sentence_transformer'."
        ) from exc

    if model is None:
        model = SentenceTransformer(model_name)
    embeds = model.encode(list(verbalized), show_progress_bar=False)
    sim = cosine_similarity(embeds)
    return np.clip(sim, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Step 3 + 4: threshold + connected components clustering
# ---------------------------------------------------------------------------

def cluster_by_threshold(
    sim_matrix: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """
    Cluster items by connected components of the thresholded similarity graph.

    An edge (i, j) exists iff sim_matrix[i, j] >= threshold. Two items are
    in the same cluster iff a path of such edges connects them. This is
    the right semantics for "if any pair is similar, group them" — it is
    transitive by construction.

    Implementation: union-find with path compression. O(n^2 * alpha(n)).

    Returns
    -------
    clusters : list of list of int
        Each inner list contains node indices belonging to one cluster.
        Cluster order is by smallest member index (stable / deterministic).
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # union by index keeps roots stable / deterministic
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(i)

    # Sort clusters by smallest member index for reproducibility
    return [sorted(members) for _, members in sorted(buckets.items())]


# ---------------------------------------------------------------------------
# Step 5: the metric
# ---------------------------------------------------------------------------

def calculate_triplet_uir(
    triples: Sequence[Triple],
    threshold: float = 0.8,
    method: str = "bertscore",
    *,
    # BERTScore-specific options
    lang: str = "en",
    model_type: Optional[str] = None,
    rescale_with_baseline: bool = False,
    batch_size: int = 64,
    device: Optional[str] = None,
    # SentenceTransformer-specific options
    st_model_name: str = "all-MiniLM-L6-v2",
    st_model: Optional[Any] = None,
    # Output controls
    verbose: bool = True,
    return_matrix: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the triplet-level UIR for a list of triples.

    Parameters
    ----------
    triples : sequence of triple
        Each triple is a 3-tuple/list or dict (see `verbalize_triple`).
    threshold : float, default 0.8
        Two triples are placed in the same cluster iff their pairwise
        similarity is >= this value. The specification uses 0.8 for BERTScore.
    method : {"bertscore", "sentence_transformer"}
        Similarity backend. "bertscore" is the specified one; the other is a
        faster approximation.
    lang, model_type, rescale_with_baseline, batch_size, device :
        Forwarded to bert_score when method="bertscore".
    st_model_name, st_model :
        Forwarded to SentenceTransformer when method="sentence_transformer".
    verbose : bool
        If True, print a summary and the most-redundant clusters.
    return_matrix : bool
        If False, the n*n similarity matrix is omitted from `details`
        (useful for very large graphs to keep results JSON-serialisable).

    Returns
    -------
    uir : float
        UIR = num_clusters / num_triples, in [0, 1]. Higher = less redundancy.
    details : dict
        {
          "triple_count":         int,
          "cluster_count":        int,
          "uir":                  float,
          "redundancy":           float,        # 1 - uir, for convenience
          "clusters":             list of cluster dicts (sorted by size desc),
          "redundant_clusters":   subset of clusters with size > 1,
          "singleton_count":      int,
          "verbalized":           list of str,
          "similarity_matrix":    np.ndarray   (omitted if return_matrix=False),
          "method":               str,
          "threshold":            float,
        }
    """
    n = len(triples)

    # --- Edge cases ---
    if n == 0:
        details = {
            "triple_count": 0,
            "cluster_count": 0,
            "uir": 1.0,
            "redundancy": 0.0,
            "clusters": [],
            "redundant_clusters": [],
            "singleton_count": 0,
            "verbalized": [],
            "method": method,
            "threshold": threshold,
        }
        if return_matrix:
            details["similarity_matrix"] = np.zeros((0, 0))
        return 1.0, details

    verbalized = [verbalize_triple(t) for t in triples]

    if n == 1:
        details = {
            "triple_count": 1,
            "cluster_count": 1,
            "uir": 1.0,
            "redundancy": 0.0,
            "clusters": [{
                "cluster_id": 0, "size": 1,
                "members": [{"index": 0, "triple": triples[0], "text": verbalized[0]}],
            }],
            "redundant_clusters": [],
            "singleton_count": 1,
            "verbalized": verbalized,
            "method": method,
            "threshold": threshold,
        }
        if return_matrix:
            details["similarity_matrix"] = np.eye(1)
        return 1.0, details

    # --- Pairwise similarity ---
    if method == "bertscore":
        sim = compute_pairwise_bertscore(
            verbalized,
            lang=lang,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
    elif method == "sentence_transformer":
        sim = compute_pairwise_cosine(
            verbalized, model_name=st_model_name, model=st_model,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'bertscore' or 'sentence_transformer'."
        )

    # --- Cluster ---
    clusters_raw = cluster_by_threshold(sim, threshold)
    num_clusters = len(clusters_raw)
    uir = num_clusters / n

    # --- Build human-readable cluster records ---
    clusters: List[Dict[str, Any]] = []
    for cid, member_ids in enumerate(clusters_raw):
        # Within-cluster cohesion = average pairwise similarity (for inspection).
        if len(member_ids) > 1:
            sub = sim[np.ix_(member_ids, member_ids)]
            iu = np.triu_indices(len(member_ids), k=1)
            cohesion = float(np.mean(sub[iu]))
        else:
            cohesion = 1.0

        clusters.append({
            "cluster_id": cid,
            "size": len(member_ids),
            "cohesion": round(cohesion, 6),
            "members": [
                {"index": i, "triple": triples[i], "text": verbalized[i]}
                for i in member_ids
            ],
        })

    # Sort clusters by size desc (then by first member index) for inspection
    clusters.sort(key=lambda c: (-c["size"], c["members"][0]["index"]))
    redundant_clusters = [c for c in clusters if c["size"] > 1]
    singleton_count = sum(1 for c in clusters if c["size"] == 1)

    if verbose:
        print("=" * 64)
        print("Triplet UIR Analysis")
        print(f"  Triples:            {n}")
        print(f"  Clusters:           {num_clusters}")
        print(f"  UIR:                {uir:.4f}   ({num_clusters}/{n})")
        print(f"  Redundancy (1-UIR): {1.0 - uir:.4f}")
        print(f"  Redundant clusters: {len(redundant_clusters)}")
        print(f"  Singletons:         {singleton_count}")
        print(f"  Method:             {method}  |  threshold = {threshold}")
        print("=" * 64)
        if redundant_clusters:
            print("\nTop redundant clusters:")
            for c in redundant_clusters[:5]:
                print(f"  cluster {c['cluster_id']}  size={c['size']}  cohesion={c['cohesion']:.4f}")
                for m in c["members"]:
                    print(f"    - {m['text']}")

    details: Dict[str, Any] = {
        "triple_count": n,
        "cluster_count": num_clusters,
        "uir": uir,
        "redundancy": 1.0 - uir,
        "clusters": clusters,
        "redundant_clusters": redundant_clusters,
        "singleton_count": singleton_count,
        "verbalized": verbalized,
        "method": method,
        "threshold": threshold,
    }
    if return_matrix:
        details["similarity_matrix"] = sim

    return uir, details


# ---------------------------------------------------------------------------
# Self-test on the worked example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reproduces the worked example: 4 triples, 3 of which paraphrase one fact,
    # and 1 unrelated triple. Expected: 2 clusters, UIR = 0.5.
    example_triples = [
        ("Eiffel Tower", "is located in", "Paris"),
        ("Eiffel Tower", "stands in", "the city of Paris"),
        ("Paris", "is the location of", "Eiffel Tower"),
        ("Louvre", "exhibits", "the Mona Lisa"),
    ]

    # Default uses BERTScore. Swap to sentence_transformer if you want a quick
    # local sanity check without downloading roberta-large.
    uir, details = calculate_triplet_uir(
        example_triples,
        threshold=0.8,
        method="bertscore",
        verbose=True,
    )
    print(f"\nFinal UIR: {uir:.4f}  (expected ~0.5)")
