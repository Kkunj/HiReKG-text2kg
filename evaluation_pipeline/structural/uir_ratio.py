import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd
from typing import Tuple, List

# Global model cache to avoid reloading
_SBERT_MODEL_CACHE = {}

def get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or load the Sentence-BERT model (cached for efficiency).
    
    Args:
        model_name: Name of the sentence-transformers model to use.
                   Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (accurate)
    
    Returns:
        Loaded SentenceTransformer model
    """
    global _SBERT_MODEL_CACHE
    if model_name not in _SBERT_MODEL_CACHE:
        print(f"Loading Sentence-BERT model: {model_name}...")
        _SBERT_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_MODEL_CACHE[model_name]


def split_to_edges(predictions):
    """
    Convert predictions (in various formats) to list of edge strings.
    Each prediction can be a dict or list representing a graph/triplets.
    """
    all_edges = []
    
    for pred in predictions:
        edges = []
        
        if isinstance(pred, list):
            # Handle list format (like kggen/rakg)
            for item in pred:
                if isinstance(item, list) and len(item) >= 3:
                    # Format: [source, relation, target, ...]
                    edge = f"{item[0]};{item[1]};{item[2]}"
                    edges.append(edge.lower().strip())
                elif isinstance(item, dict):
                    # Handle dict in list
                    if 'source' in item and 'target' in item and 'relation_type' in item:
                        edge = f"{item['source']};{item['relation_type']};{item['target']}"
                        edges.append(edge.lower().strip())
                    elif 'subject' in item and 'object' in item and 'relation' in item:
                        edge = f"{item['subject']};{item['relation']};{item['object']}"
                        edges.append(edge.lower().strip())
        
        elif isinstance(pred, dict):
            # Handle single dict format
            if 'source' in pred and 'target' in pred and 'relation_type' in pred:
                edge = f"{pred['source']};{pred['relation_type']};{pred['target']}"
                edges.append(edge.lower().strip())
            elif 'subject' in pred and 'object' in pred and 'relation' in pred:
                edge = f"{pred['subject']};{pred['relation']};{pred['object']}"
                edges.append(edge.lower().strip())
        
        all_edges.append(edges if edges else [])
    
    return all_edges


def compute_similarity_matrix(
    edges: List[str], 
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute the NxN cosine similarity matrix using Sentence-BERT embeddings.
    
    This is EXTREMELY fast compared to BERT score because:
    1. All texts are encoded to embeddings in a single batch operation
    2. Cosine similarity is computed via matrix multiplication (nearly instant)
    3. No pairwise model inference needed
    
    Args:
        edges: List of edge strings to compute similarities for
        model_name: Sentence-BERT model to use. Options:
                   - "all-MiniLM-L6-v2" (fast, 384 dim, good quality)
                   - "all-mpnet-base-v2" (slower, 768 dim, best quality)
                   - "paraphrase-MiniLM-L6-v2" (optimized for paraphrase detection)
        batch_size: Batch size for encoding (higher = faster but more memory)
        verbose: Whether to show progress information
    
    Returns:
        NxN numpy array where entry [i,j] is the cosine similarity between edge i and edge j
    """
    n = len(edges)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.array([[1.0]])
    
    # Load model (cached)
    model = get_sbert_model(model_name)
    
    # Step 1: Encode all edges to embeddings in a single batch
    if verbose:
        print(f"Encoding {n} edges to embeddings using Sentence-BERT...")
    
    embeddings = model.encode(
        edges, 
        batch_size=batch_size, 
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=True  # Pre-normalize for faster cosine similarity
    )
    
    # Step 2: Compute cosine similarity matrix (nearly instant with normalized embeddings)
    # Since embeddings are normalized, cosine_sim = dot product
    if verbose:
        print(f"Computing {n}x{n} similarity matrix...")
    
    similarity_matrix = cosine_similarity(embeddings)
    
    if verbose:
        print(f"Similarity matrix computed: shape {similarity_matrix.shape}")
    
    return similarity_matrix.astype(np.float32)


def compute_bert_score_matrix(
    edges: List[str], 
    batch_size: int = 64,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute the NxN similarity matrix using BERTScore.
    
    This is slower than Sentence-BERT but may provide different similarity scores.
    Pre-computes all pairwise similarities upfront for efficient clustering.
    
    Args:
        edges: List of edge strings to compute similarities for
        batch_size: Number of pairs to process per batch (for memory efficiency)
        verbose: Whether to show progress information
    
    Returns:
        NxN numpy array where entry [i,j] is the BERT F1 similarity between edge i and edge j
    """
    # Lazy import
    try:
        from bert_score import score as score_bert
    except ImportError:
        raise ImportError(
            "bert_score package is required for BERT model. "
            "Install with: pip install bert_score\n"
            "Or use --model sbert for faster Sentence-BERT."
        )
    
    n = len(edges)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.array([[1.0]])
    
    # Initialize similarity matrix with zeros
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Diagonal is always 1.0 (edge is identical to itself)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Generate all unique pairs (i, j) where i < j
    # We only need to compute upper triangle since similarity is symmetric
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(pairs)
    
    if verbose:
        print(f"Computing BERTScore for {total_pairs} unique pairs ({n} edges)...")
    
    # Process pairs in batches
    for batch_start in tqdm(range(0, total_pairs, batch_size), 
                            desc="Computing BERT similarities", 
                            disable=not verbose):
        batch_end = min(batch_start + batch_size, total_pairs)
        batch_pairs = pairs[batch_start:batch_end]
        
        # Prepare batch
        candidates = [edges[i] for i, j in batch_pairs]
        references = [edges[j] for i, j in batch_pairs]
        
        # Single batched BERT call
        _, _, f1_scores = score_bert(
            candidates, references,
            model_type="bert-base-uncased",
            lang='en',
            verbose=False
        )
        
        # Fill in the similarity matrix (both [i,j] and [j,i] since symmetric)
        for idx, (i, j) in enumerate(batch_pairs):
            sim = f1_scores[idx].item()
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    if verbose:
        print(f"BERTScore matrix computed: shape {similarity_matrix.shape}")
    
    return similarity_matrix


def calculate_uir_from_matrix(
    similarity_matrix: np.ndarray,
    edges: List[str] = None,
    similarity_threshold: float = 0.80,
    verbose: bool = True
) -> Tuple[float, int, int, List]:
    """
    Calculate Unique Information Ratio using a pre-computed similarity matrix.

    Clustering: CONNECTED COMPONENTS of the thresholded similarity graph.
    There is an edge between i and j iff similarity_matrix[i, j] >=
    similarity_threshold; two predictions land in the same cluster iff a
    path of such edges connects them. Properties:
        - Transitive: if A~B and B~C, then A, B, C share a cluster even
          when A~C is below the threshold (paraphrase chains collapse).
        - Order-independent: shuffling the input never changes the result.
        - Reproducible: identical inputs produce identical clusters.

    This differs from the original greedy/leader clustering, which compared
    each new edge only to existing cluster representatives -- making the
    result order-dependent and unable to merge paraphrase chains.

    Implementation: union-find with path compression. O(n^2 * alpha(n)),
    dominated by the upper-triangular scan over the similarity matrix.

    Args:
        similarity_matrix: NxN matrix where entry [i,j] is the cosine
                          similarity between prediction i and prediction j.
        edges: Optional list of edge strings (used to return readable
               cluster representatives; otherwise indices are returned).
        similarity_threshold: Threshold for considering predictions similar (default: 0.80)
        verbose: Whether to print progress information

    Returns:
        uir: Unique Information Ratio (unique clusters / total predictions)
        num_unique_clusters: Number of unique clusters found
        total_predictions: Total number of predictions
        cluster_representatives: One representative per cluster (the
            smallest-index member, deterministic). Strings if `edges` is
            provided, indices otherwise.
    """
    n = similarity_matrix.shape[0]

    if n == 0:
        if verbose:
            print('No predictions found')
        return 1.0, 0, 0, []

    if n == 1:
        if edges:
            return 1.0, 1, 1, [edges[0]]
        return 1.0, 1, 1, [0]

    # --- Union-find with path compression ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Attach the larger root index under the smaller one. This keeps
        # the smallest member of each cluster as its canonical root,
        # giving deterministic representatives regardless of merge order.
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    # Union every pair above the threshold. Walk the upper triangle only,
    # since the matrix is symmetric. Each row is processed with a vectorised
    # comparison, so the inner loop only fires on actual matches.
    iterator = range(n - 1)
    if verbose:
        iterator = tqdm(iterator, desc="Clustering edges (connected components)")

    for i in iterator:
        row_tail = similarity_matrix[i, i + 1:]
        match_offsets = np.flatnonzero(row_tail >= similarity_threshold)
        for off in match_offsets:
            union(i, int(i + 1 + off))

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({'row': i + 1, 'matches': int(match_offsets.size)})

    # --- Collect clusters by root ---
    root_to_members: dict = {}
    for i in range(n):
        r = find(i)
        root_to_members.setdefault(r, []).append(i)

    # Sort clusters by smallest member index for reproducibility, and
    # take the smallest index in each cluster as its representative.
    cluster_member_lists = [sorted(members) for _, members in sorted(root_to_members.items())]
    cluster_representative_indices = [members[0] for members in cluster_member_lists]

    num_unique_clusters = len(cluster_representative_indices)
    total_predictions = n
    uir = num_unique_clusters / total_predictions

    # Return edge strings if provided, otherwise return indices
    if edges:
        cluster_representatives = [edges[idx] for idx in cluster_representative_indices]
    else:
        cluster_representatives = cluster_representative_indices

    return uir, num_unique_clusters, total_predictions, cluster_representatives


def calculate_uir(
    pred_graph, 
    similarity_threshold=0.80, 
    flag=0, 
    verbose=True, 
    batch_size=256,
    model_type="sbert",
    sbert_model_name="all-MiniLM-L6-v2"
):
    """
    Calculate Unique Information Ratio using pre-computed similarity matrix.
    
    Supports two similarity computation methods:
    - "sbert": Sentence-BERT (fast, recommended)
    - "bert": BERTScore (slower, may provide different similarity scores)
    
    Args:
        pred_graph: List of predictions (dicts or lists)
        similarity_threshold: Similarity threshold for clustering (default: 0.80)
        flag: If 0, convert pred_graph using split_to_edges; if 1, use as-is
        verbose: Whether to show progress bar
        batch_size: Batch size for computing similarities
        model_type: "sbert" for Sentence-BERT (fast) or "bert" for BERTScore (slower)
        sbert_model_name: Sentence-BERT model to use (only if model_type="sbert"):
                         - "all-MiniLM-L6-v2" (fast, good quality - default)
                         - "all-mpnet-base-v2" (slower, best quality)
                         - "paraphrase-MiniLM-L6-v2" (optimized for paraphrase)
    
    Returns:
        uir: Unique Information Ratio (unique clusters / total predictions)
        num_unique_clusters: Number of unique clusters found
        total_predictions: Total number of predictions
        cluster_representatives: List of unique cluster representative edges
    """
    if not flag:
        pred_edges = split_to_edges([pred_graph])[0]
    else:
        pred_edges = pred_graph
    
    if len(pred_edges) == 0:
        if verbose:
            print('No edges found')
        return 1.0, 0, 0, []
    
    if len(pred_edges) == 1:
        return 1.0, 1, 1, [pred_edges[0]]
    
    # Step 1: Compute similarity matrix using selected model
    if model_type.lower() == "sbert":
        if verbose:
            print(f"\nStep 1: Computing Sentence-BERT similarity matrix for {len(pred_edges)} edges...")
            print(f"        Model: {sbert_model_name}")
        
        similarity_matrix = compute_similarity_matrix(
            pred_edges, 
            model_name=sbert_model_name,
            batch_size=batch_size, 
            verbose=verbose
        )
    elif model_type.lower() == "bert":
        if verbose:
            print(f"\nStep 1: Computing BERTScore similarity matrix for {len(pred_edges)} edges...")
            print(f"        (This may take a while...)")
        
        similarity_matrix = compute_bert_score_matrix(
            pred_edges, 
            batch_size=batch_size, 
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'sbert' or 'bert'.")
    
    # Step 2: Perform clustering using the pre-computed matrix (instant)
    if verbose:
        print(f"\nStep 2: Clustering using pre-computed similarities...")
    
    uir, num_unique_clusters, total_predictions, cluster_representatives = calculate_uir_from_matrix(
        similarity_matrix,
        edges=pred_edges,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
    
    return uir, num_unique_clusters, total_predictions, cluster_representatives


def calculate_uir_legacy(pred_graph, similarity_threshold=0.80, flag=0, verbose=True):
    """
    Legacy UIR calculation using BERTScore (DEPRECATED - much slower).
    
    NOTE: This function requires `bert_score` package which is no longer imported by default.
    To use this function, add: `from bert_score import score as score_bert` at the top.
    
    Use calculate_uir() instead for 10-100x faster performance with Sentence-BERT.
    
    Args:
        pred_graph: List of predictions (dicts or lists)
        similarity_threshold: BERT score threshold for clustering (default: 0.80)
        flag: If 0, convert pred_graph using split_to_edges; if 1, use as-is
        verbose: Whether to show progress bar
    
    Returns:
        uir: Unique Information Ratio (unique clusters / total predictions)
        num_unique_clusters: Number of unique clusters found
        total_predictions: Total number of predictions
        cluster_representatives: List of unique cluster representative edges
    """
    # Lazy import for legacy support
    try:
        from bert_score import score as score_bert
    except ImportError:
        raise ImportError(
            "bert_score package is required for legacy function. "
            "Install with: pip install bert_score\n"
            "Or use calculate_uir() which uses faster Sentence-BERT."
        )
    
    if not flag:
        pred_edges = split_to_edges([pred_graph])[0]
    else:
        pred_edges = pred_graph
    
    if len(pred_edges) == 0:
        print('No edges found')
        return 1.0, 0, 0, []
    
    if len(pred_edges) == 1:
        return 1.0, 1, 1, [pred_edges[0]]
    
    # Start with first prediction as first cluster representative
    cluster_representatives = [pred_edges[0]]
    cluster_assignments = [0]  # First pred belongs to cluster 0
    
    # Process remaining edges with progress bar
    remaining_edges = pred_edges[1:]
    pbar = tqdm(remaining_edges, desc="Clustering edges (legacy)", disable=not verbose)
    
    for pred in pbar:
        if len(cluster_representatives) == 0:
            cluster_representatives.append(pred)
            cluster_assignments.append(0)
            continue
        
        # BATCHED: Compare this edge against ALL cluster representatives at once
        candidates = [pred] * len(cluster_representatives)
        references = cluster_representatives
        
        # Single batched BERT call for all comparisons
        _, _, similarities = score_bert(
            candidates, references,
            model_type="bert-base-uncased",
            lang='en',
            verbose=False
        )
        
        # Find max similarity and check threshold
        max_sim, max_idx = similarities.max(dim=0)
        
        if max_sim.item() >= similarity_threshold:
            # Assign to existing cluster
            cluster_assignments.append(max_idx.item())
        else:
            # Create new cluster
            cluster_representatives.append(pred)
            cluster_assignments.append(len(cluster_representatives) - 1)
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'clusters': len(cluster_representatives),
            'UIR': f"{len(cluster_representatives) / (len(cluster_assignments)):.3f}"
        })
    
    num_unique_clusters = len(cluster_representatives)
    total_predictions = len(pred_edges)
    uir = num_unique_clusters / total_predictions
    
    return uir, num_unique_clusters, total_predictions, cluster_representatives


def load_kggen_relations(directory):
    """Load relations from KGgen output files."""
    relations = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json") and f != "batch_summary.json"]
    
    for file in tqdm(json_files, desc="Loading KGgen files"):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        if "pipeline_result" in preds and preds["pipeline_result"] is not None:
            if "relations" in preds["pipeline_result"]:
                # KGgen format: relations are lists [source, relation, target]
                rels = preds["pipeline_result"]["relations"]
                for rel in rels:
                    if isinstance(rel, list) and len(rel) >= 3:
                        edge = f"{rel[0]};{rel[1]};{rel[2]}"
                        relations.append(edge.lower().strip())
    
    return relations


def load_itext2kg_relations(directory):
    """Load relations from iText2KG output files."""
    relations = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json") and f != "batch_summary.json"]
    
    for file in tqdm(json_files, desc="Loading iText2KG files"):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        if "pipeline_result" in preds and preds["pipeline_result"] is not None:
            if "relations" in preds["pipeline_result"]:
                # iText2KG format: relations are dicts {source, target, relation_type}
                rels = preds["pipeline_result"]["relations"]
                for rel in rels:
                    if isinstance(rel, dict) and 'source' in rel and 'target' in rel and 'relation_type' in rel:
                        edge = f"{rel['source']};{rel['relation_type']};{rel['target']}"
                        relations.append(edge.lower().strip())
    
    return relations


def load_rakg_relations(directory):
    """Load relations from RAKG output files."""
    relations = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json") and f != "batch_summary.json"]
    
    for file in tqdm(json_files, desc="Loading RAKG files"):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        if "pipeline_result" in preds and preds["pipeline_result"] is not None:
            if "edges" in preds["pipeline_result"]:
                # RAKG format: edges are lists [source, relation, target, description]
                edges = preds["pipeline_result"]["edges"]
                for edge in edges:
                    if isinstance(edge, list) and len(edge) >= 3:
                        edge_str = f"{edge[0]};{edge[1]};{edge[2]}"
                        relations.append(edge_str.lower().strip())
    
    return relations


def load_ours_relations(directory):
    """Load relations from our approach output files."""
    relations = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json") and f != "batch_summary.json"]
    
    for file in tqdm(json_files, desc="Loading Ours files"):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        if "pipeline_result" in preds and preds["pipeline_result"] is not None:
            if "triples_final" in preds["pipeline_result"]:
                # Our format: triples_final are dicts {subject, relation, object}
                triples = preds["pipeline_result"]["triples_final"]
                for triple in triples:
                    if isinstance(triple, dict) and 'subject' in triple and 'object' in triple and 'relation' in triple:
                        edge = f"{triple['subject']};{triple['relation']};{triple['object']}"
                        relations.append(edge.lower().strip())
    
    return relations


def run_comparison(
    approaches_config, 
    similarity_threshold=0.80, 
    model_type="sbert",
    sbert_model_name="all-MiniLM-L6-v2"
):
    """
    Run UIR comparison across multiple approaches.
    
    Args:
        approaches_config: Dict mapping approach name to (directory, loader_function)
        similarity_threshold: Similarity threshold for clustering
        model_type: "sbert" for Sentence-BERT (fast) or "bert" for BERTScore (slower)
        sbert_model_name: Sentence-BERT model to use (only if model_type="sbert")
    
    Returns:
        results: Dict with UIR results for each approach
    """
    results = {}
    
    for approach_name, (directory, loader_func) in approaches_config.items():
        print(f"\n{'='*60}")
        print(f"Processing: {approach_name}")
        print(f"{'='*60}")
        
        # Load relations
        print(f"\nLoading relations from: {directory}")
        relations = loader_func(directory)
        print(f"Loaded {len(relations)} relations")
        
        if len(relations) == 0:
            print(f"WARNING: No relations found for {approach_name}")
            results[approach_name] = {
                'uir': None,
                'unique_clusters': 0,
                'total_relations': 0,
                'cluster_representatives': []
            }
            continue
        
        # Calculate UIR
        print(f"\nCalculating UIR for {approach_name}...")
        uir, num_unique, total, cluster_reps = calculate_uir(
            relations, 
            similarity_threshold=similarity_threshold, 
            flag=1,  # Already converted to edge strings
            verbose=True,
            model_type=model_type,
            sbert_model_name=sbert_model_name
        )
        
        results[approach_name] = {
            'uir': uir,
            'unique_clusters': num_unique,
            'total_relations': total,
            'cluster_representatives': cluster_reps
        }
        
        print(f"\n{approach_name} Results:")
        print(f"  UIR: {uir:.4f}")
        print(f"  Unique clusters: {num_unique}")
        print(f"  Total relations: {total}")
    
    return results


def print_comparison_table(results):
    """Print a comparison table of UIR results."""
    print(f"\n{'='*80}")
    print("UIR COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Approach':<20} {'Total Relations':>15} {'Unique Clusters':>15} {'UIR':>10}")
    print(f"{'-'*80}")
    
    for approach, data in results.items():
        if data['uir'] is not None:
            print(f"{approach:<20} {data['total_relations']:>15} {data['unique_clusters']:>15} {data['uir']:>10.4f}")
        else:
            print(f"{approach:<20} {'N/A':>15} {'N/A':>15} {'N/A':>10}")
    
    print(f"{'='*80}")
    
    # Higher UIR = less redundancy = better
    print("\nInterpretation: Higher UIR means less redundancy (more unique information)")


def plot_comparison(results, output_path=None):
    """Create a bar chart comparing UIR across approaches."""
    approaches = []
    uirs = []
    total_rels = []
    unique_clusters = []
    
    for approach, data in results.items():
        if data['uir'] is not None:
            approaches.append(approach)
            uirs.append(data['uir'])
            total_rels.append(data['total_relations'])
            unique_clusters.append(data['unique_clusters'])
    
    if len(approaches) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(approaches)]
    
    # Plot 1: UIR comparison
    ax1 = axes[0]
    bars1 = ax1.bar(approaches, uirs, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Unique Information Ratio (UIR)', fontsize=12)
    ax1.set_title('UIR Comparison\n(Higher = Less Redundancy)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, uirs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # Plot 2: Total relations
    ax2 = axes[1]
    bars2 = ax2.bar(approaches, total_rels, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Total Relations', fontsize=12)
    ax2.set_title('Total Relations Extracted', fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, total_rels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_rels)*0.02, 
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # Plot 3: Unique clusters
    ax3 = axes[2]
    bars3 = ax3.bar(approaches, unique_clusters, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Unique Clusters', fontsize=12)
    ax3.set_title('Unique Information Clusters', fontsize=14, fontweight='bold')
    for bar, val in zip(bars3, unique_clusters):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(unique_clusters)*0.02, 
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # CLI argument parsing
    parser = argparse.ArgumentParser(
        description="UIR (Unique Information Ratio) Analysis for Knowledge Graph Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Sentence-BERT (fast, default)
  python uir_redoc.py --model sbert
  
  # Use BERTScore (slower but different similarity measure)
  python uir_redoc.py --model bert
  
  # Use specific Sentence-BERT model
  python uir_redoc.py --model sbert --sbert-model all-mpnet-base-v2
  
  # Custom similarity threshold
  python uir_redoc.py --model sbert --threshold 0.85
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["sbert", "bert"],
        default="sbert",
        help="Model to use for similarity computation: 'sbert' (Sentence-BERT, fast) or 'bert' (BERTScore, slower). Default: sbert"
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-BERT model name (only used if --model sbert). Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (best quality), paraphrase-MiniLM-L6-v2. Default: all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.80,
        help="Similarity threshold for clustering (0.0 to 1.0). Default: 0.80"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="redundancy_analysis",
        help="Directory to save output files. Default: redundancy_analysis"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating comparison plot"
    )
    
    args = parser.parse_args()
    
    # Define the directories for each approach
    approaches_config = {
        "KGgen": (
            r"C:\<PROJECT_ROOT>\multimodal_RAG\GraphRAG\baseline\kggen\redocred_kggen_batch_results",
            load_kggen_relations
        ),
        "iText2KG": (
            r"C:\<PROJECT_ROOT>\multimodal_RAG\GraphRAG\baseline\itext2kg\redocred_itext2kg_batch_results",
            load_itext2kg_relations
        ),
        "RAKG": (
            r"C:\<PROJECT_ROOT>\multimodal_RAG\GraphRAG\baseline\RAKG\redocred_rakg_batch_results",
            load_rakg_relations
        ),
        "Ours": (
            r"C:\<PROJECT_ROOT>\multimodal_RAG\GraphRAG\GraphCreation\Final_Graph_Creation\redocred_batch_results",
            load_ours_relations
        )
    }
    
    # Run comparison
    print("="*80)
    print("UNIQUE INFORMATION RATIO (UIR) ANALYSIS")
    print("Comparing redundancy across different KG extraction approaches")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model.upper()}" + (f" ({args.sbert_model})" if args.model == "sbert" else ""))
    print(f"  Similarity Threshold: {args.threshold}")
    print(f"  Output Directory: {args.output_dir}")
    print("="*80)
    
    results = run_comparison(
        approaches_config, 
        similarity_threshold=args.threshold,
        model_type=args.model,
        sbert_model_name=args.sbert_model
    )
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'uir_comparison_results_{args.model}.csv')
    results_df = pd.DataFrame([
        {
            'Approach': k,
            'Total_Relations': v['total_relations'],
            'Unique_Clusters': v['unique_clusters'],
            'UIR': v['uir'],
            'Model': args.model,
            'Threshold': args.threshold
        }
        for k, v in results.items()
    ])
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Plot comparison
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, f'uir_comparison_plot_{args.model}.png')
        plot_comparison(results, output_path=plot_path)
    
    # Print sample cluster representatives for each approach
    print("\n" + "="*80)
    print("SAMPLE CLUSTER REPRESENTATIVES (first 5 per approach)")
    print("="*80)
    for approach, data in results.items():
        if data['cluster_representatives']:
            print(f"\n{approach}:")
            for i, rep in enumerate(data['cluster_representatives'][:5]):
                print(f"  {i+1}. {rep}")
