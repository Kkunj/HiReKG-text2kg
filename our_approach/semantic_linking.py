"""
Semantic Linking Pipeline - Adds implicit relations between entities using embeddings and clustering.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
from sklearn.cluster import AgglomerativeClustering
import hdbscan

from llm_client import LLMClient, EmbeddingClient

logger = logging.getLogger("KGPipeline")
from spacy.language import Language

def _get_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """
    Load a spaCy model with sentence boundaries.
    Falls back to a blank English pipeline with a sentencizer if the model is missing.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

def get_embeddings(
    texts: List[str],
    embedding_client: Optional[EmbeddingClient] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the shared EmbeddingClient.

    Args:
        texts: List of text strings to embed
        embedding_client: EmbeddingClient instance (creates a default local one if not provided)

    Returns:
        List of embedding vectors
    """
    logger.debug(f"Generating embeddings for {len(texts)} texts")

    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []

    if embedding_client is None:
        embedding_client = EmbeddingClient()
        logger.debug("Created default EmbeddingClient (local BAAI/bge-m3)")

    all_embeddings = embedding_client.embed_documents(texts)

    logger.debug(f"Successfully generated {len(all_embeddings)} embeddings")
    return all_embeddings


def find_entity_context_from_triples(
    entity: str, 
    triples: List[Tuple[str, str, str, Optional[str]]], 
    max_contexts: int = 5
) -> List[str]:
    """
    Find context for an entity by collecting evidence from triples where it appears.
    
    Args:
        entity: Entity name to search for
        triples: List of (subject, relation, object, evidence) tuples
        max_contexts: Maximum number of evidence texts to collect
    
    Returns:
        List of evidence/context strings containing the entity
    """
    logger.debug(f"Finding context for entity '{entity}' from triples")
    contexts = []
    entity_lower = entity.lower()
    
    for triple in triples:
        subject, relation, obj, evidence = triple
        
        # Check if entity appears in subject or object
        if entity_lower == subject.lower() or entity_lower == obj.lower():
            # Use evidence if available
            if evidence and evidence.strip():
                contexts.append(evidence.strip())
                logger.debug(f"Found evidence for '{entity}': {evidence[:50]}...")
            else:
                # Fallback: create context from triple itself
                context = f"{subject} {relation.replace('_', ' ')} {obj}"
                contexts.append(context)
                logger.debug(f"No evidence, using triple: {context[:50]}...")
            
            if len(contexts) >= max_contexts:
                break
    
    # If no context found, return the entity itself
    if not contexts:
        logger.warning(f"No context found for entity '{entity}', using entity name as fallback")
        contexts = [entity]
    
    logger.debug(f"Collected {len(contexts)} context(s) for entity '{entity}'")
    return contexts


def generate_entity_embeddings(
    entities: Sequence[str],
    triples_with_evidence: List[Tuple[str, str, str, Optional[str]]],
    embedding_client: Optional[EmbeddingClient] = None,
) -> List[Dict[str, Any]]:
    """
    PHASE 2: Generate embeddings for entities and their contexts from triple evidence.

    Args:
        entities: List of canonical entity names
        triples_with_evidence: List of (subject, relation, object, evidence) tuples
        embedding_client: EmbeddingClient instance

    Returns:
        List of dicts with entity, entity_embedding, and context_embedding
    """
    logger.debug(f"PHASE 2: Generating embeddings for {len(entities)} entities using triple evidence")
    logger.debug(f"   -> Generating embeddings for {len(entities)} entities...")

    # Prepare texts for embedding
    entity_texts = list(entities)
    context_texts = []

    logger.debug("Finding context from triple evidence for each entity")
    for entity in entities:
        # Get contexts from triples where this entity appears
        contexts = find_entity_context_from_triples(entity, triples_with_evidence, max_contexts=5)
        # Combine multiple evidence texts into one context
        combined_context = " ".join(contexts[:3])  # Use up to 3 pieces of evidence
        context_texts.append(combined_context)
        logger.debug(f"Entity '{entity}' combined context length: {len(combined_context)} chars")

    # Get embeddings in batch
    logger.debug(f"Getting embeddings for {len(entity_texts)} entities + {len(context_texts)} contexts")
    all_texts = entity_texts + context_texts
    all_embeddings = get_embeddings(all_texts, embedding_client=embedding_client)
    
    # Split embeddings
    entity_embeddings = all_embeddings[:len(entities)]
    context_embeddings = all_embeddings[len(entities):]
    logger.debug(f"Split embeddings: {len(entity_embeddings)} entity embeddings, {len(context_embeddings)} context embeddings")
    
    # Build result
    entity_embedding_table = []
    for i, entity in enumerate(entities):
        entity_embedding_table.append({
            "entity": entity,
            "entity_embedding": entity_embeddings[i],
            "context_embedding": context_embeddings[i],
            "context_text": context_texts[i]
        })
    
    logger.debug(f"Successfully created embedding table with {len(entity_embedding_table)} entries")
    return entity_embedding_table


def cluster_entities(
    entity_embedding_table: List[Dict[str, Any]],
    distance_threshold: float = 0.5,
    min_cluster_size: int = 3
) -> Dict[int, List[str]]:
    """
    PHASE 3: Cluster entities based on semantic similarity.
    
    Args:
        entity_embedding_table: List of entity embedding dicts
        distance_threshold: Clustering distance threshold (lower = tighter clusters)
        min_cluster_size: Minimum entities per cluster
    
    Returns:
        Dictionary mapping cluster_id to list of entity names
    """
    logger.debug(f"PHASE 3: Clustering {len(entity_embedding_table)} entities with distance_threshold={distance_threshold}")
    logger.debug(f"   → Clustering {len(entity_embedding_table)} entities...")
    
    if len(entity_embedding_table) < 2:
        logger.warning("Not enough entities to cluster (need at least 2)")
        return {}
    
    # Combine entity and context embeddings
    combined_vectors = []
    entity_names = []
    
    logger.debug("Combining entity and context embeddings")
    for item in entity_embedding_table:
        entity_vec = np.array(item["entity_embedding"])
        context_vec = np.array(item["context_embedding"])
        combined = (entity_vec + context_vec) / 2.0
        combined_vectors.append(combined)
        entity_names.append(item["entity"])
    
    X = np.array(combined_vectors)
    logger.debug(f"Combined embedding matrix shape: {X.shape}")
    
    # HDBSCAN clustering with euclidean distance
    logger.debug(f"Running HDBSCAN Clustering with min_cluster_size={min_cluster_size}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,  # Use the parameter passed to the function
        min_samples=1,                      # Allow fine structure
        metric='euclidean',                 # Best for SBERT / OpenAI embeddings
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.05      # Allows splitting dense blobs
    )
    
    labels = clusterer.fit_predict(X)
    logger.debug(f"Clustering produced {len(set(labels))} total clusters")
    
    # Group entities by cluster
    entity_clusters = {}
    for entity, label in zip(entity_names, labels):
        if label not in entity_clusters:
            entity_clusters[label] = []
        entity_clusters[label].append(entity)
    
    # Log cluster sizes before filtering
    logger.debug(f"Cluster sizes before filtering: {[(cid, len(ents)) for cid, ents in entity_clusters.items()]}")
    
    # Filter out clusters that are too small
    filtered_clusters = {
        cluster_id: entities
        for cluster_id, entities in entity_clusters.items()
        if len(entities) >= min_cluster_size
    }
    
    logger.debug(f"Found {len(filtered_clusters)} clusters with {min_cluster_size}+ entities (filtered from {len(entity_clusters)} total)")
    for cluster_id, entities in filtered_clusters.items():
        logger.debug(f"Cluster {cluster_id}: {entities}")
    
    logger.debug(f"   ✓ Found {len(filtered_clusters)} clusters with {min_cluster_size}+ entities")
    
    return filtered_clusters


SEMANTIC_RELATION_SYSTEM_PROMPT = """
You are a semantic relation expert. Given a cluster of related entities, Global summary and Cluster specific context,
identify meaningful semantic relationships between these entities.

Focus on these relation types:
- synonym / alias (entities that mean the same thing)
- part_of / has_part (component relationships)
- type_of / subtype_of (hierarchical relationships)
- relates_to (general semantic connection)
- is_process_of / is_state_of (process/state relationships)

CRITICAL: Return ONLY valid JSON. No comments, no trailing commas, all strings in double quotes.

Required format:
{
  "triples": [
    {
      "subject": "entity one",
      "relation": "relation_type",
      "object": "entity two",
      "reasoning": "brief explanation"
    }
  ]
}

Rules:
- Maximum 10 triples per response
- Only include relations that are semantically meaningful and supported by the context
- Double quotes only, no single quotes
- No trailing commas
- No comments in JSON
"""


def collect_cluster_evidence(
    cluster_entities: List[str],
    triples_with_evidence: List[Tuple[str, str, str, Optional[str]]],
    max_evidence_per_entity: int = 3
) -> str:
    """
    Collect evidence/context for a cluster by gathering evidence from triples
    where any of the cluster entities appear.
    
    Args:
        cluster_entities: List of entity names in the cluster
        triples_with_evidence: List of (subject, relation, object, evidence) tuples
        max_evidence_per_entity: Maximum evidence pieces to collect per entity
    
    Returns:
        Combined evidence text for the cluster
    """
    logger.debug(f"Collecting evidence for cluster with {len(cluster_entities)} entities")
    
    # Collect evidence for each entity in cluster
    cluster_evidence = []
    entity_lower_set = {e.lower() for e in cluster_entities}
    
    for entity in cluster_entities:
        entity_lower = entity.lower()
        entity_evidence_count = 0
        
        for triple in triples_with_evidence:
            subject, relation, obj, evidence = triple
            
            # Check if this entity appears in the triple
            if entity_lower == subject.lower() or entity_lower == obj.lower():
                # Use evidence if available
                if evidence and evidence.strip():
                    cluster_evidence.append(evidence.strip())
                    entity_evidence_count += 1
                    logger.debug(f"Found evidence for '{entity}' in cluster: {evidence[:50]}...")
                else:
                    # Fallback: use triple description
                    context = f"{subject} {relation.replace('_', ' ')} {obj}"
                    cluster_evidence.append(context)
                    entity_evidence_count += 1
                    logger.debug(f"Using triple as evidence for '{entity}': {context[:50]}...")
                
                if entity_evidence_count >= max_evidence_per_entity:
                    break
    
    # Combine all evidence, removing duplicates
    unique_evidence = list(dict.fromkeys(cluster_evidence))  # Preserve order while removing duplicates
    combined_evidence = " ".join(unique_evidence)
    
    logger.debug(f"Collected {len(unique_evidence)} unique evidence pieces for cluster (total {len(combined_evidence)} chars)")
    return combined_evidence


def filter_unconnected_entities(
    cluster_entities: List[str],
    existing_triples: List[Tuple[str, str, str]]
) -> List[str]:
    """
    Filter entities in a cluster to find those with NO existing connections.
    
    Args:
        cluster_entities: List of entity names in the cluster
        existing_triples: List of existing (subject, relation, object) triples
    
    Returns:
        List of entities that don't have connections with other entities in the cluster
    """
    logger.debug(f"Filtering {len(cluster_entities)} cluster entities against {len(existing_triples)} existing triples")
    
    # Build a set of entity pairs that already have connections
    connected_pairs = set()
    
    for subject, relation, obj in existing_triples:
        subject_lower = subject.lower()
        obj_lower = obj.lower()
        
        # Add both directions for undirected connection check
        connected_pairs.add((subject_lower, obj_lower))
        connected_pairs.add((obj_lower, subject_lower))
    
    logger.debug(f"Built connected pairs set with {len(connected_pairs)} entries")
    
    # Find entities with no connections to other cluster members
    unconnected = []
    cluster_lower = [e.lower() for e in cluster_entities]
    
    for i, entity in enumerate(cluster_entities):
        entity_lower = cluster_lower[i]
        has_connection = False
        
        # Check if this entity has connection with any other entity in cluster
        for j, other_entity in enumerate(cluster_entities):
            if i == j:
                continue
            other_lower = cluster_lower[j]
            
            if (entity_lower, other_lower) in connected_pairs:
                has_connection = True
                logger.debug(f"Entity '{entity}' has connection with '{other_entity}'")
                break
        
        if not has_connection:
            unconnected.append(entity)
            logger.debug(f"Entity '{entity}' has NO connections within cluster")
    
    logger.debug(f"Found {len(unconnected)} unconnected entities out of {len(cluster_entities)}")
    return unconnected


def generate_cluster_relations(
    cluster_entities: List[str],
    existing_triples: List[Tuple[str, str, str]],
    global_summary: str,
    llm_client: LLMClient,
    cluster_evidence: str = "",
    max_entities_per_batch: int = 10
) -> List[Dict[str, str]]:
    """
    PHASE 4: Generate semantic relations for unconnected entities in a cluster using LLM.
    
    Args:
        cluster_entities: List of entity names in the cluster
        existing_triples: List of existing triples to check for connections
        global_summary: Global document summary for context
        llm_client: LLM client for relation generation
        cluster_evidence: Combined evidence text for this cluster from triples
        max_entities_per_batch: Maximum entities to process per LLM call
    
    Returns:
        List of triple dicts with subject, relation, object, reasoning
    """
    logger.debug(f"Generating relations for cluster with {len(cluster_entities)} entities")
    
    if len(cluster_entities) < 2:
        logger.debug("Cluster too small (< 2 entities), skipping")
        return []
    
    # Filter to only entities without existing connections
    unconnected_entities = filter_unconnected_entities(cluster_entities, existing_triples)
    
    if len(unconnected_entities) < 2:
        logger.debug(f"Only {len(unconnected_entities)} unconnected entities, need at least 2. Skipping cluster.")
        return []
    
    logger.debug(f"Processing {len(unconnected_entities)} unconnected entities in batches of {max_entities_per_batch}")
    
    # Batch large clusters to avoid overwhelming the LLM
    all_triples = []
    num_batches = (len(unconnected_entities) + max_entities_per_batch - 1) // max_entities_per_batch
    
    for i in range(0, len(unconnected_entities), max_entities_per_batch):
        batch = unconnected_entities[i:i + max_entities_per_batch]
        batch_num = i // max_entities_per_batch + 1
        
        if len(batch) < 2:
            logger.debug(f"Batch {batch_num} has < 2 entities, skipping")
            continue
        
        logger.debug(f"Processing batch {batch_num}/{num_batches} with {len(batch)} entities")
        entities_str = ", ".join(f'"{e}"' for e in batch)
        
        # Build prompt with cluster evidence if available
        cluster_context = ""
        if cluster_evidence and cluster_evidence.strip():
            # Truncate if too long (keep first 1000 chars)
            evidence_text = cluster_evidence[:1000] + "..." if len(cluster_evidence) > 1000 else cluster_evidence
            cluster_context = f"""
        Cluster-specific context (evidence from triples where these entities appear):
        {evidence_text}
        """
            logger.debug(f"Including cluster evidence ({len(evidence_text)} chars) in prompt")
        
        user_prompt = f"""
        Global summary:
        {global_summary}

        Cluster specific context:
        {cluster_context}

        Entities in this cluster (with NO existing connections between them):
        {entities_str}

        Identify semantic relationships between these entities ONLY if they are meaningful and supported by the context and cluster-specific evidence.
        Consider their meanings and how they relate conceptually based on the global summary and cluster-specific context.
        Limit to 10 most important relations.
        """

        try:
            logger.debug(f"Calling LLM for batch {batch_num}")
            payload = llm_client.generate_json(
                SEMANTIC_RELATION_SYSTEM_PROMPT,
                user_prompt,
                max_output_tokens=10000
            )
            
            triples = payload.get("triples", []) or []
            logger.debug(f"Batch {batch_num} generated {len(triples)} triples")
            for triple in triples:
                logger.debug(f"  - {triple.get('subject')} -> {triple.get('relation')} -> {triple.get('object')}")
            all_triples.extend(triples)
        except Exception as e:
            logger.error(f"LLM Error in batch {batch_num}: {str(e)}", exc_info=True)
            logger.debug(f"⚠️  LLM Error (batch {batch_num}): {str(e)[:150]}")
            logger.debug(f"      Skipping this batch and continuing with remaining entities...")
            continue
    
    logger.debug(f"Generated total of {len(all_triples)} triples for this cluster")
    return all_triples

def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def split_into_chunks(
    text: str,
    sentences_per_chunk: int = 3,
) -> List[str]:
    """
    Split text into chunks containing roughly `sentences_per_chunk` sentences each.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    nlp = _get_spacy_model()
    doc = nlp(cleaned)
    sentences = [normalize_whitespace(sent.text) for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return [cleaned]

    chunks: List[str] = []
    for idx in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[idx : idx + sentences_per_chunk]
        chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)

    return chunks

def generate_semantic_relations(
    entity_clusters: Dict[int, List[str]],
    existing_triples: List[Tuple[str, str, str]],
    global_summary: str,
    llm_client: LLMClient,
    triples_with_evidence: Optional[List[Tuple[str, str, str, Optional[str]]]] = None
) -> List[Tuple[str, str, str]]:
    """
    PHASE 4-5: Generate and consolidate semantic relations from all clusters.
    
    Args:
        entity_clusters: Dictionary of cluster_id to entity lists
        existing_triples: List of existing triples to check for connections
        global_summary: Global document summary
        llm_client: LLM client
        triples_with_evidence: Optional list of triples with evidence for cluster context
    
    Returns:
        List of (subject, relation, object) tuples
    """
    logger.debug(f"PHASE 4-5: Generating semantic relations for {len(entity_clusters)} clusters")
    logger.debug(f"   → Generating semantic relations for {len(entity_clusters)} clusters...")
    
    all_triples = []
    
    for cluster_id, entities in entity_clusters.items():
        logger.debug(f"Processing cluster {cluster_id} with {len(entities)} entities")
        logger.debug(f"      • Cluster {cluster_id} ({len(entities)} entities)... ")
        
        # Collect cluster-specific evidence if available
        cluster_evidence = ""
        if triples_with_evidence:
            cluster_evidence = collect_cluster_evidence(entities, triples_with_evidence, max_evidence_per_entity=3)
            logger.debug(f"Collected {len(cluster_evidence)} chars of evidence for cluster {cluster_id}")
        
        triples = generate_cluster_relations(
            entities, 
            existing_triples, 
            global_summary, 
            llm_client,
            cluster_evidence=cluster_evidence
        )
        
        # Convert to tuples
        converted_count = 0
        for triple in triples:
            if isinstance(triple, dict):
                subject = triple.get("subject", "").strip()
                relation = triple.get("relation", "").strip().lower().replace(" ", "_")
                obj = triple.get("object", "").strip()
                
                if subject and relation and obj:
                    all_triples.append((subject, relation, obj))
                    converted_count += 1
                    logger.debug(f"Converted triple: ({subject}, {relation}, {obj})")
                else:
                    logger.warning(f"Skipped invalid triple: {triple}")
        
        logger.debug(f"Cluster {cluster_id} produced {converted_count} valid triples")
        logger.debug(f"✓ {len(triples)} relations")
    
    # Remove duplicates
    unique_triples = list(set(all_triples))
    duplicates_removed = len(all_triples) - len(unique_triples)
    
    logger.debug(f"Generated {len(unique_triples)} unique semantic relations ({duplicates_removed} duplicates removed)")
    logger.debug(f"   ✓ Generated {len(unique_triples)} unique semantic relations")
    
    return unique_triples


def merge_semantic_triples(
    explicit_triples: List[Tuple[str, str, str, Optional[str]]],
    semantic_triples: List[Tuple[str, str, str]]
) -> Tuple[List[Tuple[str, str, str, Optional[str]]], List[Tuple[str, str, str, Optional[str]]]]:
    """
    PHASE 6: Merge explicit and semantic triples, avoiding duplicates.
    
    Args:
        explicit_triples: Triples extracted from text with evidence (s, r, o, evidence)
        semantic_triples: Triples from semantic linking (s, r, o) - no evidence
    
    Returns:
        Tuple of (all_triples, semantic_only_triples) - both with evidence field
    """
    logger.debug(f"   → Merging {len(explicit_triples)} explicit + {len(semantic_triples)} semantic triples...")
    
    # Convert explicit triples to set for fast lookup (using only s, r, o for comparison)
    explicit_set = set((s, r, o) for s, r, o, _ in explicit_triples)
    
    # Filter semantic triples to only include new ones, add None as evidence
    semantic_only = [
        (s, r, o, None) for s, r, o in semantic_triples
        if (s, r, o) not in explicit_set
    ]
    
    # Combine
    all_triples = list(explicit_triples) + semantic_only
    
    logger.debug(f"   ✓ Final graph: {len(all_triples)} triples ({len(semantic_only)} new semantic relations)")
    
    return all_triples, semantic_only


def run_semantic_linking_pipeline(
    entities_refined: List[str],
    triples_final: List[Tuple[str, str, str, Optional[str]]],
    global_summary: str,
    llm_client: LLMClient,
    embedding_client: Optional[EmbeddingClient] = None,
    distance_threshold: float = 0.5,
    min_cluster_size: int = 2
) -> Dict[str, Any]:
    """
    Complete semantic linking pipeline to enrich knowledge graph with implicit relations.

    Args:
        entities_refined: List of canonical entities
        triples_final: List of (subject, relation, object, evidence) tuples
        global_summary: Global document summary
        llm_client: LLM client for relation generation
        embedding_client: EmbeddingClient instance for embeddings
        distance_threshold: Clustering threshold
        min_cluster_size: Minimum entities per cluster

    Returns:
        Dictionary with enriched entities and triples
    """
    logger.debug("Starting semantic linking pipeline with triple evidence-based context")
    logger.debug("\n SEMANTIC LINKING PIPELINE")
    logger.debug("=" * 70)

    # Phase 2: Generate embeddings using evidence from triples
    logger.debug("\n PHASE 2: Embedding Generation (using triple evidence)")
    logger.debug(f"Using {len(triples_final)} triples for entity context extraction")
    entity_embedding_table = generate_entity_embeddings(
        entities_refined,
        triples_final,
        embedding_client=embedding_client,
    )
    
    # Phase 3: Cluster entities
    logger.debug("\n🎯 PHASE 3: Semantic Clustering")
    entity_clusters = cluster_entities(
        entity_embedding_table,
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size
    )
    
    if not entity_clusters:
        logger.debug("   ⚠️  No clusters found - skipping semantic relation generation")
        return {
            "final_entities": entities_refined,
            "final_triples": triples_final,
            "semantic_triples": [],
            "entity_clusters": {}
        }
    
    # Phase 4-5: Generate semantic relations
    logger.debug("\n🧠 PHASE 4-5: Semantic Relation Generation")
    # Convert triples_final to simple 3-tuples for semantic relation generation
    triples_simple = [(s, r, o) for s, r, o, evidence in triples_final]
    semantic_triples = generate_semantic_relations(
        entity_clusters,
        triples_simple,
        global_summary,
        llm_client,
        triples_with_evidence=triples_final  # Pass full triples with evidence for cluster context
    )
    
    # Filter out "relates_to" relations from semantic triples
    semantic_triples_before = len(semantic_triples)
    semantic_triples = [
        (s, r, o) for s, r, o in semantic_triples 
        if r != "relates_to"
    ]
    relates_to_filtered = semantic_triples_before - len(semantic_triples)
    if relates_to_filtered > 0:
        logger.debug(f"Filtered out {relates_to_filtered} 'relates_to' triples from semantic relations")
        logger.debug(f"   ⚠️  Filtered out {relates_to_filtered} 'relates_to' relations (too generic)")
    
    # Phase 6: Merge triples (pass 4-tuples with evidence)
    logger.debug("\n🔀 PHASE 6: Merging Semantic Relations")
    final_triples, semantic_only = merge_semantic_triples(
        triples_final,  # Pass original 4-tuples with evidence
        semantic_triples
    )
    
    logger.debug("\n" + "=" * 70)
    logger.debug("✅ SEMANTIC LINKING COMPLETE")
    logger.debug("=" * 70)
    
    return {
        "final_entities": entities_refined,
        "final_triples": final_triples,
        "semantic_triples": semantic_only,
        "entity_clusters": entity_clusters,
        "entity_embeddings": entity_embedding_table
    }





