"""
Knowledge Graph Creation Pipeline

This module implements the knowledge graph creation pipeline. It consists of 5 phases:
1. Text Preprocessing
2. Local Entity Extraction
3. Entity Refinement
4. Local Triple Extraction
5. Object Resolution
6. Semantic Linking (Optional)
7. Storing to Neo4j (Optional)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Sequence
from nltk.stem import WordNetLemmatizer

from llm_client import LLMClient, EmbeddingClient
from neo4j_writer import write_triples_to_neo4j
from prompts import (
    ENTITY_REFINEMENT_SYSTEM_PROMPT,
    LOCAL_ENTITY_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRIPLE_VERIFICATION_SYSTEM_PROMPT,
    build_entity_refinement_user_prompt,
    build_local_entity_user_prompt,
    build_summary_user_prompt,
    build_triple_extraction_user_prompt,
    build_triple_verification_user_prompt,
)
from text_processing import normalize_whitespace, split_into_chunks

from semantic_linking import run_semantic_linking_pipeline
from object_resolution import ObjectResolution

#Data classes
@dataclass
class TripleRecord:
    subject: str
    relation: str
    object: str
    evidence: Optional[str] = None
    chunk_id: Optional[int] = None

    def as_tuple(self):
        return (self.subject, self.relation, self.object)

    def as_dict(self):
        payload = {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "evidence": self.evidence,  # Always include evidence (None for semantic triples)
        }
        if self.chunk_id is not None:
            payload["chunk_id"] = self.chunk_id # type: ignore
        return payload


@dataclass
class ChunkArtifacts:
    chunk_id: int
    text: str
    entities: List[str] = field(default_factory=list)
    triples_raw: List[TripleRecord] = field(default_factory=list)
    triples_clean: List[TripleRecord] = field(default_factory=list)

    def as_dict(self):
        return {
            "id": self.chunk_id,
            "text": self.text,
            "entities": self.entities,
            "triples_raw": [triple.as_dict() for triple in self.triples_raw],
            "triples_clean": [triple.as_dict() for triple in self.triples_clean],
        }


#Helper functions
def _clean_entity(value: str, enforce_limit: bool = True) -> Optional[str]:
    """Clean and normalize an entity string."""
    text = normalize_whitespace(value)
    if not text:
        return None
    if enforce_limit:
        words = text.split()
        if len(words) > 3:
            text = " ".join(words[:3])
    return text.lower()


def _clean_relation(value: str) -> str:
    text = normalize_whitespace(value).lower()
    return text




class KGCreationPipeline:
    """
    Implements the KG creation pipeline. There are 5 phases in the pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        sentences_per_chunk: int = 3,
        summarize_text: bool = False,
        entity_refinement_mode: str = "llm",
        enable_semantic_linking: bool = False,
        embedding_model: str = "BAAI/bge-m3",
        embedding_backend: str = "local",
        cluster_distance_threshold: float = 0.9,
        min_cluster_size: int = 2,
        save_experiments: bool = True,
        experiment_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        store_to_neo4j: bool = False,
    ) -> None:
        self.llm = llm_client
        self.sentences_per_chunk = max(1, sentences_per_chunk)
        self.summarize_text = summarize_text
        self.entity_refinement_mode = entity_refinement_mode
        self.enable_semantic_linking = enable_semantic_linking
        self.embedding_backend = embedding_backend
        # Local backend always uses BAAI/bge-m3; custom model only for non-local backends
        self.embedding_model = "BAAI/bge-m3" if embedding_backend == "local" else embedding_model
        self.cluster_distance_threshold = cluster_distance_threshold
        self.min_cluster_size = min_cluster_size
        self.save_experiments = save_experiments
        self.store_to_neo4j = store_to_neo4j
        self.experiment_name = experiment_name or f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logger or logging.getLogger("KGPipeline")


    #Function to run the whole pipeline
    def run(self, text: str) -> Dict[str, Any]:
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("No input text provided.")

        self.logger.info("=" * 70)
        self.logger.info("STARTING LOCAL-GLOBAL KNOWLEDGE GRAPH PIPELINE")
        self.logger.info("=" * 70)

        # Phase 1 — Text preprocessing
        self.logger.info("PHASE 1: Text Preprocessing")
        chunk_texts = split_into_chunks(raw_text, sentences_per_chunk=self.sentences_per_chunk)
        chunks = [ChunkArtifacts(chunk_id=idx + 1, text=chunk) for idx, chunk in enumerate(chunk_texts)]
        self.logger.info(f"   Split text into {len(chunks)} chunks ({self.sentences_per_chunk} sentences/chunk)")

        # Phase 2 — Local entity extraction
        self.logger.info("PHASE 2: Local Entity Extraction (High Recall)")
        entities_raw = self._extract_local_entities(chunks)
        self.logger.info(f"   Extracted {len(entities_raw)} raw entities from all chunks")
        


        # Phase 3 - Entity refinement
        # For LLM based entity refinement, we need to summarize the text first to avoid passing too many tokens to the LLM ( in case of large text ).
        if self.summarize_text:
            self.logger.info("PHASE 3A (PREP): Summarizing the text")
            summary = self._summarize_text(raw_text)
            self.logger.info(f"   Generated summary ({len(summary)} chars)")
        else:
            self.logger.info("PHASE 3A (PREP): Skipping summarization")
            summary = raw_text
              
        
        self.logger.info("PHASE 3B: Entity Refinement")
        entities_refined = self._refine_entities(summary, entities_raw, entity_refinement_mode=self.entity_refinement_mode)
        self.logger.info(f"   Refined to {len(entities_refined)} canonical entities")

        

        # Phase 4 — Local triple extraction
        self.logger.info("PHASE 4: Local Triple Extraction")
        self._extract_chunk_triples(chunks, summary, entities_refined)
        total_raw_triples = sum(len(chunk.triples_raw) for chunk in chunks)
        self.logger.info(f"   Extracted {total_raw_triples} raw triples across all chunks")
        
        # Phase 4A — Triple verification and fixing
        self.logger.info("PHASE 4A: Triple Verification")
        verification_stats = self._verify_and_fix_triples(chunks)
        self.logger.info(f"   Verified {verification_stats['total_triples']} triples: "
                        f"{verification_stats['valid_count']} valid, "
                        f"{verification_stats['fixed_count']} fixed, "
                        f"{verification_stats['failed_count']} discarded")
        total_verified_triples = sum(len(chunk.triples_raw) for chunk in chunks)
        self.logger.info(f"   Remaining triples after verification: {total_verified_triples}")
        

        # Phase 5 - Object resolution
        self.logger.info("PHASE 5: Object Resolution")
        resolver = ObjectResolution(llm_client=self.llm, logger=self.logger)
        self.logger.debug(f"   Model for resolution: {resolver.model}")
        
        chunks, resolution_history = resolver.repair_triples(chunks=chunks, entities_refined=entities_refined)
        
        triples_final = []
        for chunk in chunks:
            triples_final.extend(chunk.triples_clean)
        self.logger.info(f"   Using {len(triples_final)} triples")

        # Semantic Linking Pipeline (Optional) - Optional step, because it adds noise to the graph.
        semantic_results = None
        if self.enable_semantic_linking:
            
            
            embedding_client = EmbeddingClient(
                model=self.embedding_model,
                backend=self.embedding_backend,
                logger=self.logger,
            )

            # Convert triples to tuples with evidence (subject, relation, object, evidence)
            triples_with_evidence = [
                (triple.subject, triple.relation, triple.object, triple.evidence)
                for triple in triples_final
            ]

            semantic_results = run_semantic_linking_pipeline(
                entities_refined=entities_refined,
                triples_final=triples_with_evidence,
                global_summary=summary,
                llm_client=self.llm,
                embedding_client=embedding_client,
                distance_threshold=self.cluster_distance_threshold,
                min_cluster_size=self.min_cluster_size
            )
            
            # Update entities and triples with semantic linking results
            entities_refined = semantic_results["final_entities"]
            # Convert back to TripleRecord objects (evidence is preserved in 4-tuples)
            final_triples_tuples = semantic_results["final_triples"]
            triples_final = [
                TripleRecord(subject=s, relation=r, object=o, evidence=evidence)
                for s, r, o, evidence in final_triples_tuples
            ]


        # Phase 6 - Storing to Neo4j
        if self.store_to_neo4j:
            self.logger.info("PHASE 7: Storing to Neo4j")
            write_triples_to_neo4j([triple.as_tuple() for triple in triples_final])
            self.logger.info(f"   Successfully stored {len(triples_final)} triples to Neo4j")

        self.logger.info("=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 70)

        result = {
            "summary": summary,
            "chunks": [chunk.as_dict() for chunk in chunks],
            "entities_raw": entities_raw,
            "entities_refined": entities_refined,
            "triples_final": [triple.as_dict() for triple in triples_final],
        }
        
        if semantic_results:
            result["semantic_triples"] = [
                {"subject": s, "relation": r, "object": o, "evidence": evidence}
                for s, r, o, evidence in semantic_results["semantic_triples"]
            ]
            result["entity_clusters"] = semantic_results["entity_clusters"]
        
        # Save experiment data to JSON files
        if self.save_experiments:
            self._save_experiment_data(
                processed_chunks=chunk_texts,
                entities_raw=entities_raw,
                entities_refined=entities_refined,
                summary=summary,
                triples_final=triples_final,
                semantic_results=semantic_results,
                resolution_history=resolution_history,
                verification_stats=verification_stats,
                final_result=result
            )
        
        return result

    #Helper functions -----------------------------------------------------

    def _extract_local_entities(self, chunks: Sequence[ChunkArtifacts]) -> List[str]:
        entities_raw: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            self.logger.debug(f"   Processing chunk {idx}/{len(chunks)}...")
            try:
                payload = self.llm.generate_json(
                    LOCAL_ENTITY_SYSTEM_PROMPT,
                    build_local_entity_user_prompt(chunk.text),
                )
                candidates = payload.get("entities", []) or []
                clean_candidates = [entity for entity in (_clean_entity(item, enforce_limit=True) for item in candidates) if entity]
                chunk.entities = clean_candidates
                entities_raw.extend(clean_candidates)
                self.logger.debug(f"   Chunk {idx}: {len(clean_candidates)} entities")
            except Exception as e:
                self.logger.error(f"   Chunk {idx} ERROR: {str(e)[:100]}")
                chunk.entities = []
                continue
        return entities_raw

    def _refine_entities(self, raw_text: str, entities_raw: Sequence[str], entity_refinement_mode: str) -> List[str]:
        self.logger.debug(f"   Refining {len(entities_raw)} raw entities...")
        try:
            lemmatizer = WordNetLemmatizer()
            normalized_words = []
            seen = set()

            for word in entities_raw:
                # Normalize: lowercase and lemmatize
                norm_word = lemmatizer.lemmatize(word.lower())
                if norm_word not in seen:
                    seen.add(norm_word)
                    normalized_words.append(word)  # append original or normalized word as needed

            entities_raw = normalized_words
            if entity_refinement_mode == "llm":
                payload = self.llm.generate_json(
                    ENTITY_REFINEMENT_SYSTEM_PROMPT,
                    build_entity_refinement_user_prompt(raw_text, entities_raw),
                )
                refined = payload.get("entities_refined", []) or []
            else:
                refined = entities_raw
            # refined = entities_raw  # TEMPORARY: skip LLM refinement
            unique: List[str] = []
            seen = set()
            for entity in refined:
                clean = _clean_entity(entity, enforce_limit=True)
                if not clean:
                    continue
                key = clean.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(clean)
            return unique
        except Exception as e:
            self.logger.error(f"ERROR: {str(e)[:100]}")
            self.logger.warning("   Falling back to raw entities (deduplication only)")
            # Fallback: deduplicate raw entities
            unique: List[str] = []
            seen = set()
            for entity in entities_raw:
                clean = _clean_entity(entity, enforce_limit=True)
                if not clean:
                    continue
                key = clean.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(clean)
            return unique

    def _summarize_text(self, raw_text: str) -> str:
        self.logger.debug("   Generating global summary...")
        try:
            payload = self.llm.generate_json(
                SUMMARY_SYSTEM_PROMPT,
                build_summary_user_prompt(raw_text),
                max_output_tokens=10000,
            )
            summary = payload.get("summary", "").strip()
            if not summary:
                summary = raw_text[:500]
            return summary
        except Exception as e:
            self.logger.error(f"ERROR: {str(e)[:100]}")
            self.logger.warning("   Falling back to text truncation")
            return raw_text[:500]

    def _extract_chunk_triples(
        self,
        chunks: Sequence[ChunkArtifacts],
        summary: str,
        refined_entities: Sequence[str],
    ) -> None:
        for idx, chunk in enumerate(chunks, 1):
            self.logger.debug(f"   Extracting triples from chunk {idx}/{len(chunks)}...")
            try:
                payload = self.llm.generate_json(
                    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
                    build_triple_extraction_user_prompt(chunk.text, summary, refined_entities),
                    max_output_tokens=10000,
                )
                chunk.triples_raw = self._parse_triples(payload, chunk_id=chunk.chunk_id, chunk_text=chunk.text)
                self.logger.debug(f"   Chunk {idx}: {len(chunk.triples_raw)} triples")
            except Exception as e:
                self.logger.error(f"   Chunk {idx} ERROR: {str(e)[:100]}")
                chunk.triples_raw = []
                continue

    def _is_triple_valid(self, triple: TripleRecord) -> bool:
        """
        Check if a triple is valid.
        A triple is INVALID if the relation contains the subject or object string.
        """
        relation_lower = triple.relation.lower()
        subject_lower = triple.subject.lower()
        object_lower = triple.object.lower()
        
        # Check if relation contains subject or object
        if subject_lower in relation_lower:
            return False
        if object_lower in relation_lower:
            return False
        
        return True

    def _verify_and_fix_triples(self, chunks: Sequence[ChunkArtifacts]) -> Dict[str, Any]:
        """
        Verify all triples in chunks and fix invalid ones using LLM.
        
        A triple is invalid if its relation contains the subject or object string.
        
        Returns:
            Dict with verification statistics
        """
        total_triples = 0
        valid_count = 0
        fixed_count = 0
        failed_count = 0
        verification_details = []
        
        for chunk in chunks:
            verified_triples = []
            
            for triple in chunk.triples_raw:
                total_triples += 1
                
                if self._is_triple_valid(triple):
                    # Triple is valid, keep as-is
                    verified_triples.append(triple)
                    valid_count += 1
                    verification_details.append({
                        "chunk_id": chunk.chunk_id,
                        "original": triple.as_dict(),
                        "status": "valid",
                        "fixed": None
                    })
                else:
                    # Triple is invalid, attempt to fix with LLM
                    self.logger.debug(
                        f"   Invalid triple found: '{triple.subject}' -> '{triple.relation}' -> '{triple.object}'"
                    )
                    
                    try:
                        payload = self.llm.generate_json(
                            TRIPLE_VERIFICATION_SYSTEM_PROMPT,
                            build_triple_verification_user_prompt(
                                subject=triple.subject,
                                relation=triple.relation,
                                object_val=triple.object,
                                evidence=triple.evidence or ""
                            ),
                            max_output_tokens=10000,
                        )
                        
                        # Parse the fixed triple
                        new_subject = _clean_entity(payload.get("subject", triple.subject), enforce_limit=True)
                        new_relation = _clean_relation(payload.get("relation", triple.relation))
                        new_object = _clean_entity(payload.get("object", triple.object), enforce_limit=False)
                        
                        if new_subject and new_relation and new_object:
                            fixed_triple = TripleRecord(
                                subject=new_subject,
                                relation=new_relation,
                                object=new_object,
                                evidence=triple.evidence,
                                chunk_id=triple.chunk_id
                            )
                            
                            # Verify the fix actually worked
                            if self._is_triple_valid(fixed_triple):
                                verified_triples.append(fixed_triple)
                                fixed_count += 1
                                self.logger.debug(
                                    f"   Fixed: '{fixed_triple.subject}' -> '{fixed_triple.relation}' -> '{fixed_triple.object}'"
                                )
                                verification_details.append({
                                    "chunk_id": chunk.chunk_id,
                                    "original": triple.as_dict(),
                                    "status": "fixed",
                                    "fixed": fixed_triple.as_dict()
                                })
                            else:
                                # LLM fix didn't work, discard triple
                                failed_count += 1
                                self.logger.warning(
                                    f"   LLM fix still invalid, discarding triple: {triple.as_dict()}"
                                )
                                verification_details.append({
                                    "chunk_id": chunk.chunk_id,
                                    "original": triple.as_dict(),
                                    "status": "discarded",
                                    "fixed": None,
                                    "reason": "LLM fix still contained subject/object in relation"
                                })
                        else:
                            failed_count += 1
                            self.logger.warning(f"   Invalid LLM response, discarding triple")
                            verification_details.append({
                                "chunk_id": chunk.chunk_id,
                                "original": triple.as_dict(),
                                "status": "discarded",
                                "fixed": None,
                                "reason": "LLM returned incomplete triple"
                            })
                            
                    except Exception as e:
                        # If LLM call fails, discard the invalid triple
                        failed_count += 1
                        self.logger.error(f"   Failed to fix triple: {str(e)[:100]}")
                        verification_details.append({
                            "chunk_id": chunk.chunk_id,
                            "original": triple.as_dict(),
                            "status": "discarded",
                            "fixed": None,
                            "reason": f"LLM error: {str(e)[:100]}"
                        })
            
            # Update chunk with verified triples
            chunk.triples_raw = verified_triples
        
        return {
            "total_triples": total_triples,
            "valid_count": valid_count,
            "fixed_count": fixed_count,
            "failed_count": failed_count,
            "details": verification_details
        }

    # Utilities ---------------------------------------------------------

    def _save_experiment_data(
        self,
        processed_chunks: List[str],
        entities_raw: List[str],
        entities_refined: List[str],
        summary: str,
        triples_final: List[TripleRecord],
        semantic_results: Optional[Dict[str, Any]] = None,
        resolution_history: Optional[List[Dict[str, Any]]] = None,
        verification_stats: Optional[Dict[str, Any]] = None,
        final_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save all pipeline data to JSON files in experiments directory.
        
        Args:
            processed_chunks: List of text chunks
            entities_raw: Raw entities with duplicates
            entities_refined: Refined canonical entities
            summary: Global summary
            triples_final: Final triples
            semantic_results: Optional semantic linking results (includes entity_embeddings)
            resolution_history: Optional list of object resolution before/after details
            verification_stats: Optional triple verification statistics and details
            final_result: Optional final result object from the pipeline run
        """
        if not self.save_experiments:
            return
        
        # Create experiments directory structure
        experiment_dir = Path(__file__).parent / "experiments" / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving experiment data to: {experiment_dir}")
        
        # 1. Save processed text (chunks)
        processed_text_data = {
            "total_chunks": len(processed_chunks),
            "sentences_per_chunk": self.sentences_per_chunk,
            "chunks": [
                {
                    "chunk_id": idx + 1,
                    "text": chunk,
                    "word_count": len(chunk.split())
                }
                for idx, chunk in enumerate(processed_chunks)
            ]
        }
        self._write_json(experiment_dir / "processed_text.json", processed_text_data)
        
        # 2. Save local entities (raw/original)
        local_entities_data = {
            "total_count": len(entities_raw),
            "unique_count": len(set(entities_raw)),
            "entity_refinement_mode": self.entity_refinement_mode,
            "entities": entities_raw,
            "frequency": {
                entity: entities_raw.count(entity)
                for entity in set(entities_raw)
            }
        }
        self._write_json(experiment_dir / "local_entities.json", local_entities_data)
        
        # 3. Save refined entities (with mode)
        refined_entities_data = {
            "total_count": len(entities_refined),
            "entity_refinement_mode": self.entity_refinement_mode,
            "entities": entities_refined
        }
        self._write_json(experiment_dir / "refined_entities.json", refined_entities_data)
        
        # 4. Save global summary
        global_summary_data = {
            "summary": summary,
            "length": len(summary),
            "word_count": len(summary.split())
        }
        self._write_json(experiment_dir / "global_summary.json", global_summary_data)
        
        # 5. Save triplets
        triplets_data = {
            "total_count": len(triples_final),
            "triples": [triple.as_dict() for triple in triples_final],
            "statistics": {
                "unique_subjects": len(set(t.subject for t in triples_final)),
                "unique_relations": len(set(t.relation for t in triples_final)),
                "unique_objects": len(set(t.object for t in triples_final)),
                "triples_with_evidence": sum(1 for t in triples_final if t.evidence)
            }
        }
        self._write_json(experiment_dir / "triplets.json", triplets_data)
        
        # 6. Save object resolution history (before/after for each triplet)
        if resolution_history:
            # Calculate statistics
            repaired_count = sum(1 for r in resolution_history if r["status"] == "repaired")
            clean_count = sum(1 for r in resolution_history if r["status"] == "clean")
            failed_count = sum(1 for r in resolution_history if r["status"] == "failed")
            
            resolution_data = {
                "total_triples_processed": len(resolution_history),
                "statistics": {
                    "clean_triples": clean_count,
                    "repaired_triples": repaired_count,
                    "failed_repairs": failed_count
                },
                "resolution_details": resolution_history
            }
            self._write_json(experiment_dir / "object_resolution.json", resolution_data)
        
        # 6b. Save triple verification results
        if verification_stats:
            verification_data = {
                "total_triples_processed": verification_stats.get("total_triples", 0),
                "statistics": {
                    "valid_triples": verification_stats.get("valid_count", 0),
                    "fixed_triples": verification_stats.get("fixed_count", 0),
                    "discarded_triples": verification_stats.get("failed_count", 0)
                },
                "verification_details": verification_stats.get("details", [])
            }
            self._write_json(experiment_dir / "triple_verification.json", verification_data)
        
        # 7. Save final result object
        if final_result:
            self._write_json(experiment_dir / "final_output.json", final_result)
        
        # 8, 9 & 10. Save semantic linking results (if available)
        if semantic_results:
            # Cluster information
            cluster_info_data = {
                "total_clusters": len(semantic_results.get("entity_clusters", {})),
                "distance_threshold": self.cluster_distance_threshold,
                "min_cluster_size": self.min_cluster_size,
                "clusters": {
                    str(cluster_id): {
                        "size": len(entities),
                        "entities": entities
                    }
                    for cluster_id, entities in semantic_results.get("entity_clusters", {}).items()
                }
            }
            self._write_json(experiment_dir / "cluster_information.json", cluster_info_data)
            
            # Semantic triples
            semantic_triples_data = {
                "total_count": len(semantic_results.get("semantic_triples", [])),
                "triples": [
                    {
                        "subject": s,
                        "relation": r,
                        "object": o,
                        "evidence": evidence
                    }
                    for s, r, o, evidence in semantic_results.get("semantic_triples", [])
                ]
            }
            self._write_json(experiment_dir / "semantic_triplets.json", semantic_triples_data)
            
            # Entity embeddings with context
            entity_embeddings_data = {
                "total_entities": len(semantic_results.get("entity_embeddings", [])),
                "embedding_model": self.embedding_model,
                "embedding_dimension": len(semantic_results.get("entity_embeddings", [{}])[0].get("entity_embedding", [])) if semantic_results.get("entity_embeddings") else 0,
                "entities": [
                    {
                        "entity": item.get("entity"),
                        "context_text": item.get("context_text"),
                        "context_length": len(item.get("context_text", "")),
                        # "entity_embedding": item.get("entity_embedding"),
                        # "context_embedding": item.get("context_embedding")
                    }
                    for item in semantic_results.get("entity_embeddings", [])
                ]
            }
            self._write_json(experiment_dir / "entity_embeddings.json", entity_embeddings_data)
        
        # Calculate number of files saved
        files_saved = 7  # Base: processed_text, local_entities, refined_entities, global_summary, triplets, object_resolution, final_output
        if verification_stats:
            files_saved += 1  # triple_verification
        if semantic_results:
            files_saved += 3  # cluster_information, semantic_triplets, entity_embeddings
        self.logger.info(f"   Saved {files_saved} JSON files to {experiment_dir}")
    
    def _write_json(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write data to JSON file with pretty formatting."""
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, dict):
                return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        converted_data = convert_numpy_types(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

    def _parse_triples(self, payload: Any, chunk_id: Optional[int], chunk_text: Optional[str]) -> List[TripleRecord]:
        if isinstance(payload, dict):
            triples_raw = payload.get("triples", []) or []
        elif isinstance(payload, list):
            triples_raw = payload
        else:
            triples_raw = []
        triples: List[TripleRecord] = []
        for triple in triples_raw:
            if isinstance(triple, dict):
                # Subject: enforce 3-word limit (must match entity list)
                subject = _clean_entity(triple.get("subject", ""), enforce_limit=True)
                relation = _clean_relation(triple.get("relation", ""))
                # Object: NO word limit (can be longer descriptive phrases)
                obj = _clean_entity(triple.get("object", ""), enforce_limit=False)
                if chunk_text:
                    evidence = normalize_whitespace(chunk_text) or None
                else:
                    evidence = None
            else:
                # Fallback for tuples or lists
                try:
                    subject, relation, obj = triple[:3]
                except Exception:
                    continue
                # Subject: enforce 3-word limit (must match entity list)
                subject = _clean_entity(subject, enforce_limit=True)
                relation = _clean_relation(relation)
                # Object: NO word limit (can be longer descriptive phrases)
                obj = _clean_entity(obj, enforce_limit=False)
                evidence = None

            if not (subject and relation and obj):
                continue
            triples.append(
                TripleRecord(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    evidence=evidence,
                    chunk_id=chunk_id,
                )
            )
        return triples


# ---------------------------------------------------------------------------
# Single-function entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # LLM
    "model": "gpt-4o-mini",
    "model_type": "openai",       # "openai", "gemini", "nim", or "local"
    "base_url": None,             # required when model_type="local"
    "temperature": 0.2,
    "max_output_tokens": 10000,
    "max_retries": 3,

    # Text chunking
    "sentences_per_chunk": 4,

    # Summarization
    "summarize_text": True,

    # Entity refinement
    "entity_refinement_mode": "deterministic",  # "deterministic" or "llm"

    # Semantic linking (optional enrichment)
    "enable_semantic_linking": False,
    "embedding_model": "BAAI/bge-m3",       # only used when embedding_backend != "local"
    "embedding_backend": "local",            # "local" (always BAAI/bge-m3) or "openai"
    "cluster_distance_threshold": 0.9,
    "min_cluster_size": 2,

    # Storage
    "store_to_neo4j": False,

    # Experiment output
    "save_experiments": True,
    "experiment_name": "exp_1",

    # Logging
    "log_to_console": True,
    "log_to_file": True,
}


def _setup_pipeline_logging(
    experiment_name: str,
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    """Create a logger that writes to console and/or an experiment log file."""
    logger = logging.getLogger("KGPipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    if log_file:
        log_dir = Path(__file__).parent / "experiments" / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"pipeline_{timestamp}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    if console:
        import sys
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
        logger.addHandler(ch)

    return logger


def run_pipeline(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the full text-to-knowledge-graph pipeline with a single config dict.

    Args:
        text:   Input text to convert into a knowledge graph.
        config: Configuration dictionary. Any key omitted falls back to
                DEFAULT_CONFIG.  Pass ``None`` or ``{}`` to use all defaults.

    Returns:
        Pipeline results dictionary containing:
            - summary, chunks, entities_raw, entities_refined, triples_final
            - semantic_triples & entity_clusters  (if semantic linking enabled)
    """
    from dotenv import load_dotenv
    load_dotenv()

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # --- Logging ---
    logger = _setup_pipeline_logging(
        experiment_name=cfg["experiment_name"],
        console=cfg["log_to_console"],
        log_file=cfg["log_to_file"],
    )

    # --- LLM client ---
    llm_client = LLMClient(
        model=cfg["model"],
        model_type=cfg["model_type"],
        base_url=cfg["base_url"],
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"],
        max_retries=cfg["max_retries"],
        logger=logger,
    )

    # --- Pipeline ---
    pipeline = KGCreationPipeline(
        llm_client=llm_client,
        sentences_per_chunk=cfg["sentences_per_chunk"],
        summarize_text=cfg["summarize_text"],
        entity_refinement_mode=cfg["entity_refinement_mode"],
        enable_semantic_linking=cfg["enable_semantic_linking"],
        embedding_model=cfg["embedding_model"],
        embedding_backend=cfg["embedding_backend"],
        cluster_distance_threshold=cfg["cluster_distance_threshold"],
        min_cluster_size=cfg["min_cluster_size"],
        save_experiments=cfg["save_experiments"],
        experiment_name=cfg["experiment_name"],
        store_to_neo4j=cfg["store_to_neo4j"],
        logger=logger,
    )

    logger.info("=" * 50)
    logger.info("KNOWLEDGE GRAPH PIPELINE STARTED")
    logger.info(f"Model: {cfg['model']} ({cfg['model_type']})")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Input length: {len(text)} chars")
    logger.info("=" * 50)

    results = pipeline.run(text)

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Entities: {len(results['entities_refined'])}  |  "
                f"Triples: {len(results['triples_final'])}")
    logger.info("=" * 50)

    return results
