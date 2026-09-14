"""
RAKG Baseline Pipeline — Document-level Retrieval Augmented Knowledge Graph Construction

Reimplements the RAKG pipeline using the shared LLMClient for all LLM calls.
Mirrors the interface and output structure of ``our_approach/pipeline.py``.

Pipeline phases (faithful to the original RAKG paper):
  1. Sentence segmentation
  2. Sentence vectorization (embeddings)
  3. Per-sentence entity extraction (NER)
  4. Entity similarity candidates (embedding cosine similarity)
  5. Entity disambiguation (LLM judges candidate pairs)
  6. Entity merging (union-find)
  7. Per-entity KG construction (corpus retrieval + graph retrieval + LLM)
  8. KG conversion (merge subgraphs into unified output)

Public entry point
------------------
    from pipeline import run_pipeline

    results = run_pipeline(text, config)
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Shared LLM client from our_approach ──────────────────────────────────────
_OUR_APPROACH_DIR = str(Path(__file__).resolve().parent.parent.parent / "our_approach")
if _OUR_APPROACH_DIR not in sys.path:
    sys.path.insert(0, _OUR_APPROACH_DIR)

from llm_client import LLMClient, EmbeddingClient  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
#  Prompts — faithful to RAKG/src/prompt.py
# ═════════════════════════════════════════════════════════════════════════════

NER_SYSTEM_PROMPT = (
    "You are a named entity recognition assistant responsible for "
    "identifying named entities from the given text."
)

NER_USER_PROMPT_TEMPLATE = """Text: {text}
Notes:
1. First, you should determine whether the text contains any information. If it's just meaningless symbols, directly output: {{"State": false}}. If the text contains information, proceed to the next step.
2. You should consider the entire text for named entity recognition.
3. The identified entities should consist of three parts: name, type, and description.
    - name: The main subject of the named entity.
    - type: The category of the subject.
    - description: A summary description of the subject, explaining what it is.
4. Since multiple named entities may be identified in a single text, you need to output them in a specific format.
    The output format should be:
    {{
        "entity1": {{
            "name": "Entity Name 1",
            "type": "Entity Type 1",
            "description": "Entity Description 1"
        }},
        "entity2": {{
            "name": "Entity Name 2",
            "type": "Entity Type 2",
            "description": "Entity Description 2"
        }}
    }}"""

SIMILARITY_SYSTEM_PROMPT = (
    "You are a knowledge graph entity disambiguation assistant responsible "
    "for determining whether two entities are essentially the same entity."
)

SIMILARITY_USER_PROMPT_TEMPLATE = """For example:
Entity 1: "name": "Henan Business Daily", "type": "Media Organization", "description": "A commercial newspaper in Henan Province."
Entity 2: "name": "Top News - Henan Business Daily", "type": "Organization Name", "description": "A news media organization in Henan Province."
Essentially, they are the same entity.

Entity 1: {entity1}
Entity 2: {entity2}
Notes:
1. You should initially judge whether the two entities might be the same based on their names and types, and if they might be the same, analyze their descriptions in detail to determine if they are indeed the same.
2. Your output format should be: if they are the same entity output {{"result": true}}, if not the same output {{"result": false}}."""

KG_EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction assistant, responsible for "
    "extracting attributes and relationships related to a specified entity "
    "from the text, in combination with other relevant knowledge graphs."
)

KG_EXTRACTION_USER_PROMPT_TEMPLATE = """Text: {text}
Target Entity: {target_entity}
Related Knowledge Graphs: {related_kg}
Requirements:
1. Integrate the entire text to comprehensively extract relationships related to the specified entity and build a sub-graph.
2. Extract attributes of the specified entity and relationships between the specified entity and other entities.
   - Attributes describe characteristics (e.g., "Jordan - Gender: Male").
   - For relationships, the head entity must be the specified entity. "Specified Entity - Owns - Other Entity" is valid.
3. Determine when to classify information as a relationship vs an attribute.
4. Use knowledge from related KGs to establish reverse relationships for bidirectional coverage.
5. Remove duplicate attributes and relationships.
6. Output format:
    {{
    "central_entity": {{
        "name": "",
        "type": "",
        "description": "",
        "attributes": [
            {{"key": "", "value": ""}}
        ],
        "relationships": [
            {{
                "relation": "",
                "target_name": "",
                "target_type": "",
                "target_description": "",
                "relation_description": ""
            }}
        ]
    }}
    }}"""


# ═════════════════════════════════════════════════════════════════════════════
#  Text processing
# ═════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    """Split text into sentences (mirrors RAKG/src/textPrcess.py)."""
    pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?])\s+')
    return [s.strip() for s in pattern.split(text) if s.strip()]


# ═════════════════════════════════════════════════════════════════════════════
#  JSON writer
# ═════════════════════════════════════════════════════════════════════════════

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline class
# ═════════════════════════════════════════════════════════════════════════════

class RAKGPipeline:
    """Reimplements the RAKG pipeline with the shared LLMClient."""

    def __init__(
        self,
        llm_client: LLMClient,
        similarity_llm_client: Optional[LLMClient],
        embedding_client: EmbeddingClient,
        similarity_threshold: float = 0.60,
        retrieval_top_k: int = 5,
        save_experiments: bool = True,
        experiment_name: str = "rakg_exp_1",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm_client
        self.sim_llm = similarity_llm_client or llm_client
        self.embeddings = embedding_client
        self.similarity_threshold = similarity_threshold
        self.retrieval_top_k = retrieval_top_k
        self.save_experiments = save_experiments
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger("RAKGPipeline")

    # ── Phase 1: sentence segmentation ──────────────────────────────────

    def _split_text(self, text: str) -> Dict[str, Any]:
        sentences = split_sentences(text)
        sentence_to_id = {}
        id_to_sentence = {}
        for idx, sent in enumerate(sentences):
            sent_id = f"s{idx + 1}"
            sentence_to_id[sent] = sent_id
            id_to_sentence[sent_id] = sent
        self.logger.info(f"PHASE 1 | Split text into {len(sentences)} sentences")
        return {
            "sentences": sentences,
            "sentence_to_id": sentence_to_id,
            "id_to_sentence": id_to_sentence,
        }

    # ── Phase 2: sentence vectorization ─────────────────────────────────

    def _vectorize_sentences(self, sentences: List[str]) -> List[List[float]]:
        self.logger.info(f"PHASE 2 | Vectorizing {len(sentences)} sentences...")
        vectors = self.embeddings.embed_documents(sentences)
        self.logger.info(f"         Done (dim={len(vectors[0]) if vectors else 0})")
        return vectors

    # ── Phase 3: per-sentence NER ───────────────────────────────────────

    def _extract_entities_single(self, text: str) -> Dict[str, Any]:
        """Extract entities from a single sentence via LLM."""
        user_prompt = NER_USER_PROMPT_TEMPLATE.format(text=text)
        response = self.llm.generate_json(NER_SYSTEM_PROMPT, user_prompt)
        # If the model signals no content
        if "State" in response or "state" in response:
            return {}
        return response

    def _extract_entities_all(
        self, sentences: List[str], sentence_to_id: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract and merge entities across all sentences."""
        self.logger.info(f"PHASE 3 | Extracting entities from {len(sentences)} sentences...")
        all_entities: Dict[str, Any] = {}
        entity_num = 1
        for i, sent in enumerate(sentences, 1):
            raw = self._extract_entities_single(sent)
            if not raw:
                continue
            # Renumber and add chunk_id
            for _old_key, value in raw.items():
                new_key = f"entity{entity_num}"
                value["chunkid"] = sentence_to_id.get(sent, f"s{i}")
                all_entities[new_key] = value
                entity_num += 1
            if i % 5 == 0 or i == len(sentences):
                self.logger.debug(f"         Processed {i}/{len(sentences)} sentences")
        self.logger.info(f"         Extracted {len(all_entities)} raw entities")
        return all_entities

    # ── Phase 4: similarity candidates (embedding-based) ────────────────

    def _get_similarity_candidates(
        self, entities: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Find entity pairs that are embedding-similar above threshold."""
        self.logger.info(
            f"PHASE 4 | Finding similarity candidates "
            f"(threshold={self.similarity_threshold})..."
        )
        entity_texts = {
            k: f"{v['name']} {v['type']}" for k, v in entities.items()
        }
        keys = list(entity_texts.keys())
        if len(keys) < 2:
            return []

        texts = [entity_texts[k] for k in keys]
        vectors = self.embeddings.embed_documents(texts)

        sim_matrix = np.zeros((len(keys), len(keys)))
        for i, j in combinations(range(len(keys)), 2):
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            sim_matrix[i][j] = sim

        candidates = [
            (keys[i], keys[j])
            for i, j in zip(*np.where(sim_matrix > self.similarity_threshold))
        ]
        self.logger.info(f"         Found {len(candidates)} candidate pairs")
        return candidates

    # ── Phase 5: entity disambiguation (LLM-based) ─────────────────────

    def _disambiguate_entities(
        self,
        entities: Dict[str, Any],
        candidates: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Use LLM to confirm which candidate pairs are truly the same entity."""
        self.logger.info(
            f"PHASE 5 | Disambiguating {len(candidates)} candidate pairs..."
        )
        confirmed = []
        for ent_a, ent_b in candidates:
            entity1 = entities.get(ent_a)
            entity2 = entities.get(ent_b)
            if not entity1 or not entity2:
                continue
            user_prompt = SIMILARITY_USER_PROMPT_TEMPLATE.format(
                entity1=json.dumps(entity1, ensure_ascii=False),
                entity2=json.dumps(entity2, ensure_ascii=False),
            )
            try:
                result = self.sim_llm.generate_json(
                    SIMILARITY_SYSTEM_PROMPT, user_prompt
                )
                if result.get("result", False):
                    confirmed.append((ent_a, ent_b))
            except Exception as exc:
                self.logger.warning(
                    f"         Disambiguation failed for ({ent_a}, {ent_b}): {exc}"
                )
        self.logger.info(f"         Confirmed {len(confirmed)} duplicate pairs")
        return confirmed

    # ── Phase 6: entity merging (union-find) ────────────────────────────

    @staticmethod
    def _merge_entities(
        entity_dic: Dict[str, Any],
        sim_pairs: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Merge confirmed-duplicate entities via union-find."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for e in entity_dic:
            parent[e] = e
        for a, b in sim_pairs:
            if a in entity_dic and b in entity_dic:
                union(a, b)

        groups: Dict[str, List[str]] = {}
        for e in entity_dic:
            root = find(e)
            groups.setdefault(root, []).append(e)

        for group in groups.values():
            if len(group) <= 1:
                continue
            main = group[0]
            descriptions = set()
            chunkids = set()
            for e in group:
                descriptions.add(entity_dic[e]["description"])
                chunkids.add(entity_dic[e]["chunkid"])
                if e != main:
                    del entity_dic[e]
            entity_dic[main]["description"] = ";;;".join(descriptions)
            entity_dic[main]["chunkid"] = ";;;".join(chunkids)

        return entity_dic

    # ── Phase 7: per-entity KG construction ─────────────────────────────

    def _build_entity_kg(
        self,
        entity_dic: Dict[str, Any],
        entity_id: str,
        id_to_sentence: Dict[str, str],
        sentences: List[str],
        sentence_to_id: Dict[str, str],
        vectors: List[List[float]],
    ) -> Dict[str, Any]:
        """Build an entity-centric subgraph using corpus retrieval + LLM."""
        # Get chunk sentences for this entity
        chunkids = entity_dic[entity_id].get("chunkid", "").split(";;;")
        chunk_sentences = [
            id_to_sentence[cid.strip()]
            for cid in chunkids
            if cid.strip() in id_to_sentence
        ]

        # Retrieve top-k similar sentences
        query = entity_dic[entity_id].get("name", "")
        query_vec = self.embeddings.embed_query(query)
        sentence_vecs = np.array(vectors)
        sims = cosine_similarity([query_vec], sentence_vecs)[0]
        top_indices = np.argsort(sims)[::-1][: self.retrieval_top_k]
        retrieved = [sentences[i] for i in top_indices]

        # Combine chunk + retrieved sentences (deduplicated)
        context_sentences = list(set(chunk_sentences + retrieved))
        context_text = ", ".join(context_sentences)

        # LLM extraction
        user_prompt = KG_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            text=context_text,
            target_entity=entity_dic[entity_id].get("name", ""),
            related_kg="none",
        )
        result = self.llm.generate_json(KG_EXTRACTION_SYSTEM_PROMPT, user_prompt)
        return result

    def _build_kg_all(
        self,
        entity_dic: Dict[str, Any],
        id_to_sentence: Dict[str, str],
        sentences: List[str],
        sentence_to_id: Dict[str, str],
        vectors: List[List[float]],
    ) -> Dict[str, Any]:
        """Build entity-centric subgraphs for all entities."""
        total = len(entity_dic)
        self.logger.info(f"PHASE 7 | Building KG for {total} entities...")
        results = {}
        for idx, entity_id in enumerate(entity_dic, 1):
            try:
                result = self._build_entity_kg(
                    entity_dic, entity_id, id_to_sentence,
                    sentences, sentence_to_id, vectors,
                )
                results[entity_id] = result
            except Exception as exc:
                self.logger.warning(
                    f"         Failed for {entity_id} "
                    f"({entity_dic[entity_id].get('name', '?')}): {exc}"
                )
            if idx % 5 == 0 or idx == total:
                self.logger.info(f"         Processed {idx}/{total} entities")
        return results

    # ── Phase 8: KG conversion ──────────────────────────────────────────

    @staticmethod
    def _convert_kg(raw_kg: Dict[str, Any]) -> Dict[str, Any]:
        """Merge per-entity subgraphs into unified entities + relations.

        Mirrors ``RAKG/src/kgAgent.py NER_Agent.convert_knowledge_graph``.
        """
        entity_registry: Dict[str, Dict[str, Any]] = {}
        relations: List[Dict[str, str]] = []

        for _entity_key, subgraph in raw_kg.items():
            central = subgraph.get("central_entity", {})
            name = central.get("name", "")
            if not name:
                continue

            # Register central entity
            if name not in entity_registry:
                entity_registry[name] = {
                    "name": name,
                    "type": central.get("type", ""),
                    "description": central.get("description", ""),
                }

            # Process relationships
            for rel in central.get("relationships", []):
                target_names = (
                    rel["target_name"]
                    if isinstance(rel.get("target_name"), list)
                    else [rel.get("target_name", "")]
                )
                for target_name in target_names:
                    if not target_name:
                        continue
                    # Register target entity
                    if target_name not in entity_registry:
                        entity_registry[target_name] = {
                            "name": target_name,
                            "type": rel.get("target_type", ""),
                            "description": rel.get("target_description", ""),
                        }
                    relations.append({
                        "subject": name,
                        "relation": rel.get("relation", ""),
                        "object": target_name,
                        "evidence": rel.get("relation_description", ""),
                    })

        return {
            "entities": list(entity_registry.values()),
            "relations": relations,
        }

    # ── Main run ────────────────────────────────────────────────────────

    def run(self, text: str) -> Dict[str, Any]:
        """Execute the full RAKG pipeline."""
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("No input text provided.")

        start = time.time()

        # Phase 1 — sentence segmentation
        text_data = self._split_text(raw_text)
        sentences = text_data["sentences"]
        sentence_to_id = text_data["sentence_to_id"]
        id_to_sentence = text_data["id_to_sentence"]

        # Phase 2 — sentence vectorization
        vectors = self._vectorize_sentences(sentences)

        # Phase 3 — NER
        raw_entities = self._extract_entities_all(sentences, sentence_to_id)

        # Phase 4 — similarity candidates
        candidates = self._get_similarity_candidates(raw_entities)

        # Phase 5 — entity disambiguation
        confirmed_pairs = self._disambiguate_entities(raw_entities, candidates)

        # Phase 6 — entity merging
        self.logger.info("PHASE 6 | Merging duplicate entities...")
        entities_before = len(raw_entities)
        merged_entities = self._merge_entities(raw_entities, confirmed_pairs)
        self.logger.info(
            f"         Entities: {entities_before} -> {len(merged_entities)}"
        )

        # Phase 7 — per-entity KG construction
        raw_kg = self._build_kg_all(
            merged_entities, id_to_sentence, sentences, sentence_to_id, vectors
        )

        # Phase 8 — KG conversion
        self.logger.info("PHASE 8 | Converting to unified KG format...")
        converted = self._convert_kg(raw_kg)

        elapsed = time.time() - start

        # Build output matching our_approach format
        entities_refined = [e["name"] for e in converted["entities"]]
        triples_final = converted["relations"]

        result: Dict[str, Any] = {
            "sentences": sentences,
            "entities_raw": [
                {"name": v["name"], "type": v["type"], "description": v["description"]}
                for v in merged_entities.values()
            ],
            "entities_refined": entities_refined,
            "triples_final": triples_final,
            "converted_kg": converted,
            "metadata": {
                "model": self.llm.model,
                "num_sentences": len(sentences),
                "num_raw_entities": entities_before,
                "num_merged_entities": len(merged_entities),
                "num_disambiguation_pairs": len(confirmed_pairs),
                "similarity_threshold": self.similarity_threshold,
                "retrieval_top_k": self.retrieval_top_k,
                "time_taken_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            },
        }

        if self.save_experiments:
            self._save_experiment_data(result)

        return result

    # ── Experiment saving ───────────────────────────────────────────────

    def _save_experiment_data(self, result: Dict[str, Any]) -> None:
        exp_dir = Path(__file__).resolve().parent / "experiments" / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving experiment data to: {exp_dir}")

        # 1. Processed text (sentences)
        _write_json(exp_dir / "processed_text.json", {
            "total_sentences": len(result["sentences"]),
            "sentences": [
                {"id": f"s{i+1}", "text": s, "word_count": len(s.split())}
                for i, s in enumerate(result["sentences"])
            ],
        })

        # 2. Raw entities
        _write_json(exp_dir / "local_entities.json", {
            "total_count": len(result["entities_raw"]),
            "entities": result["entities_raw"],
        })

        # 3. Refined entities
        _write_json(exp_dir / "refined_entities.json", {
            "total_count": len(result["entities_refined"]),
            "entities": result["entities_refined"],
        })

        # 4. Triplets
        triples = result["triples_final"]
        _write_json(exp_dir / "triplets.json", {
            "total_count": len(triples),
            "triples": triples,
            "statistics": {
                "unique_subjects": len(set(t["subject"] for t in triples)),
                "unique_relations": len(set(t["relation"] for t in triples)),
                "unique_objects": len(set(t["object"] for t in triples)),
            },
        })

        # 5. Complete output
        _write_json(exp_dir / "final_output.json", result)

        self.logger.info(f"   Saved 5 JSON files to {exp_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  Logging setup
# ═════════════════════════════════════════════════════════════════════════════

def _setup_pipeline_logging(
    experiment_name: str,
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("RAKGPipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    if log_file:
        log_dir = Path(__file__).resolve().parent / "experiments" / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            log_dir / f"pipeline_{timestamp}.log", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
        logger.addHandler(ch)

    return logger


# ═════════════════════════════════════════════════════════════════════════════
#  Default configuration
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict[str, Any] = {
    # Main LLM (NER + KG extraction)
    "model": "gpt-4o-mini",
    "model_type": "openai",       # "openai", "gemini", "nim", or "local"
    "base_url": None,
    "temperature": 0.0,
    "max_output_tokens": 10000,
    "max_retries": 3,

    # Similarity LLM (entity disambiguation) — defaults to main LLM if None
    "similarity_model": None,
    "similarity_model_type": None,

    # Embeddings (local sentence-transformers by default, no API needed)
    "embedding_model": "BAAI/bge-m3",       # only used when embedding_backend != "local"
    "embedding_backend": "local",            # "local" (always BAAI/bge-m3) or "openai"

    # Pipeline
    "similarity_threshold": 0.60,
    "retrieval_top_k": 5,

    # Experiment output
    "save_experiments": True,
    "experiment_name": "rakg_exp_1",

    # Logging
    "log_to_console": True,
    "log_to_file": True,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the RAKG knowledge-graph pipeline.

    Interface mirrors ``our_approach.pipeline.run_pipeline()``.
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

    # --- Main LLM client ---
    llm_client = LLMClient(
        model=cfg["model"],
        model_type=cfg["model_type"],
        base_url=cfg["base_url"],
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"],
        max_retries=cfg["max_retries"],
        logger=logger,
    )

    # --- Similarity LLM client (optional, falls back to main) ---
    sim_llm = None
    if cfg.get("similarity_model"):
        sim_llm = LLMClient(
            model=cfg["similarity_model"],
            model_type=cfg.get("similarity_model_type") or cfg["model_type"],
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_output_tokens"],
            max_retries=cfg["max_retries"],
            logger=logger,
        )

    # --- Embedding client (local BGE-M3 by default) ---
    # Local backend always uses BAAI/bge-m3; custom model only for non-local backends
    emb_backend = cfg["embedding_backend"]
    emb_model = "BAAI/bge-m3" if emb_backend == "local" else cfg["embedding_model"]
    embedding_client = EmbeddingClient(
        model=emb_model,
        backend=emb_backend,
        logger=logger,
    )

    # --- Pipeline ---
    pipeline = RAKGPipeline(
        llm_client=llm_client,
        similarity_llm_client=sim_llm,
        embedding_client=embedding_client,
        similarity_threshold=cfg["similarity_threshold"],
        retrieval_top_k=cfg["retrieval_top_k"],
        save_experiments=cfg["save_experiments"],
        experiment_name=cfg["experiment_name"],
        logger=logger,
    )

    logger.info("=" * 50)
    logger.info("RAKG BASELINE PIPELINE STARTED")
    logger.info(f"Model: {cfg['model']} ({cfg['model_type']})")
    logger.info(f"Embedding: {cfg['embedding_model']}")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Input length: {len(text)} chars")
    logger.info("=" * 50)

    results = pipeline.run(text)

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(
        f"Entities: {len(results['entities_refined'])}  |  "
        f"Triples: {len(results['triples_final'])}"
    )
    logger.info("=" * 50)

    return results
