"""
KGGen Baseline Pipeline — Knowledge Graph Construction from Text

Reimplements the kg-gen pipeline (entity extraction → relation extraction → deduplication)
using the shared LLMClient for all LLM calls.  Mirrors the interface and output structure
of ``our_approach/pipeline.py`` so that both pipelines can be compared fairly.

Public entry point
------------------
    from pipeline import run_pipeline

    results = run_pipeline(text, config)
"""

import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk

# ── Shared LLM client from our_approach ──────────────────────────────────────
_OUR_APPROACH_DIR = str(Path(__file__).resolve().parent.parent.parent / "our_approach")
if _OUR_APPROACH_DIR not in sys.path:
    sys.path.insert(0, _OUR_APPROACH_DIR)

from llm_client import LLMClient  # noqa: E402

# ── Ensure NLTK sentence-tokeniser resources ────────────────────────────────
for _res_path, _res_name in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]:
    try:
        nltk.data.find(_res_path)
    except LookupError:
        nltk.download(_res_name, quiet=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Prompts — faithful to the DSPy signatures used in the default kg-gen flow
#
#  Source references (cloned repo):
#    Entity sig  → kg-gen/src/kg_gen/steps/_1_get_entities.py  class TextEntities
#    Relation sig→ kg-gen/src/kg_gen/steps/_2_get_relations.py  extraction_sig()
#    Fix sig     → kg-gen/src/kg_gen/steps/_2_get_relations.py  FixedRelations
# ═════════════════════════════════════════════════════════════════════════════

ENTITY_SYSTEM_PROMPT = (
    "Extract key entities from the source text. "
    "Extracted entities are subjects or objects.\n"
    "This is for an extraction task, please be THOROUGH and accurate "
    "to the reference text.\n\n"
    "Return a JSON object with a single key \"entities\" containing a "
    "THOROUGH list of key entities as strings."
)

RELATION_SYSTEM_PROMPT = (
    "Extract subject-predicate-object triples from the source text. "
    "Subject and object must be from the entities list. "
    "Entities provided were previously extracted from the same source text.\n"
    "This is for an extraction task, please be thorough, accurate, "
    "and faithful to the reference text.\n\n"
    "Return a JSON object with a single key \"relations\" containing a list "
    "of objects each with \"subject\", \"predicate\", and \"object\" keys."
)

RELATION_FIX_SYSTEM_PROMPT = (
    "Fix the relations so that every subject and object of the relations "
    "are exact matches to an entity in the provided entities list. "
    "Keep the predicate the same. "
    "The meaning of every relation should stay faithful to the reference text. "
    "If you cannot maintain the meaning of the original relation relative to "
    "the source text, then do not return it.\n\n"
    "Return a JSON object with a single key \"fixed_relations\" containing "
    "the corrected list of objects each with \"subject\", \"predicate\", "
    "and \"object\" keys."
)


# ═════════════════════════════════════════════════════════════════════════════
#  Text chunking  (character-based, sentence-aware — same logic as kg-gen)
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chunk_size: int = 5000) -> List[str]:
    """Split *text* into chunks that respect sentence boundaries.

    Mirrors ``kg_gen.utils.chunk_text.chunk_text`` so that results are
    comparable with the original kg-gen library.
    """
    sentences = nltk.sent_tokenize(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chunk_size:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            if len(sentence) > max_chunk_size:
                # Fallback: split by words
                words = sentence.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= max_chunk_size:
                        temp += word + " "
                    else:
                        chunks.append(temp.strip())
                        temp = word + " "
                if temp:
                    chunks.append(temp.strip())
            else:
                current = sentence + " "

    if current:
        chunks.append(current.strip())
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
#  Deduplication helpers
# ═════════════════════════════════════════════════════════════════════════════

def _deduplicate_semhash(
    entities: List[str],
    relations: List[Tuple[str, str, str]],
    threshold: float = 0.95,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    """Deduplicate entities/edges via *semhash* (from the cloned kg-gen repo).

    Falls back to basic case-insensitive deduplication when the required
    packages (``semhash``, ``inflect``) are not installed.
    """
    log = logger or logging.getLogger("KGGenPipeline")
    try:
        _kg_gen_src = str(Path(__file__).resolve().parent / "kg-gen" / "src")
        if _kg_gen_src not in sys.path:
            sys.path.insert(0, _kg_gen_src)
        from kg_gen.utils.deduplicate import run_semhash_deduplication
        from kg_gen.models import Graph

        graph = Graph(
            entities=set(entities),
            edges={r[1] for r in relations},
            relations=set(relations),
        )
        deduped = run_semhash_deduplication(graph, threshold)
        return (
            list(deduped.entities),
            list(deduped.edges),
            [tuple(r) for r in deduped.relations],
        )
    except (ImportError, Exception) as exc:
        log.warning(f"semhash deduplication unavailable ({exc}); using basic dedup")
        seen: Dict[str, str] = {}
        for e in entities:
            key = e.strip().lower()
            if key not in seen:
                seen[key] = e
        deduped_entities = list(seen.values())
        entity_set = set(deduped_entities)
        deduped_relations = list(
            {r for r in relations if r[0] in entity_set and r[2] in entity_set}
        )
        deduped_edges = list({r[1] for r in deduped_relations})
        return deduped_entities, deduped_edges, deduped_relations


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

class KGGenPipeline:
    """Reimplements the kg-gen extraction pipeline with the shared LLMClient."""

    def __init__(
        self,
        llm_client: LLMClient,
        chunk_size: int = 5000,
        context: str = "",
        deduplication: str = "semhash",
        semhash_threshold: float = 0.95,
        save_experiments: bool = True,
        experiment_name: str = "kggen_exp_1",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm_client
        self.chunk_size = chunk_size
        self.context = context
        self.deduplication = deduplication
        self.semhash_threshold = semhash_threshold
        self.save_experiments = save_experiments
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger("KGGenPipeline")

    # ── Phase 1: chunk text ─────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        chunks = chunk_text(text, self.chunk_size)
        self.logger.info(
            f"PHASE 1 | Text chunked into {len(chunks)} chunk(s) "
            f"(max {self.chunk_size} chars each)"
        )
        return chunks

    # ── Phase 2: entity extraction (per chunk) ──────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from *text*.

        Mirrors ``kg_gen.steps._1_get_entities.get_entities`` (DSPy default
        path with ``TextEntities`` signature).
        """
        user_prompt = f"source_text:\n{text}"
        response = self.llm.generate_json(ENTITY_SYSTEM_PROMPT, user_prompt)
        return response.get("entities", [])

    # ── Phase 2b: entity filtering ──────────────────────────────────────

    @staticmethod
    def _filter_entities(entities: List[str]) -> List[str]:
        """Remove entities that contain double-quote characters.

        Mirrors ``kg_gen.steps._2_get_relations._filter_entities``.
        """
        return [e for e in entities if '"' not in e]

    # ── Phase 3: relation extraction (per chunk) ────────────────────────

    def _extract_relations(
        self, text: str, entities: List[str]
    ) -> List[Tuple[str, str, str]]:
        """Extract relations, with a fixing-fallback for mismatched entities.

        Mirrors ``kg_gen.steps._2_get_relations.get_relations``:
        1. Extract relations where subject/object must come from *entities*.
        2. If any relation has a subject/object NOT in *entities*, make a
           second LLM call (``_fix_relations``) to remap them — matching
           the ``FixedRelations`` ChainOfThought step in the original code.
        3. Final filter: only keep relations whose subject AND object are
           exact members of *entities*.
        """
        entities_str = json.dumps(entities)
        user_prompt = (
            f"source_text:\n{text}\n\n"
            f"entities:\n{entities_str}"
        )
        response = self.llm.generate_json(RELATION_SYSTEM_PROMPT, user_prompt)

        raw_relations = self._parse_raw_relations(response)
        entities_set = set(entities)

        valid: List[Tuple[str, str, str]] = []
        needs_fixing: List[Tuple[str, str, str]] = []

        for s, p, o in raw_relations:
            if s in entities_set and o in entities_set:
                valid.append((s, p, o))
            else:
                needs_fixing.append((s, p, o))

        # ── Fallback: fix mismatched relations (mirrors FixedRelations) ──
        if needs_fixing:
            self.logger.debug(
                f"         {len(needs_fixing)} relation(s) have mismatched "
                f"entities — attempting fix..."
            )
            fixed = self._fix_relations(text, entities, needs_fixing)
            valid.extend(fixed)
            self.logger.debug(
                f"         Recovered {len(fixed)}/{len(needs_fixing)} "
                f"relation(s) after fixing"
            )

        return valid

    # ── Phase 3b: relation fixing fallback ──────────────────────────────

    def _fix_relations(
        self,
        text: str,
        entities: List[str],
        broken_relations: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """Ask the LLM to remap subject/object to exact entity matches.

        Mirrors ``kg_gen.steps._2_get_relations.FixedRelations``
        (the ``dspy.ChainOfThought`` fallback).
        """
        relations_payload = [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in broken_relations
        ]
        entities_str = json.dumps(entities)
        relations_str = json.dumps(relations_payload, ensure_ascii=False)

        user_prompt = (
            f"source_text:\n{text}\n\n"
            f"entities:\n{entities_str}\n\n"
            f"relations:\n{relations_str}"
        )

        try:
            response = self.llm.generate_json(
                RELATION_FIX_SYSTEM_PROMPT, user_prompt
            )
        except Exception as exc:
            self.logger.warning(f"         Relation-fix LLM call failed: {exc}")
            return []

        raw_fixed = response.get("fixed_relations", [])
        entities_set = set(entities)
        fixed: List[Tuple[str, str, str]] = []
        for rel in raw_fixed:
            s, p, o = self._unpack_relation(rel)
            if s and p and o and s in entities_set and o in entities_set:
                fixed.append((s, p, o))
        return fixed

    # ── helpers for parsing relation dicts / lists ──────────────────────

    @staticmethod
    def _unpack_relation(rel) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract (subject, predicate, object) from a dict or list."""
        if isinstance(rel, dict):
            return rel.get("subject"), rel.get("predicate"), rel.get("object")
        if isinstance(rel, (list, tuple)) and len(rel) >= 3:
            return rel[0], rel[1], rel[2]
        return None, None, None

    @staticmethod
    def _parse_raw_relations(
        response: Dict[str, Any],
    ) -> List[Tuple[str, str, str]]:
        """Parse the LLM JSON response into a list of (s, p, o) tuples."""
        raw = response.get("relations", [])
        parsed: List[Tuple[str, str, str]] = []
        for rel in raw:
            if isinstance(rel, dict):
                s = rel.get("subject")
                p = rel.get("predicate")
                o = rel.get("object")
            elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                s, p, o = rel[0], rel[1], rel[2]
            else:
                continue
            if s and p and o:
                parsed.append((s, p, o))
        return parsed

    # ── Phase 4: deduplication ──────────────────────────────────────────

    def _deduplicate(
        self,
        entities: List[str],
        relations: List[Tuple[str, str, str]],
    ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        if self.deduplication == "none":
            edges = list({r[1] for r in relations})
            return entities, edges, relations

        self.logger.info(f"PHASE 4 | Deduplicating ({self.deduplication})...")
        deduped_ents, deduped_edges, deduped_rels = _deduplicate_semhash(
            entities, relations, self.semhash_threshold, self.logger
        )
        self.logger.info(
            f"         Entities: {len(entities)} -> {len(deduped_ents)}"
        )
        self.logger.info(
            f"         Relations: {len(relations)} -> {len(deduped_rels)}"
        )
        return deduped_ents, deduped_edges, deduped_rels

    # ── Main run ────────────────────────────────────────────────────────

    def run(self, text: str) -> Dict[str, Any]:
        """Execute the full kg-gen pipeline and return structured results."""
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("No input text provided.")

        start = time.time()

        # Phase 1 — chunk
        chunks = self._chunk_text(raw_text)

        # Phase 2 & 3 — extract entities + relations per chunk
        all_entities: List[str] = []
        all_relations: List[Tuple[str, str, str]] = []
        chunk_artifacts: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, 1):
            self.logger.info(
                f"PHASE 2 | Extracting entities from chunk {idx}/{len(chunks)}..."
            )
            entities = self._extract_entities(chunk)
            self.logger.info(f"         Found {len(entities)} entities")

            # Filter entities (remove those with double-quote chars)
            # Mirrors kg_gen.steps._2_get_relations._filter_entities
            entities_before_filter = len(entities)
            entities = self._filter_entities(entities)
            if len(entities) < entities_before_filter:
                self.logger.debug(
                    f"         Filtered to {len(entities)} entities "
                    f"(removed {entities_before_filter - len(entities)} "
                    f"entries with double quotes)"
                )

            self.logger.info(
                f"PHASE 3 | Extracting relations from chunk {idx}/{len(chunks)}..."
            )
            relations = self._extract_relations(chunk, entities)
            self.logger.info(f"         Found {len(relations)} relations")

            all_entities.extend(entities)
            all_relations.extend(relations)
            chunk_artifacts.append(
                {
                    "chunk_id": idx,
                    "text": chunk,
                    "entities": entities,
                    "relations": [
                        {"subject": s, "relation": p, "object": o}
                        for s, p, o in relations
                    ],
                }
            )

        # Merge across chunks
        merged_entities = list(set(all_entities))
        merged_relations = list(set(all_relations))
        self.logger.info(
            f"         Merged: {len(merged_entities)} unique entities, "
            f"{len(merged_relations)} unique relations"
        )

        # Phase 4 — deduplication
        entities_refined, edges_refined, relations_refined = self._deduplicate(
            merged_entities, merged_relations
        )

        elapsed = time.time() - start

        # Build final triples matching our_approach dict format
        triples_final = [
            {"subject": s, "relation": p, "object": o}
            for s, p, o in relations_refined
        ]

        result: Dict[str, Any] = {
            "chunks": chunk_artifacts,
            "entities_raw": merged_entities,
            "entities_refined": entities_refined,
            "edges": edges_refined,
            "triples_final": triples_final,
            "metadata": {
                "context": self.context,
                "model": self.llm.model,
                "chunk_size": self.chunk_size,
                "num_chunks": len(chunks),
                "deduplication": self.deduplication,
                "time_taken_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            },
        }

        if self.save_experiments:
            self._save_experiment_data(result)

        return result

    # ── Experiment saving (matches our_approach directory layout) ────────

    def _save_experiment_data(self, result: Dict[str, Any]) -> None:
        exp_dir = Path(__file__).resolve().parent / "experiments" / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving experiment data to: {exp_dir}")

        # 1. Processed text (chunks)
        _write_json(
            exp_dir / "processed_text.json",
            {
                "total_chunks": len(result["chunks"]),
                "chunk_size": self.chunk_size,
                "chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "word_count": len(c["text"].split()),
                    }
                    for c in result["chunks"]
                ],
            },
        )

        # 2. Raw entities (before dedup)
        entity_freq = Counter(result["entities_raw"])
        _write_json(
            exp_dir / "local_entities.json",
            {
                "total_count": len(result["entities_raw"]),
                "unique_count": len(set(result["entities_raw"])),
                "entities": list(set(result["entities_raw"])),
                "frequency": dict(entity_freq),
            },
        )

        # 3. Refined entities (after dedup)
        _write_json(
            exp_dir / "refined_entities.json",
            {
                "total_count": len(result["entities_refined"]),
                "deduplication_method": self.deduplication,
                "entities": result["entities_refined"],
            },
        )

        # 4. Triplets
        triples = result["triples_final"]
        _write_json(
            exp_dir / "triplets.json",
            {
                "total_count": len(triples),
                "triples": triples,
                "statistics": {
                    "unique_subjects": len(set(t["subject"] for t in triples)),
                    "unique_relations": len(set(t["relation"] for t in triples)),
                    "unique_objects": len(set(t["object"] for t in triples)),
                },
            },
        )

        # 5. Complete output
        _write_json(exp_dir / "final_output.json", result)

        self.logger.info(f"   Saved 5 JSON files to {exp_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  Logging setup  (mirrors our_approach/_setup_pipeline_logging)
# ═════════════════════════════════════════════════════════════════════════════

def _setup_pipeline_logging(
    experiment_name: str,
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    """Create a logger that writes to console and/or an experiment log file."""
    logger = logging.getLogger("KGGenPipeline")
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
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
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
    # LLM
    "model": "gpt-4o-mini",
    "model_type": "openai",       # "openai", "gemini", "nim", or "local"
    "base_url": None,             # required when model_type="local"
    "temperature": 0.0,
    "max_output_tokens": 10000,
    "max_retries": 3,

    # Pipeline
    "chunk_size": 5000,           # max characters per chunk
    "context": "",                # domain description for the text
    "deduplication": "semhash",   # "semhash" or "none"
    "semhash_threshold": 0.95,

    # Experiment output
    "save_experiments": True,
    "experiment_name": "kggen_exp_1",

    # Logging
    "log_to_console": True,
    "log_to_file": True,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the KGGen knowledge-graph pipeline.

    Interface mirrors ``our_approach.pipeline.run_pipeline()``.

    Args:
        text:   Raw input text to convert into a knowledge graph.
        config: Configuration dictionary.  Any key omitted falls back to
                ``DEFAULT_CONFIG``.  Pass ``None`` or ``{}`` for all defaults.

    Returns:
        Dict with keys: chunks, entities_raw, entities_refined, edges,
        triples_final, metadata.
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

    # --- LLM client (shared with our_approach) ---
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
    pipeline = KGGenPipeline(
        llm_client=llm_client,
        chunk_size=cfg["chunk_size"],
        context=cfg["context"],
        deduplication=cfg["deduplication"],
        semhash_threshold=cfg["semhash_threshold"],
        save_experiments=cfg["save_experiments"],
        experiment_name=cfg["experiment_name"],
        logger=logger,
    )

    logger.info("=" * 50)
    logger.info("KGGEN BASELINE PIPELINE STARTED")
    logger.info(f"Model: {cfg['model']} ({cfg['model_type']})")
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
