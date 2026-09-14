"""
Batched Knowledge Graph Creation Pipeline

Orchestrates the sequential pipeline (our_approach/pipeline.py) across many
documents by replacing the LLM + embedding calls with OpenAI Batch API
submissions. Each "phase" that contains independent calls becomes ONE batch:

    Batch 1:  entity extraction (per chunk) + summarization (per doc)
    Batch 2:  entity refinement (per doc)
    Batch 3:  triple extraction (per chunk)
    Batch 4:  triple verification (only for invalid triples)
    Sequential:  object resolution (re-uses ObjectResolution unchanged)
    Batch 5:  entity + context embeddings (text-embedding-3-large)
    Sequential:  semantic linking clustering + cluster-relation LLM calls

Failures from any batch (JSON parse errors, API errors, batch_expired) are
retried synchronously via the existing LLMClient.generate_json(), preserving
the same fallback behavior as the sequential pipeline.

Output: one folder per document under
    experiments/<experiment_name>/<doc_id>/
with the exact same JSON files + log format as pipeline.KGCreationPipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Sequential-pipeline imports — used READ-ONLY, no modifications
_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent
if str(_OUR_APPROACH_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH_DIR))

from llm_client import LLMClient, EmbeddingClient  # noqa: E402
from pipeline import (  # noqa: E402
    ChunkArtifacts,
    KGCreationPipeline,
    TripleRecord,
    _clean_entity,
    _clean_relation,
)
from prompts import (  # noqa: E402
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
from text_processing import split_into_chunks  # noqa: E402
from object_resolution import ObjectResolution  # noqa: E402
from semantic_linking import run_semantic_linking_pipeline  # noqa: E402
import semantic_linking as _semantic_linking_module  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402


# Monkey-patch: semantic_linking.py:547 calls logger.debug(..., end="") which
# Python's logging module doesn't accept. We don't modify the sequential file,
# so we wrap its module logger to drop the stray kwarg.
def _install_semantic_linking_logger_patch() -> None:
    _orig_debug = _semantic_linking_module.logger.debug

    def _safe_debug(msg, *args, **kwargs):
        kwargs.pop("end", None)
        return _orig_debug(msg, *args, **kwargs)

    _semantic_linking_module.logger.debug = _safe_debug


_install_semantic_linking_logger_patch()

from batch.batch_client import (
    BatchClient,
    BatchRequest,
    BatchResultEntry,
    build_embeddings_request,
    build_responses_request,
)


# ---------------------------------------------------------------------------
# Per-document state (in memory during the run)
# ---------------------------------------------------------------------------
@dataclass
class DocState:
    doc_id: str
    text: str
    chunks: List[ChunkArtifacts] = field(default_factory=list)
    summary: str = ""
    entities_raw: List[str] = field(default_factory=list)
    entities_refined: List[str] = field(default_factory=list)
    triples_final: List[TripleRecord] = field(default_factory=list)
    resolution_history: List[Dict[str, Any]] = field(default_factory=list)
    verification_stats: Dict[str, Any] = field(default_factory=dict)
    semantic_results: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Cached embedding client — lookup-only, fulfills EmbeddingClient contract
# so we can feed precomputed batch embeddings into run_semantic_linking_pipeline
# without modifying any sequential code.
# ---------------------------------------------------------------------------
class CachedEmbeddingClient:
    """Drop-in replacement for EmbeddingClient that serves precomputed vectors."""

    def __init__(self, cache: Dict[str, List[float]], logger: logging.Logger):
        self._cache = cache
        self.logger = logger

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = self._cache.get(t)
            if vec is None:
                raise KeyError(
                    f"CachedEmbeddingClient miss for text (first 80 chars): {t[:80]!r}"
                )
            out.append(vec)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class BatchKGPipeline:
    """
    Runs the full KG pipeline across many documents using batched OpenAI calls
    where possible. Output format per document matches the sequential pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.experiment_name: str = config["experiment_name"]
        self.experiment_dir: Path = (
            _OUR_APPROACH_DIR / "experiments" / self.experiment_name
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._build_logger()

        # Sync LLM client — used for retrying batch failures and for sequential phases
        self.llm = LLMClient(
            model=config["model"],
            model_type=config["model_type"],
            base_url=config.get("base_url"),
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            max_retries=config["max_retries"],
            logger=self.logger,
        )

        # Batch client — writes its state under the experiment dir
        self.batch_workdir = self.experiment_dir / "batch_workdir"
        self.batch = BatchClient(
            workdir=self.batch_workdir,
            poll_interval_seconds=60,
            logger=self.logger,
        )

    # -------------------------------------------------------------------
    # Logging — same format as sequential pipeline._setup_pipeline_logging
    # -------------------------------------------------------------------
    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("KGPipeline")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        if self.config.get("log_to_file", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(
                self.experiment_dir / f"pipeline_{timestamp}.log", encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

        if self.config.get("log_to_console", True):
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
            logger.addHandler(ch)

        return logger

    # -------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------
    def run(self, docs: Dict[str, str], dry_run: bool = False) -> Dict[str, DocState]:
        """
        Run the full batched pipeline over many documents.

        Args:
            docs: {doc_id: text} mapping
            dry_run: if True, build input .jsonl files and report plan only;
                     do NOT upload/submit to OpenAI. Useful for cost validation.

        Returns:
            {doc_id: DocState} with all populated artifacts
        """
        states: Dict[str, DocState] = {
            doc_id: DocState(doc_id=doc_id, text=text) for doc_id, text in docs.items()
        }

        self.logger.info("=" * 70)
        self.logger.info(
            f"BATCH KG PIPELINE — {len(states)} docs — experiment={self.experiment_name}"
        )
        self.logger.info(f"Model: {self.config['model']}")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info("=" * 70)

        # Phase 1 (local): chunking
        self._phase1_preprocess(states)

        # Batch 1: entity extraction (per chunk) + summarization (per doc)
        self._batch_entity_and_summary(states, dry_run=dry_run)
        if dry_run:
            self._report_dry_run_counts(states)
            return states

        # Batch 2: entity refinement (per doc)
        self._batch_entity_refinement(states)

        # Batch 3: triple extraction (per chunk)
        self._batch_triple_extraction(states)

        # Batch 4: triple verification (only for invalid triples)
        self._batch_triple_verification(states)

        # Sequential: object resolution (per doc)
        self._sequential_object_resolution(states)

        # Optional: semantic linking via batched embeddings
        if self.config.get("enable_semantic_linking", False):
            self._semantic_linking(states)

        # Save per-doc outputs in the sequential format
        self._save_all_experiments(states)

        self.logger.info("=" * 70)
        self.logger.info("BATCH PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        return states

    # -------------------------------------------------------------------
    # Phase 1: text preprocessing (local, no API)
    # -------------------------------------------------------------------
    def _phase1_preprocess(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 1: Text Preprocessing")
        spc = self.config["sentences_per_chunk"]
        total_chunks = 0
        for st in states.values():
            chunk_texts = split_into_chunks(st.text.strip(), sentences_per_chunk=spc)
            st.chunks = [
                ChunkArtifacts(chunk_id=i + 1, text=ct)
                for i, ct in enumerate(chunk_texts)
            ]
            total_chunks += len(st.chunks)
            self.logger.debug(f"   {st.doc_id}: {len(st.chunks)} chunks")
        self.logger.info(f"   Total chunks across docs: {total_chunks}")

    # -------------------------------------------------------------------
    # Batch 1: entity extraction + summarization
    # -------------------------------------------------------------------
    def _batch_entity_and_summary(
        self, states: Dict[str, DocState], dry_run: bool
    ) -> None:
        self.logger.info("BATCH 1: Local Entity Extraction + Summarization")
        requests: List[BatchRequest] = []

        for st in states.values():
            # One entity-extraction request per chunk
            for chunk in st.chunks:
                requests.append(
                    build_responses_request(
                        custom_id=f"entity|{st.doc_id}|{chunk.chunk_id}",
                        model=self.config["model"],
                        system_prompt=LOCAL_ENTITY_SYSTEM_PROMPT,
                        user_prompt=build_local_entity_user_prompt(chunk.text),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )
            # One summary request per doc (if summarization enabled)
            if self.config.get("summarize_text", True):
                requests.append(
                    build_responses_request(
                        custom_id=f"summary|{st.doc_id}",
                        model=self.config["model"],
                        system_prompt=SUMMARY_SYSTEM_PROMPT,
                        user_prompt=build_summary_user_prompt(st.text),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )

        self.logger.info(f"   Total requests in batch 1: {len(requests)}")
        if dry_run:
            self.batch.write_input_file("batch1_entity_summary", requests)
            return

        results = self.batch.run_job(
            job_name="batch1_entity_summary",
            endpoint="/v1/responses",
            requests=requests,
        )

        # Distribute results back to DocStates (with sync retry on failure)
        for custom_id, payload in self._iter_parsed_json_results(
            results,
            requests,
            retry_phase_label="batch1",
        ):
            if custom_id.startswith("entity|"):
                _, doc_id, chunk_id_str = custom_id.split("|", 2)
                chunk_id = int(chunk_id_str)
                chunk = next(c for c in states[doc_id].chunks if c.chunk_id == chunk_id)
                candidates = payload.get("entities", []) or []
                clean_candidates = [
                    e
                    for e in (
                        _clean_entity(item, enforce_limit=True) for item in candidates
                    )
                    if e
                ]
                chunk.entities = clean_candidates
                states[doc_id].entities_raw.extend(clean_candidates)
            elif custom_id.startswith("summary|"):
                _, doc_id = custom_id.split("|", 1)
                summary = (payload.get("summary") or "").strip()
                if not summary:
                    summary = states[doc_id].text[:500]
                states[doc_id].summary = summary

        # If summarization was disabled, fall back to raw text per sequential behavior
        if not self.config.get("summarize_text", True):
            for st in states.values():
                st.summary = st.text

        for st in states.values():
            self.logger.info(
                f"   {st.doc_id}: {len(st.entities_raw)} raw entities, "
                f"summary={len(st.summary)} chars"
            )

    # -------------------------------------------------------------------
    # Batch 2: entity refinement
    # -------------------------------------------------------------------
    def _batch_entity_refinement(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH 2: Entity Refinement")
        mode = self.config.get("entity_refinement_mode", "llm")

        # Always do the deterministic lemmatization dedup first (matches sequential)
        lemmatizer = WordNetLemmatizer()
        for st in states.values():
            seen = set()
            normalized = []
            for w in st.entities_raw:
                k = lemmatizer.lemmatize(w.lower())
                if k not in seen:
                    seen.add(k)
                    normalized.append(w)
            st.entities_raw = normalized

        if mode != "llm":
            # Deterministic mode: no batch needed
            for st in states.values():
                st.entities_refined = self._dedupe_entities(st.entities_raw)
            return

        # LLM mode: one request per doc
        requests: List[BatchRequest] = []
        for st in states.values():
            requests.append(
                build_responses_request(
                    custom_id=f"refine|{st.doc_id}",
                    model=self.config["model"],
                    system_prompt=ENTITY_REFINEMENT_SYSTEM_PROMPT,
                    user_prompt=build_entity_refinement_user_prompt(
                        st.summary, st.entities_raw
                    ),
                    temperature=self.config["temperature"],
                    max_output_tokens=self.config["max_output_tokens"],
                )
            )

        self.logger.info(f"   Total requests in batch 2: {len(requests)}")
        results = self.batch.run_job(
            job_name="batch2_refine",
            endpoint="/v1/responses",
            requests=requests,
        )

        for custom_id, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch2"
        ):
            _, doc_id = custom_id.split("|", 1)
            refined = payload.get("entities_refined", []) or []
            states[doc_id].entities_refined = self._dedupe_entities(refined)

        # Any doc that came back empty → fallback to raw entities
        for st in states.values():
            if not st.entities_refined:
                self.logger.warning(
                    f"   {st.doc_id}: refinement empty, falling back to raw entities"
                )
                st.entities_refined = self._dedupe_entities(st.entities_raw)
            self.logger.info(
                f"   {st.doc_id}: {len(st.entities_refined)} refined entities"
            )

    @staticmethod
    def _dedupe_entities(entities: Sequence[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for e in entities:
            c = _clean_entity(e, enforce_limit=True)
            if not c:
                continue
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            unique.append(c)
        return unique

    # -------------------------------------------------------------------
    # Batch 3: triple extraction
    # -------------------------------------------------------------------
    def _batch_triple_extraction(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH 3: Triple Extraction")
        requests: List[BatchRequest] = []
        for st in states.values():
            for chunk in st.chunks:
                requests.append(
                    build_responses_request(
                        custom_id=f"triple|{st.doc_id}|{chunk.chunk_id}",
                        model=self.config["model"],
                        system_prompt=TRIPLE_EXTRACTION_SYSTEM_PROMPT,
                        user_prompt=build_triple_extraction_user_prompt(
                            chunk.text, st.summary, st.entities_refined
                        ),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )

        self.logger.info(f"   Total requests in batch 3: {len(requests)}")
        results = self.batch.run_job(
            job_name="batch3_triples",
            endpoint="/v1/responses",
            requests=requests,
        )

        for custom_id, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch3"
        ):
            _, doc_id, chunk_id_str = custom_id.split("|", 2)
            chunk_id = int(chunk_id_str)
            chunk = next(c for c in states[doc_id].chunks if c.chunk_id == chunk_id)
            chunk.triples_raw = self._parse_triples_payload(
                payload, chunk_id=chunk.chunk_id, chunk_text=chunk.text
            )

        for st in states.values():
            total = sum(len(c.triples_raw) for c in st.chunks)
            self.logger.info(f"   {st.doc_id}: {total} raw triples")

    @staticmethod
    def _parse_triples_payload(
        payload: Any, chunk_id: int, chunk_text: str
    ) -> List[TripleRecord]:
        from text_processing import normalize_whitespace

        if isinstance(payload, dict):
            raw = payload.get("triples", []) or []
        elif isinstance(payload, list):
            raw = payload
        else:
            return []

        out: List[TripleRecord] = []
        for t in raw:
            if isinstance(t, dict):
                subject = _clean_entity(t.get("subject", ""), enforce_limit=True)
                relation = _clean_relation(t.get("relation", ""))
                obj = _clean_entity(t.get("object", ""), enforce_limit=False)
                evidence = normalize_whitespace(chunk_text) or None
            else:
                try:
                    subject, relation, obj = t[:3]
                except Exception:
                    continue
                subject = _clean_entity(subject, enforce_limit=True)
                relation = _clean_relation(relation)
                obj = _clean_entity(obj, enforce_limit=False)
                evidence = None
            if not (subject and relation and obj):
                continue
            out.append(
                TripleRecord(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    evidence=evidence,
                    chunk_id=chunk_id,
                )
            )
        return out

    # -------------------------------------------------------------------
    # Batch 4: triple verification (only on invalid triples)
    # -------------------------------------------------------------------
    def _batch_triple_verification(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH 4: Triple Verification")

        # Identify invalid triples across all docs using sequential's rule
        invalids: List[Tuple[str, int, int, TripleRecord]] = []  # (doc_id, chunk_id, triple_idx, triple)
        stats_per_doc: Dict[str, Dict[str, Any]] = {}

        for st in states.values():
            total = 0
            valid = 0
            details: List[Dict[str, Any]] = []
            for chunk in st.chunks:
                for idx, tr in enumerate(chunk.triples_raw):
                    total += 1
                    if self._is_triple_valid(tr):
                        valid += 1
                        details.append(
                            {
                                "chunk_id": chunk.chunk_id,
                                "original": tr.as_dict(),
                                "status": "valid",
                                "fixed": None,
                            }
                        )
                    else:
                        invalids.append((st.doc_id, chunk.chunk_id, idx, tr))
            stats_per_doc[st.doc_id] = {
                "total_triples": total,
                "valid_count": valid,
                "fixed_count": 0,
                "failed_count": 0,
                "details": details,
            }

        self.logger.info(
            f"   Identified {len(invalids)} invalid triples across {len(states)} docs"
        )

        if not invalids:
            for doc_id, stats in stats_per_doc.items():
                states[doc_id].verification_stats = stats
            return

        # Submit verification batch
        requests: List[BatchRequest] = []
        for doc_id, chunk_id, triple_idx, tr in invalids:
            requests.append(
                build_responses_request(
                    custom_id=f"verify|{doc_id}|{chunk_id}|{triple_idx}",
                    model=self.config["model"],
                    system_prompt=TRIPLE_VERIFICATION_SYSTEM_PROMPT,
                    user_prompt=build_triple_verification_user_prompt(
                        subject=tr.subject,
                        relation=tr.relation,
                        object_val=tr.object,
                        evidence=tr.evidence or "",
                    ),
                    temperature=self.config["temperature"],
                    max_output_tokens=10000,
                )
            )

        results = self.batch.run_job(
            job_name="batch4_verify",
            endpoint="/v1/responses",
            requests=requests,
        )

        # Build a lookup of parsed payloads (with sync retry on failure)
        payload_by_id: Dict[str, Any] = {}
        for cid, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch4"
        ):
            payload_by_id[cid] = payload

        # Apply fixes: replace invalid triples with fixed ones, or discard
        # Rebuild chunk.triples_raw with only (valid + fixed) triples
        keep_per_chunk: Dict[Tuple[str, int], List[TripleRecord]] = {}
        for st in states.values():
            for chunk in st.chunks:
                keep_per_chunk[(st.doc_id, chunk.chunk_id)] = [
                    tr for tr in chunk.triples_raw if self._is_triple_valid(tr)
                ]

        for doc_id, chunk_id, triple_idx, tr in invalids:
            cid = f"verify|{doc_id}|{chunk_id}|{triple_idx}"
            payload = payload_by_id.get(cid)
            stats = stats_per_doc[doc_id]
            if payload is None:
                stats["failed_count"] += 1
                stats["details"].append(
                    {
                        "chunk_id": chunk_id,
                        "original": tr.as_dict(),
                        "status": "discarded",
                        "fixed": None,
                        "reason": "LLM error (batch + sync retry both failed)",
                    }
                )
                continue

            new_subject = _clean_entity(
                payload.get("subject", tr.subject), enforce_limit=True
            )
            new_relation = _clean_relation(payload.get("relation", tr.relation))
            new_object = _clean_entity(
                payload.get("object", tr.object), enforce_limit=False
            )
            if new_subject and new_relation and new_object:
                fixed = TripleRecord(
                    subject=new_subject,
                    relation=new_relation,
                    object=new_object,
                    evidence=tr.evidence,
                    chunk_id=tr.chunk_id,
                )
                if self._is_triple_valid(fixed):
                    keep_per_chunk[(doc_id, chunk_id)].append(fixed)
                    stats["fixed_count"] += 1
                    stats["details"].append(
                        {
                            "chunk_id": chunk_id,
                            "original": tr.as_dict(),
                            "status": "fixed",
                            "fixed": fixed.as_dict(),
                        }
                    )
                else:
                    stats["failed_count"] += 1
                    stats["details"].append(
                        {
                            "chunk_id": chunk_id,
                            "original": tr.as_dict(),
                            "status": "discarded",
                            "fixed": None,
                            "reason": "LLM fix still contained subject/object in relation",
                        }
                    )
            else:
                stats["failed_count"] += 1
                stats["details"].append(
                    {
                        "chunk_id": chunk_id,
                        "original": tr.as_dict(),
                        "status": "discarded",
                        "fixed": None,
                        "reason": "LLM returned incomplete triple",
                    }
                )

        # Write back cleaned triples_raw and save stats
        for st in states.values():
            for chunk in st.chunks:
                chunk.triples_raw = keep_per_chunk[(st.doc_id, chunk.chunk_id)]
            st.verification_stats = stats_per_doc[st.doc_id]
            s = st.verification_stats
            self.logger.info(
                f"   {st.doc_id}: total={s['total_triples']} valid={s['valid_count']} "
                f"fixed={s['fixed_count']} discarded={s['failed_count']}"
            )

    @staticmethod
    def _is_triple_valid(triple: TripleRecord) -> bool:
        r = triple.relation.lower()
        if triple.subject.lower() in r:
            return False
        if triple.object.lower() in r:
            return False
        return True

    # -------------------------------------------------------------------
    # Sequential: object resolution (reuses ObjectResolution unchanged)
    # -------------------------------------------------------------------
    def _sequential_object_resolution(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 5 (sequential): Object Resolution")
        resolver = ObjectResolution(llm_client=self.llm, logger=self.logger)
        for st in states.values():
            self.logger.info(f"   Resolving {st.doc_id}...")
            st.chunks, st.resolution_history = resolver.repair_triples(
                chunks=st.chunks, entities_refined=st.entities_refined
            )
            triples_final: List[TripleRecord] = []
            for chunk in st.chunks:
                triples_final.extend(chunk.triples_clean)
            st.triples_final = triples_final
            self.logger.info(f"   {st.doc_id}: {len(triples_final)} final triples")

    # -------------------------------------------------------------------
    # Semantic linking: batch all embeddings, then sequential clustering + LLM
    # -------------------------------------------------------------------
    def _semantic_linking(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 6 (batched embeddings): Semantic Linking")
        embed_model = self.config.get("embedding_model", "text-embedding-3-large")
        backend = self.config.get("embedding_backend", "openai")
        if backend != "openai":
            self.logger.warning(
                f"   Semantic linking with backend={backend} runs sequentially; "
                f"use backend='openai' to benefit from batch embeddings."
            )
            self._semantic_linking_sequential(states)
            return

        # Step 1: pre-compute the texts that run_semantic_linking_pipeline will
        # embed for each doc — same logic as semantic_linking.generate_entity_embeddings
        from semantic_linking import find_entity_context_from_triples

        texts_to_embed: List[str] = []
        seen_texts: set = set()
        per_doc_plan: Dict[str, Tuple[List[str], List[str]]] = {}  # doc_id -> (entities, contexts)
        for st in states.values():
            triples_with_ev = [
                (t.subject, t.relation, t.object, t.evidence) for t in st.triples_final
            ]
            entity_texts = list(st.entities_refined)
            context_texts: List[str] = []
            for entity in entity_texts:
                contexts = find_entity_context_from_triples(
                    entity, triples_with_ev, max_contexts=5
                )
                combined = " ".join(contexts[:3])
                context_texts.append(combined)
            per_doc_plan[st.doc_id] = (entity_texts, context_texts)
            for t in entity_texts + context_texts:
                if t not in seen_texts:
                    seen_texts.add(t)
                    texts_to_embed.append(t)

        self.logger.info(
            f"   Embedding {len(texts_to_embed)} unique texts across {len(states)} docs"
        )

        # Step 2: one batch request per unique text
        requests: List[BatchRequest] = []
        index_to_text: Dict[int, str] = {}
        for i, text in enumerate(texts_to_embed):
            requests.append(
                build_embeddings_request(
                    custom_id=f"embed|{i}",
                    model=embed_model,
                    text=text,
                )
            )
            index_to_text[i] = text

        embedding_cache: Dict[str, List[float]] = {}
        if requests:
            results = self.batch.run_job(
                job_name="batch5_embeddings",
                endpoint="/v1/embeddings",
                requests=requests,
            )
            for cid, req in zip((r.custom_id for r in requests), requests):
                entry = results.get(cid)
                text = index_to_text[int(cid.split("|", 1)[1])]
                if entry is None or not entry.success or entry.embedding is None:
                    self.logger.warning(
                        f"   Embedding failed for cid={cid}; retrying sync..."
                    )
                    # Fallback: sync embedding via EmbeddingClient (openai backend)
                    fallback = EmbeddingClient(
                        model=embed_model, backend="openai", logger=self.logger
                    )
                    embedding_cache[text] = fallback.embed_query(text)
                else:
                    embedding_cache[text] = entry.embedding

        cached_client = CachedEmbeddingClient(cache=embedding_cache, logger=self.logger)

        # Step 3: per-doc clustering + cluster-relation LLM calls (sequential)
        for st in states.values():
            triples_with_ev = [
                (t.subject, t.relation, t.object, t.evidence) for t in st.triples_final
            ]
            self.logger.info(f"   Semantic linking for {st.doc_id}...")
            st.semantic_results = run_semantic_linking_pipeline(
                entities_refined=st.entities_refined,
                triples_final=triples_with_ev,
                global_summary=st.summary,
                llm_client=self.llm,
                embedding_client=cached_client,
                distance_threshold=self.config.get("cluster_distance_threshold", 0.9),
                min_cluster_size=self.config.get("min_cluster_size", 2),
            )
            # Update triples_final with semantic-linking output
            final_triples_tuples = st.semantic_results["final_triples"]
            st.triples_final = [
                TripleRecord(subject=s, relation=r, object=o, evidence=ev)
                for s, r, o, ev in final_triples_tuples
            ]

    def _semantic_linking_sequential(self, states: Dict[str, DocState]) -> None:
        embedding_client = EmbeddingClient(
            model=self.config.get("embedding_model", "BAAI/bge-m3"),
            backend=self.config.get("embedding_backend", "local"),
            logger=self.logger,
        )
        for st in states.values():
            triples_with_ev = [
                (t.subject, t.relation, t.object, t.evidence) for t in st.triples_final
            ]
            st.semantic_results = run_semantic_linking_pipeline(
                entities_refined=st.entities_refined,
                triples_final=triples_with_ev,
                global_summary=st.summary,
                llm_client=self.llm,
                embedding_client=embedding_client,
                distance_threshold=self.config.get("cluster_distance_threshold", 0.9),
                min_cluster_size=self.config.get("min_cluster_size", 2),
            )
            final_triples_tuples = st.semantic_results["final_triples"]
            st.triples_final = [
                TripleRecord(subject=s, relation=r, object=o, evidence=ev)
                for s, r, o, ev in final_triples_tuples
            ]

    # -------------------------------------------------------------------
    # Save per-doc outputs in the exact sequential format
    # -------------------------------------------------------------------
    def _save_all_experiments(self, states: Dict[str, DocState]) -> None:
        self.logger.info("Saving per-document outputs...")

        for st in states.values():
            doc_dir_name = f"{self.experiment_name}/{st.doc_id}"
            # Create a disposable KGCreationPipeline instance just to use its save logic
            per_doc = KGCreationPipeline(
                llm_client=self.llm,
                sentences_per_chunk=self.config["sentences_per_chunk"],
                summarize_text=self.config.get("summarize_text", True),
                entity_refinement_mode=self.config.get("entity_refinement_mode", "llm"),
                enable_semantic_linking=self.config.get("enable_semantic_linking", False),
                embedding_model=self.config.get("embedding_model", "text-embedding-3-large"),
                embedding_backend=self.config.get("embedding_backend", "openai"),
                cluster_distance_threshold=self.config.get("cluster_distance_threshold", 0.9),
                min_cluster_size=self.config.get("min_cluster_size", 2),
                save_experiments=True,
                experiment_name=doc_dir_name,
                logger=self.logger,
                store_to_neo4j=False,
            )

            # Build final_result in the same shape as pipeline.run()
            final_result: Dict[str, Any] = {
                "summary": st.summary,
                "chunks": [c.as_dict() for c in st.chunks],
                "entities_raw": st.entities_raw,
                "entities_refined": st.entities_refined,
                "triples_final": [t.as_dict() for t in st.triples_final],
            }
            if st.semantic_results:
                final_result["semantic_triples"] = [
                    {"subject": s, "relation": r, "object": o, "evidence": ev}
                    for s, r, o, ev in st.semantic_results["semantic_triples"]
                ]
                final_result["entity_clusters"] = st.semantic_results["entity_clusters"]

            per_doc._save_experiment_data(
                processed_chunks=[c.text for c in st.chunks],
                entities_raw=st.entities_raw,
                entities_refined=st.entities_refined,
                summary=st.summary,
                triples_final=st.triples_final,
                semantic_results=st.semantic_results,
                resolution_history=st.resolution_history,
                verification_stats=st.verification_stats,
                final_result=final_result,
            )

    # -------------------------------------------------------------------
    # Shared retry helper: parse batch results + fall back to sync generate_json
    # -------------------------------------------------------------------
    def _iter_parsed_json_results(
        self,
        results: Dict[str, BatchResultEntry],
        requests: Sequence[BatchRequest],
        retry_phase_label: str,
    ):
        """
        Yield (custom_id, parsed_json_payload) for every request. For any entry
        that failed (API error, missing, or invalid JSON), fall back to a sync
        call using the existing LLMClient.generate_json() which has its own
        retry + JSON-repair logic.
        """
        req_by_id: Dict[str, BatchRequest] = {r.custom_id: r for r in requests}

        for req in requests:
            cid = req.custom_id
            entry = results.get(cid)
            payload: Optional[Dict[str, Any]] = None

            if entry is not None and entry.success and entry.content:
                try:
                    payload = json.loads(LLMClient._clean_json_response(entry.content))
                except Exception as exc:
                    self.logger.warning(
                        f"[{retry_phase_label}] JSON parse failed for {cid}: {exc}"
                    )
                    payload = None

            if payload is None:
                self.logger.info(f"[{retry_phase_label}] Sync retry for {cid}")
                payload = self._sync_retry(req)

            if payload is not None:
                yield cid, payload

    def _sync_retry(self, req: BatchRequest) -> Optional[Dict[str, Any]]:
        """
        Re-issue a failed batch request synchronously using LLMClient.generate_json.
        Extracts system/user from the original request body. Returns None if the
        sync retry also fails (caller uses the same fallback behavior as
        sequential pipeline — empty list, skip, etc.).
        """
        try:
            body = req.body
            msgs = body.get("input") or body.get("messages") or []
            system = next(
                (m["content"] for m in msgs if m.get("role") == "system"), ""
            )
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            return self.llm.generate_json(
                system_prompt=system,
                user_prompt=user,
                max_output_tokens=body.get("max_output_tokens")
                or body.get("max_tokens"),
            )
        except Exception as exc:
            self.logger.error(f"   Sync retry failed for {req.custom_id}: {exc}")
            return None

    # -------------------------------------------------------------------
    # Dry-run reporting
    # -------------------------------------------------------------------
    def _report_dry_run_counts(self, states: Dict[str, DocState]) -> None:
        total_chunks = sum(len(st.chunks) for st in states.values())
        n_docs = len(states)
        summarize = self.config.get("summarize_text", True)
        lines = [
            "",
            "DRY RUN — nothing submitted to OpenAI.",
            f"  Docs: {n_docs}",
            f"  Total chunks: {total_chunks}",
            f"  Batch 1 requests: {total_chunks} (entity) + {n_docs if summarize else 0} (summary) = {total_chunks + (n_docs if summarize else 0)}",
            f"  Batch 2 requests: up to {n_docs} (refinement, LLM mode)",
            f"  Batch 3 requests: {total_chunks} (triple extraction)",
            "  Batch 4 requests: depends on how many triples come back invalid",
            "  Embeddings batch: depends on refined entities + contexts (computed after phase 5)",
            f"  Input file for batch 1 written to: {self.batch_workdir / 'batch1_entity_summary' / 'input.jsonl'}",
        ]
        for line in lines:
            self.logger.info(line)
