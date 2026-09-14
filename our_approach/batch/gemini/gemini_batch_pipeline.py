"""
Gemini Batch KG Pipeline

Subclasses batch.batch_pipeline.BatchKGPipeline, overriding only the
API-specific methods (request building, batch submission, result parsing,
sync retry).  All pipeline logic — preprocessing, entity dedup, triple
parsing/validation, object resolution, experiment saving — is inherited
unchanged from the parent class.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OUR_APPROACH_DIR = Path(__file__).resolve().parent.parent.parent
if str(_OUR_APPROACH_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH_DIR))

from llm_client import LLMClient, EmbeddingClient  # noqa: E402
from pipeline import (  # noqa: E402
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
from semantic_linking import (  # noqa: E402
    find_entity_context_from_triples,
    run_semantic_linking_pipeline,
)

# Parent class — we inherit all API-agnostic pipeline logic from here
from batch.batch_pipeline import (  # noqa: E402
    BatchKGPipeline,
    CachedEmbeddingClient,
    DocState,
)
from batch.batch_client import BatchResultEntry  # noqa: E402

# Gemini-specific batch client
from batch.gemini.gemini_batch_client import (  # noqa: E402
    GeminiBatchClient,
    GeminiBatchRequest,
    build_embed_request,
    build_generate_request,
)

from nltk.stem import WordNetLemmatizer  # noqa: E402


class GeminiBatchKGPipeline(BatchKGPipeline):
    """
    Runs the full KG pipeline using the Gemini Batch API for LLM and
    embedding calls.  Inherits all pipeline logic from BatchKGPipeline,
    overriding only:
        - __init__          (creates GeminiBatchClient instead of OpenAI BatchClient)
        - _batch_entity_and_summary
        - _batch_entity_refinement
        - _batch_triple_extraction
        - _batch_triple_verification
        - _semantic_linking
        - _iter_parsed_json_results
        - _sync_retry
        - _report_dry_run_counts
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        # Intentionally bypass BatchKGPipeline.__init__() to avoid creating
        # an OpenAI BatchClient.  Replicate the setup with Gemini clients.
        self.config = config
        self.experiment_name: str = config["experiment_name"]
        self.experiment_dir: Path = (
            _OUR_APPROACH_DIR / "experiments" / self.experiment_name
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._build_logger()

        # Sync LLM client — for retrying batch failures and sequential phases.
        # model_type should be "gemini" so generate_json() routes correctly.
        self.llm = LLMClient(
            model=config["model"],
            model_type=config["model_type"],
            base_url=config.get("base_url"),
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            max_retries=config["max_retries"],
            logger=self.logger,
        )

        # Gemini batch client
        self.batch_workdir = self.experiment_dir / "batch_workdir"
        self.batch = GeminiBatchClient(
            workdir=self.batch_workdir,
            poll_interval_seconds=60,
            logger=self.logger,
        )

        # Model names for batch creation (Gemini requires model at submit time)
        self.generation_model: str = config["model"]
        self.embedding_model: str = config.get("embedding_model", "gemini-embedding-2")

    # -------------------------------------------------------------------
    # Batch 1: entity extraction + summarisation
    # -------------------------------------------------------------------
    def _batch_entity_and_summary(
        self, states: Dict[str, DocState], dry_run: bool
    ) -> None:
        self.logger.info("BATCH 1: Local Entity Extraction + Summarization")
        requests: List[GeminiBatchRequest] = []

        for st in states.values():
            for chunk in st.chunks:
                requests.append(
                    build_generate_request(
                        key=f"entity|{st.doc_id}|{chunk.chunk_id}",
                        system_prompt=LOCAL_ENTITY_SYSTEM_PROMPT,
                        user_prompt=build_local_entity_user_prompt(chunk.text),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )
            if self.config.get("summarize_text", True):
                requests.append(
                    build_generate_request(
                        key=f"summary|{st.doc_id}",
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
            model=self.generation_model,
            request_type="generate",
            requests=requests,
        )

        # Distribute results back to DocStates (with sync retry on failure)
        for key, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch1",
        ):
            if key.startswith("entity|"):
                _, doc_id, chunk_id_str = key.split("|", 2)
                chunk_id = int(chunk_id_str)
                chunk = next(
                    c for c in states[doc_id].chunks if c.chunk_id == chunk_id
                )
                candidates = payload.get("entities", []) or []
                clean_candidates = [
                    e
                    for e in (
                        _clean_entity(item, enforce_limit=True)
                        for item in candidates
                    )
                    if e
                ]
                chunk.entities = clean_candidates
                states[doc_id].entities_raw.extend(clean_candidates)
            elif key.startswith("summary|"):
                _, doc_id = key.split("|", 1)
                summary = (payload.get("summary") or "").strip()
                if not summary:
                    summary = states[doc_id].text[:500]
                states[doc_id].summary = summary

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

        # Deterministic lemmatisation dedup first (inherited logic)
        lemmatizer = WordNetLemmatizer()
        for st in states.values():
            seen: set = set()
            normalized: List[str] = []
            for w in st.entities_raw:
                k = lemmatizer.lemmatize(w.lower())
                if k not in seen:
                    seen.add(k)
                    normalized.append(w)
            st.entities_raw = normalized

        if mode != "llm":
            for st in states.values():
                st.entities_refined = self._dedupe_entities(st.entities_raw)
            return

        requests: List[GeminiBatchRequest] = []
        for st in states.values():
            requests.append(
                build_generate_request(
                    key=f"refine|{st.doc_id}",
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
            model=self.generation_model,
            request_type="generate",
            requests=requests,
        )

        for key, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch2"
        ):
            _, doc_id = key.split("|", 1)
            refined = payload.get("entities_refined", []) or []
            states[doc_id].entities_refined = self._dedupe_entities(refined)

        for st in states.values():
            if not st.entities_refined:
                self.logger.warning(
                    f"   {st.doc_id}: refinement empty, falling back to raw entities"
                )
                st.entities_refined = self._dedupe_entities(st.entities_raw)
            self.logger.info(
                f"   {st.doc_id}: {len(st.entities_refined)} refined entities"
            )

    # -------------------------------------------------------------------
    # Batch 3: triple extraction
    # -------------------------------------------------------------------
    def _batch_triple_extraction(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH 3: Triple Extraction")
        requests: List[GeminiBatchRequest] = []
        for st in states.values():
            for chunk in st.chunks:
                requests.append(
                    build_generate_request(
                        key=f"triple|{st.doc_id}|{chunk.chunk_id}",
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
            model=self.generation_model,
            request_type="generate",
            requests=requests,
        )

        for key, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch3"
        ):
            _, doc_id, chunk_id_str = key.split("|", 2)
            chunk_id = int(chunk_id_str)
            chunk = next(
                c for c in states[doc_id].chunks if c.chunk_id == chunk_id
            )
            chunk.triples_raw = self._parse_triples_payload(
                payload, chunk_id=chunk.chunk_id, chunk_text=chunk.text
            )

        for st in states.values():
            total = sum(len(c.triples_raw) for c in st.chunks)
            self.logger.info(f"   {st.doc_id}: {total} raw triples")

    # -------------------------------------------------------------------
    # Batch 4: triple verification
    # -------------------------------------------------------------------
    def _batch_triple_verification(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH 4: Triple Verification")

        invalids: List[Tuple[str, int, int, TripleRecord]] = []
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
                        invalids.append(
                            (st.doc_id, chunk.chunk_id, idx, tr)
                        )
            stats_per_doc[st.doc_id] = {
                "total_triples": total,
                "valid_count": valid,
                "fixed_count": 0,
                "failed_count": 0,
                "details": details,
            }

        self.logger.info(
            f"   Identified {len(invalids)} invalid triples across "
            f"{len(states)} docs"
        )

        if not invalids:
            for doc_id, stats in stats_per_doc.items():
                states[doc_id].verification_stats = stats
            return

        requests: List[GeminiBatchRequest] = []
        for doc_id, chunk_id, triple_idx, tr in invalids:
            requests.append(
                build_generate_request(
                    key=f"verify|{doc_id}|{chunk_id}|{triple_idx}",
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
            model=self.generation_model,
            request_type="generate",
            requests=requests,
        )

        payload_by_id: Dict[str, Any] = {}
        for k, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batch4"
        ):
            payload_by_id[k] = payload

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
            new_relation = _clean_relation(
                payload.get("relation", tr.relation)
            )
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

        for st in states.values():
            for chunk in st.chunks:
                chunk.triples_raw = keep_per_chunk[(st.doc_id, chunk.chunk_id)]
            st.verification_stats = stats_per_doc[st.doc_id]
            s = st.verification_stats
            self.logger.info(
                f"   {st.doc_id}: total={s['total_triples']} "
                f"valid={s['valid_count']} fixed={s['fixed_count']} "
                f"discarded={s['failed_count']}"
            )

    # -------------------------------------------------------------------
    # Semantic linking: batch embeddings via Gemini, then sequential
    # clustering + LLM
    # -------------------------------------------------------------------
    def _semantic_linking(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 6 (batched embeddings): Semantic Linking")
        embed_model = self.config.get("embedding_model", "gemini-embedding-2")
        backend = self.config.get("embedding_backend", "gemini")

        # For non-Gemini backends, fall back to sequential (inherited)
        if backend not in ("gemini", "openai"):
            self._semantic_linking_sequential(states)
            return

        # Step 1: collect texts to embed (same logic as parent)
        texts_to_embed: List[str] = []
        seen_texts: set = set()
        for st in states.values():
            triples_with_ev = [
                (t.subject, t.relation, t.object, t.evidence)
                for t in st.triples_final
            ]
            entity_texts = list(st.entities_refined)
            for entity in entity_texts:
                contexts = find_entity_context_from_triples(
                    entity, triples_with_ev, max_contexts=5
                )
                combined = " ".join(contexts[:3])
                # entity text + context text
                for t in [entity, combined]:
                    if t not in seen_texts:
                        seen_texts.add(t)
                        texts_to_embed.append(t)

        self.logger.info(
            f"   Embedding {len(texts_to_embed)} unique texts across "
            f"{len(states)} docs"
        )

        # Step 2: build batch embedding requests
        requests: List[GeminiBatchRequest] = []
        index_to_text: Dict[int, str] = {}
        for i, text in enumerate(texts_to_embed):
            requests.append(build_embed_request(key=f"embed|{i}", text=text))
            index_to_text[i] = text

        embedding_cache: Dict[str, List[float]] = {}
        if requests:
            results = self.batch.run_job(
                job_name="batch5_embeddings",
                model=embed_model,
                request_type="embed",
                requests=requests,
            )
            for req in requests:
                entry = results.get(req.key)
                text = index_to_text[int(req.key.split("|", 1)[1])]
                if entry is None or not entry.success or entry.embedding is None:
                    self.logger.warning(
                        f"   Embedding failed for key={req.key}; retrying sync..."
                    )
                    fallback = EmbeddingClient(
                        model=embed_model,
                        backend=backend,
                        logger=self.logger,
                    )
                    embedding_cache[text] = fallback.embed_query(text)
                else:
                    embedding_cache[text] = entry.embedding

        cached_client = CachedEmbeddingClient(
            cache=embedding_cache, logger=self.logger
        )

        # Step 3: per-doc clustering + cluster-relation LLM calls (sequential)
        for st in states.values():
            triples_with_ev = [
                (t.subject, t.relation, t.object, t.evidence)
                for t in st.triples_final
            ]
            self.logger.info(f"   Semantic linking for {st.doc_id}...")
            st.semantic_results = run_semantic_linking_pipeline(
                entities_refined=st.entities_refined,
                triples_final=triples_with_ev,
                global_summary=st.summary,
                llm_client=self.llm,
                embedding_client=cached_client,
                distance_threshold=self.config.get(
                    "cluster_distance_threshold", 0.9
                ),
                min_cluster_size=self.config.get("min_cluster_size", 2),
            )
            final_triples_tuples = st.semantic_results["final_triples"]
            st.triples_final = [
                TripleRecord(subject=s, relation=r, object=o, evidence=ev)
                for s, r, o, ev in final_triples_tuples
            ]

    # -------------------------------------------------------------------
    # Retry helpers — adapted for Gemini request shapes
    # -------------------------------------------------------------------
    def _iter_parsed_json_results(
        self,
        results: Dict[str, BatchResultEntry],
        requests: Sequence[GeminiBatchRequest],
        retry_phase_label: str,
    ):
        """
        Yield (key, parsed_json_payload) for every request.  Falls back to
        a sync LLMClient.generate_json() call on failure.
        """
        for req in requests:
            k = req.key
            entry = results.get(k)
            payload: Optional[Dict[str, Any]] = None

            if entry is not None and entry.success and entry.content:
                try:
                    payload = json.loads(
                        LLMClient._clean_json_response(entry.content)
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"[{retry_phase_label}] JSON parse failed for {k}: {exc}"
                    )
                    payload = None

            if payload is None:
                self.logger.info(
                    f"[{retry_phase_label}] Sync retry for {k}"
                )
                payload = self._sync_retry(req)

            if payload is not None:
                yield k, payload

    def _sync_retry(
        self, req: GeminiBatchRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Re-issue a failed Gemini batch request synchronously via
        LLMClient.generate_json().  Extracts system/user prompts from the
        Gemini request body shape.
        """
        try:
            body = req.request

            # System instruction
            sys_instr = body.get("system_instruction", {})
            sys_parts = sys_instr.get("parts", [])
            system = sys_parts[0].get("text", "") if sys_parts else ""

            # User content
            contents = body.get("contents", [])
            user = ""
            for c in contents:
                if c.get("role") == "user":
                    parts = c.get("parts", [])
                    user = parts[0].get("text", "") if parts else ""
                    break

            gen_config = body.get("generation_config", {})
            return self.llm.generate_json(
                system_prompt=system,
                user_prompt=user,
                max_output_tokens=gen_config.get("max_output_tokens"),
            )
        except Exception as exc:
            self.logger.error(f"   Sync retry failed for {req.key}: {exc}")
            return None

    # -------------------------------------------------------------------
    # Dry-run reporting (cosmetic: says "Gemini" instead of "OpenAI")
    # -------------------------------------------------------------------
    def _report_dry_run_counts(self, states: Dict[str, DocState]) -> None:
        total_chunks = sum(len(st.chunks) for st in states.values())
        n_docs = len(states)
        summarize = self.config.get("summarize_text", True)
        lines = [
            "",
            "DRY RUN — nothing submitted to Gemini.",
            f"  Model: {self.generation_model}",
            f"  Docs: {n_docs}",
            f"  Total chunks: {total_chunks}",
            f"  Batch 1 requests: {total_chunks} (entity) + "
            f"{n_docs if summarize else 0} (summary) = "
            f"{total_chunks + (n_docs if summarize else 0)}",
            f"  Batch 2 requests: up to {n_docs} (refinement, LLM mode)",
            f"  Batch 3 requests: {total_chunks} (triple extraction)",
            "  Batch 4 requests: depends on how many triples come back invalid",
            "  Embeddings batch: depends on refined entities + contexts "
            "(computed after phase 5)",
            f"  Input file for batch 1 written to: "
            f"{self.batch_workdir / 'batch1_entity_summary' / 'input.jsonl'}",
        ]
        for line in lines:
            self.logger.info(line)
