"""
Batched RAKG Pipeline

Orchestrates the sequential RAKG pipeline (baselines/rakg/pipeline.py) across
many documents by replacing every LLM and embedding call with OpenAI Batch API
submissions. Mirrors the structure of our_approach/batch/batch_pipeline.py.

DAG of batches (each level waits on the previous one):

    Phase 1 (local):    sentence segmentation
    Batches S + N:      sentence embeddings  ||  per-sentence NER
                        (different endpoints — submitted in parallel,
                         waited together; both depend only on phase 1)
    Batch E:            embeddings for "name type" of every raw entity
    Local:              per-doc cosine matrix → candidate pairs
    Batch D:            disambiguation LLM call per candidate pair
    Local:              per-doc union-find merging
    Batch Q:            query-name embeddings for every merged entity
    Batch K:            per-merged-entity KG extraction LLM calls
    Local:              KG conversion + save to experiments/

Failures from any batch (JSON parse errors, API errors, batch_expired) are
retried synchronously via the shared LLMClient.generate_json, preserving the
fallback behaviour of the sequential pipeline.

Output: one folder per document under
    baselines/rakg/experiments/<experiment_name>/<doc_id>/
with the same JSON files the sequential RAKG pipeline writes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Path setup ────────────────────────────────────────────────────────────────
# Both baselines/rakg/pipeline.py and our_approach/pipeline.py exist; importing
# the wrong one would shadow RAKG prompts. Import RAKG first, register it in
# sys.modules under a unique alias, *then* add our_approach to sys.path.
_THIS_DIR = Path(__file__).resolve().parent                      # .../baselines/rakg/batch
_RAKG_DIR = _THIS_DIR.parent                                     # .../baselines/rakg
_OUR_APPROACH_DIR = _THIS_DIR.parent.parent.parent / "our_approach"

import importlib.util as _ilu  # noqa: E402

_rakg_spec = _ilu.spec_from_file_location("rakg_pipeline", _RAKG_DIR / "pipeline.py")
_rakg_pipeline_mod = _ilu.module_from_spec(_rakg_spec)
sys.modules["rakg_pipeline"] = _rakg_pipeline_mod
_rakg_spec.loader.exec_module(_rakg_pipeline_mod)

KG_EXTRACTION_SYSTEM_PROMPT = _rakg_pipeline_mod.KG_EXTRACTION_SYSTEM_PROMPT
KG_EXTRACTION_USER_PROMPT_TEMPLATE = _rakg_pipeline_mod.KG_EXTRACTION_USER_PROMPT_TEMPLATE
NER_SYSTEM_PROMPT = _rakg_pipeline_mod.NER_SYSTEM_PROMPT
NER_USER_PROMPT_TEMPLATE = _rakg_pipeline_mod.NER_USER_PROMPT_TEMPLATE
SIMILARITY_SYSTEM_PROMPT = _rakg_pipeline_mod.SIMILARITY_SYSTEM_PROMPT
SIMILARITY_USER_PROMPT_TEMPLATE = _rakg_pipeline_mod.SIMILARITY_USER_PROMPT_TEMPLATE
RAKGPipeline = _rakg_pipeline_mod.RAKGPipeline
split_sentences = _rakg_pipeline_mod.split_sentences

# Now safe to add our_approach to sys.path for the shared LLM + batch clients
if str(_OUR_APPROACH_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_APPROACH_DIR))

from llm_client import LLMClient, EmbeddingClient  # noqa: E402
from batch.batch_client import (  # noqa: E402
    BatchClient,
    BatchJobMeta,
    BatchRequest,
    BatchResultEntry,
    build_embeddings_request,
    build_responses_request,
)


def _ensure_json_word(prompt: str) -> str:
    """OpenAI /v1/responses with text.format type 'json_object' requires the
    literal word 'json' somewhere in the input messages. RAKG's original
    prompts use schema templates but never the literal word, so we append a
    minimal marker rather than modifying the original prompt constants."""
    if "json" in prompt.lower():
        return prompt
    return prompt.rstrip() + "\n\nRespond with valid JSON only."


# ─────────────────────────────────────────────────────────────────────────────
# Per-document state
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DocState:
    doc_id: str
    text: str
    sentences: List[str] = field(default_factory=list)
    sentence_to_id: Dict[str, str] = field(default_factory=dict)
    id_to_sentence: Dict[str, str] = field(default_factory=dict)
    sentence_vectors: List[List[float]] = field(default_factory=list)

    # entity_key ("entity1", "entity2", ...) → {"name", "type", "description", "chunkid"}
    raw_entities: Dict[str, Any] = field(default_factory=dict)

    candidate_pairs: List[Tuple[str, str]] = field(default_factory=list)
    confirmed_pairs: List[Tuple[str, str]] = field(default_factory=list)
    merged_entities: Dict[str, Any] = field(default_factory=dict)

    # entity_key → KG-extraction response payload
    raw_kg: Dict[str, Any] = field(default_factory=dict)
    converted_kg: Dict[str, Any] = field(default_factory=dict)

    elapsed: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class BatchRAKGPipeline:
    """Runs the RAKG pipeline across many documents using OpenAI Batch API."""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.experiment_name: str = config["experiment_name"]
        self.experiment_dir: Path = (
            _RAKG_DIR / "experiments" / self.experiment_name
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._build_logger()

        # Sync LLM client — for retrying batch failures
        self.llm = LLMClient(
            model=config["model"],
            model_type=config["model_type"],
            base_url=config.get("base_url"),
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            max_retries=config["max_retries"],
            logger=self.logger,
        )

        # Batch client — one workdir per experiment
        self.batch_workdir = self.experiment_dir / "batch_workdir"
        self.batch = BatchClient(
            workdir=self.batch_workdir,
            poll_interval_seconds=config.get("poll_interval_seconds", 60),
            logger=self.logger,
        )

        # Sync embedding fallback (only used when a batch embedding entry fails)
        self._sync_embed: Optional[EmbeddingClient] = None

    # -------------------------------------------------------------------------
    # Logging — same format as sequential RAKG pipeline
    # -------------------------------------------------------------------------
    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("BatchRAKGPipeline")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        if self.config.get("log_to_file", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(
                self.experiment_dir / f"pipeline_{timestamp}.log",
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(fh)

        if self.config.get("log_to_console", True):
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
            logger.addHandler(ch)

        return logger

    # -------------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------------
    def run(
        self, docs: Dict[str, str], dry_run: bool = False
    ) -> Dict[str, DocState]:
        """
        Args:
            docs: {doc_id: text}
            dry_run: write input .jsonl files only, do not submit to OpenAI.
        """
        states: Dict[str, DocState] = {
            doc_id: DocState(doc_id=doc_id, text=text)
            for doc_id, text in docs.items()
        }

        run_t0 = time.time()
        self.logger.info("=" * 70)
        self.logger.info(
            f"BATCH RAKG PIPELINE -- {len(states)} docs -- "
            f"experiment={self.experiment_name}"
        )
        self.logger.info(f"Model: {self.config['model']}")
        self.logger.info(f"Embedding model: {self.config['embedding_model']}")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info("=" * 70)

        # Phase 1 (local): sentence segmentation
        self._phase1_split(states)

        # Batches S + N (parallel): sentence embeddings  ||  per-sentence NER
        sentence_embed_cache = self._batches_sentences_and_ner(states, dry_run=dry_run)
        if dry_run:
            self._report_dry_run(states)
            return states

        # Distribute sentence vectors back to each doc
        for st in states.values():
            st.sentence_vectors = [sentence_embed_cache[s] for s in st.sentences]

        # Batch E: embed "name type" of every raw entity → per-doc candidate pairs
        self._batch_entity_embeddings_and_candidates(states)

        # Batch D: disambiguation per candidate pair
        self._batch_disambiguation(states)

        # Local: union-find merging
        self._merge_entities_local(states)

        # Batch Q: query-name embedding per merged entity (per doc)
        query_embed_cache = self._batch_query_embeddings(states)

        # Batch K: KG extraction per merged entity
        self._batch_kg_extraction(states, query_embed_cache)

        # Local: KG conversion + save
        self._convert_and_save(states, run_t0)

        self.logger.info("=" * 70)
        self.logger.info("BATCH RAKG PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        return states

    # -------------------------------------------------------------------------
    # Phase 1: sentence segmentation (local)
    # -------------------------------------------------------------------------
    def _phase1_split(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 1: Sentence Segmentation")
        total = 0
        for st in states.values():
            sents = split_sentences(st.text.strip())
            st.sentences = sents
            st.sentence_to_id = {s: f"s{i+1}" for i, s in enumerate(sents)}
            st.id_to_sentence = {f"s{i+1}": s for i, s in enumerate(sents)}
            total += len(sents)
            self.logger.debug(f"   {st.doc_id}: {len(sents)} sentences")
        self.logger.info(f"   Total sentences across docs: {total}")

    # -------------------------------------------------------------------------
    # Batches S + N: sentence embeddings  ||  per-sentence NER (parallel)
    # -------------------------------------------------------------------------
    def _batches_sentences_and_ner(
        self, states: Dict[str, DocState], dry_run: bool
    ) -> Dict[str, List[float]]:
        self.logger.info("BATCH S + N: Sentence Embeddings  ||  Per-Sentence NER")

        # ----- Build Batch S: deduplicated sentence embeddings -----
        seen: set = set()
        unique_sents: List[str] = []
        for st in states.values():
            for s in st.sentences:
                if s not in seen:
                    seen.add(s)
                    unique_sents.append(s)

        embed_requests: List[BatchRequest] = []
        for i, s in enumerate(unique_sents):
            embed_requests.append(
                build_embeddings_request(
                    custom_id=f"sent_embed|{i}",
                    model=self.config["embedding_model"],
                    text=s,
                )
            )

        # ----- Build Batch N: NER per (doc_id, sentence_index) -----
        ner_requests: List[BatchRequest] = []
        for st in states.values():
            for idx, sent in enumerate(st.sentences):
                ner_requests.append(
                    build_responses_request(
                        custom_id=f"ner|{st.doc_id}|{idx}",
                        model=self.config["model"],
                        system_prompt=_ensure_json_word(NER_SYSTEM_PROMPT),
                        user_prompt=NER_USER_PROMPT_TEMPLATE.format(text=sent),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )

        self.logger.info(
            f"   Batch S: {len(embed_requests)} unique sentence embeddings"
        )
        self.logger.info(f"   Batch N: {len(ner_requests)} NER calls")

        if dry_run:
            self.batch.write_input_file("batchS_sent_embed", embed_requests)
            self.batch.write_input_file("batchN_ner", ner_requests)
            return {}

        # ----- Submit both, wait both (parallel on OpenAI) -----
        jobs = [
            {
                "name": "batchS_sent_embed",
                "endpoint": "/v1/embeddings",
                "requests": embed_requests,
            },
            {
                "name": "batchN_ner",
                "endpoint": "/v1/responses",
                "requests": ner_requests,
            },
        ]
        results_by_job = self._run_jobs_parallel(jobs)

        # ----- Cache sentence embeddings -----
        embed_cache: Dict[str, List[float]] = {}
        embed_results = results_by_job["batchS_sent_embed"]
        for req in embed_requests:
            cid = req.custom_id
            entry = embed_results.get(cid)
            text = unique_sents[int(cid.split("|", 1)[1])]
            if entry is not None and entry.success and entry.embedding is not None:
                embed_cache[text] = entry.embedding
            else:
                self.logger.warning(
                    f"   [batchS] embedding miss for cid={cid}; sync fallback"
                )
                embed_cache[text] = self._sync_embed_one(text)

        # ----- Distribute NER results into per-doc raw_entities -----
        # We must assign deterministic entity keys per doc, in sentence order,
        # matching the sequential pipeline's behaviour.
        ner_payloads: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
        for cid, payload in self._iter_parsed_json_results(
            results_by_job["batchN_ner"], ner_requests, retry_phase_label="batchN"
        ):
            _, doc_id, idx_str = cid.split("|", 2)
            ner_payloads[(doc_id, int(idx_str))] = payload

        for st in states.values():
            entity_num = 1
            for idx, sent in enumerate(st.sentences):
                payload = ner_payloads.get((st.doc_id, idx))
                if not payload:
                    continue
                # The model signals "no information" with {"State": false}
                if "State" in payload or "state" in payload:
                    continue
                # Per the prompt, payload keys are entity1/entity2/...; values
                # have name/type/description. We renumber and stamp chunkid.
                for _old_key, value in payload.items():
                    if not isinstance(value, dict):
                        continue
                    if not value.get("name"):
                        continue
                    new_key = f"entity{entity_num}"
                    value = dict(value)  # don't mutate the parsed payload
                    value.setdefault("type", "")
                    value.setdefault("description", "")
                    value["chunkid"] = st.sentence_to_id.get(sent, f"s{idx+1}")
                    st.raw_entities[new_key] = value
                    entity_num += 1
            self.logger.info(
                f"   {st.doc_id}: {len(st.sentences)} sentences -> "
                f"{len(st.raw_entities)} raw entities"
            )

        return embed_cache

    # -------------------------------------------------------------------------
    # Batch E: entity (name+type) embeddings → per-doc candidate pairs
    # -------------------------------------------------------------------------
    def _batch_entity_embeddings_and_candidates(
        self, states: Dict[str, DocState]
    ) -> None:
        self.logger.info("BATCH E: Entity (name+type) embeddings + candidates")
        threshold = self.config.get("similarity_threshold", 0.60)

        # Build deduplicated entity-text list
        seen: set = set()
        unique_texts: List[str] = []
        for st in states.values():
            for v in st.raw_entities.values():
                t = f"{v.get('name','')} {v.get('type','')}".strip()
                if t and t not in seen:
                    seen.add(t)
                    unique_texts.append(t)

        embed_cache: Dict[str, List[float]] = {}
        if unique_texts:
            requests: List[BatchRequest] = [
                build_embeddings_request(
                    custom_id=f"ent_embed|{i}",
                    model=self.config["embedding_model"],
                    text=t,
                )
                for i, t in enumerate(unique_texts)
            ]
            self.logger.info(f"   {len(requests)} unique entity-text embeddings")
            results = self.batch.run_job(
                job_name="batchE_entity_embed",
                endpoint="/v1/embeddings",
                requests=requests,
            )
            for req in requests:
                cid = req.custom_id
                entry = results.get(cid)
                text = unique_texts[int(cid.split("|", 1)[1])]
                if entry is not None and entry.success and entry.embedding is not None:
                    embed_cache[text] = entry.embedding
                else:
                    self.logger.warning(
                        f"   [batchE] embedding miss for cid={cid}; sync fallback"
                    )
                    embed_cache[text] = self._sync_embed_one(text)

        # Per-doc candidate computation (cosine + threshold)
        total_pairs = 0
        for st in states.values():
            keys = list(st.raw_entities.keys())
            if len(keys) < 2:
                st.candidate_pairs = []
                continue
            texts = [
                f"{st.raw_entities[k].get('name','')} {st.raw_entities[k].get('type','')}".strip()
                for k in keys
            ]
            vectors = np.array([embed_cache[t] for t in texts])
            sim_matrix = np.zeros((len(keys), len(keys)))
            for i, j in combinations(range(len(keys)), 2):
                sim_matrix[i][j] = cosine_similarity([vectors[i]], [vectors[j]])[0][0]

            pairs = [
                (keys[i], keys[j])
                for i, j in zip(*np.where(sim_matrix > threshold))
            ]
            st.candidate_pairs = pairs
            total_pairs += len(pairs)
            self.logger.debug(f"   {st.doc_id}: {len(pairs)} candidate pairs")
        self.logger.info(f"   Total candidate pairs: {total_pairs}")

    # -------------------------------------------------------------------------
    # Batch D: disambiguation LLM call per candidate pair
    # -------------------------------------------------------------------------
    def _batch_disambiguation(self, states: Dict[str, DocState]) -> None:
        self.logger.info("BATCH D: Entity Disambiguation")
        requests: List[BatchRequest] = []
        for st in states.values():
            for ent_a, ent_b in st.candidate_pairs:
                e1 = st.raw_entities.get(ent_a)
                e2 = st.raw_entities.get(ent_b)
                if not e1 or not e2:
                    continue
                requests.append(
                    build_responses_request(
                        custom_id=f"disambig|{st.doc_id}|{ent_a}|{ent_b}",
                        model=self.config["model"],
                        system_prompt=_ensure_json_word(SIMILARITY_SYSTEM_PROMPT),
                        user_prompt=SIMILARITY_USER_PROMPT_TEMPLATE.format(
                            entity1=json.dumps(e1, ensure_ascii=False),
                            entity2=json.dumps(e2, ensure_ascii=False),
                        ),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )

        self.logger.info(f"   {len(requests)} disambiguation calls")
        if not requests:
            for st in states.values():
                st.confirmed_pairs = []
            return

        results = self.batch.run_job(
            job_name="batchD_disambig",
            endpoint="/v1/responses",
            requests=requests,
        )

        for cid, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batchD"
        ):
            _, doc_id, ent_a, ent_b = cid.split("|", 3)
            if bool(payload.get("result", False)):
                states[doc_id].confirmed_pairs.append((ent_a, ent_b))

        for st in states.values():
            self.logger.info(
                f"   {st.doc_id}: {len(st.confirmed_pairs)} confirmed duplicates"
            )

    # -------------------------------------------------------------------------
    # Local: union-find merging (delegates to RAKGPipeline._merge_entities)
    # -------------------------------------------------------------------------
    def _merge_entities_local(self, states: Dict[str, DocState]) -> None:
        self.logger.info("PHASE 6 (local): Union-find merging")
        for st in states.values():
            before = len(st.raw_entities)
            # _merge_entities mutates the dict in place AND returns it.
            # Pass a copy so st.raw_entities stays intact for any debugging.
            st.merged_entities = RAKGPipeline._merge_entities(
                dict(st.raw_entities), st.confirmed_pairs
            )
            self.logger.info(
                f"   {st.doc_id}: {before} -> {len(st.merged_entities)} entities"
            )

    # -------------------------------------------------------------------------
    # Batch Q: query-name embedding per merged entity
    # -------------------------------------------------------------------------
    def _batch_query_embeddings(
        self, states: Dict[str, DocState]
    ) -> Dict[str, List[float]]:
        self.logger.info("BATCH Q: Per-merged-entity query embeddings")
        seen: set = set()
        unique_names: List[str] = []
        for st in states.values():
            for v in st.merged_entities.values():
                name = v.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    unique_names.append(name)

        cache: Dict[str, List[float]] = {}
        if not unique_names:
            return cache

        requests: List[BatchRequest] = [
            build_embeddings_request(
                custom_id=f"qry_embed|{i}",
                model=self.config["embedding_model"],
                text=name,
            )
            for i, name in enumerate(unique_names)
        ]
        self.logger.info(f"   {len(requests)} unique query embeddings")
        results = self.batch.run_job(
            job_name="batchQ_query_embed",
            endpoint="/v1/embeddings",
            requests=requests,
        )

        for req in requests:
            cid = req.custom_id
            entry = results.get(cid)
            name = unique_names[int(cid.split("|", 1)[1])]
            if entry is not None and entry.success and entry.embedding is not None:
                cache[name] = entry.embedding
            else:
                self.logger.warning(
                    f"   [batchQ] embedding miss for cid={cid}; sync fallback"
                )
                cache[name] = self._sync_embed_one(name)

        return cache

    # -------------------------------------------------------------------------
    # Batch K: KG extraction LLM call per merged entity
    # -------------------------------------------------------------------------
    def _batch_kg_extraction(
        self,
        states: Dict[str, DocState],
        query_cache: Dict[str, List[float]],
    ) -> None:
        self.logger.info("BATCH K: Per-entity KG Extraction")
        top_k = self.config.get("retrieval_top_k", 5)

        # For each merged entity in each doc, retrieve top-k similar sentences
        # locally (cheap; just numpy) and prepare the KG-extraction prompt.
        requests: List[BatchRequest] = []
        for st in states.values():
            if not st.merged_entities:
                continue
            sentence_vecs = np.array(st.sentence_vectors) if st.sentences else None
            for entity_id, ent in st.merged_entities.items():
                # Sentences this entity appeared in (from chunkid trail)
                chunk_sentences: List[str] = []
                for cid in ent.get("chunkid", "").split(";;;"):
                    cid = cid.strip()
                    if cid in st.id_to_sentence:
                        chunk_sentences.append(st.id_to_sentence[cid])

                # Top-k retrieval over this doc's sentence vectors
                retrieved: List[str] = []
                name = ent.get("name", "")
                if name in query_cache and sentence_vecs is not None and len(st.sentences):
                    sims = cosine_similarity(
                        [query_cache[name]], sentence_vecs
                    )[0]
                    top_idx = np.argsort(sims)[::-1][:top_k]
                    retrieved = [st.sentences[i] for i in top_idx]

                context = ", ".join(list(set(chunk_sentences + retrieved)))
                requests.append(
                    build_responses_request(
                        custom_id=f"kg|{st.doc_id}|{entity_id}",
                        model=self.config["model"],
                        system_prompt=_ensure_json_word(KG_EXTRACTION_SYSTEM_PROMPT),
                        user_prompt=KG_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                            text=context,
                            target_entity=name,
                            related_kg="none",
                        ),
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                    )
                )

        self.logger.info(f"   {len(requests)} KG-extraction calls")
        if not requests:
            return

        results = self.batch.run_job(
            job_name="batchK_kg_extract",
            endpoint="/v1/responses",
            requests=requests,
        )

        for cid, payload in self._iter_parsed_json_results(
            results, requests, retry_phase_label="batchK"
        ):
            _, doc_id, entity_id = cid.split("|", 2)
            states[doc_id].raw_kg[entity_id] = payload

        for st in states.values():
            self.logger.info(
                f"   {st.doc_id}: KG extracted for "
                f"{len(st.raw_kg)}/{len(st.merged_entities)} entities"
            )

    # -------------------------------------------------------------------------
    # Local: KG conversion + per-doc save
    # -------------------------------------------------------------------------
    def _convert_and_save(
        self, states: Dict[str, DocState], run_t0: float
    ) -> None:
        self.logger.info("PHASE 8 (local): KG conversion + save")

        for st in states.values():
            st.converted_kg = RAKGPipeline._convert_kg(st.raw_kg)
            st.elapsed = time.time() - run_t0  # whole-batch elapsed; no per-doc timer

            entities_refined = [e["name"] for e in st.converted_kg["entities"]]
            triples_final = st.converted_kg["relations"]

            result: Dict[str, Any] = {
                "sentences": st.sentences,
                "entities_raw": [
                    {
                        "name": v.get("name", ""),
                        "type": v.get("type", ""),
                        "description": v.get("description", ""),
                    }
                    for v in st.merged_entities.values()
                ],
                "entities_refined": entities_refined,
                "triples_final": triples_final,
                "converted_kg": st.converted_kg,
                "metadata": {
                    "model": self.config["model"],
                    "embedding_model": self.config["embedding_model"],
                    "num_sentences": len(st.sentences),
                    "num_raw_entities": len(st.raw_entities),
                    "num_merged_entities": len(st.merged_entities),
                    "num_disambiguation_pairs": len(st.confirmed_pairs),
                    "similarity_threshold": self.config.get("similarity_threshold", 0.60),
                    "retrieval_top_k": self.config.get("retrieval_top_k", 5),
                    "time_taken_seconds": round(st.elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                    "batch_mode": True,
                },
            }

            # Reuse RAKGPipeline._save_experiment_data verbatim by spinning up a
            # disposable instance pointed at experiments/<exp>/<doc_id>/.
            doc_dir_name = f"{self.experiment_name}/{st.doc_id}"
            disposable = RAKGPipeline(
                llm_client=self.llm,
                similarity_llm_client=None,
                embedding_client=self._get_sync_embed(),
                similarity_threshold=self.config.get("similarity_threshold", 0.60),
                retrieval_top_k=self.config.get("retrieval_top_k", 5),
                save_experiments=True,
                experiment_name=doc_dir_name,
                logger=self.logger,
            )
            disposable._save_experiment_data(result)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _run_jobs_parallel(
        self, jobs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, BatchResultEntry]]:
        """
        Submit several batch jobs back-to-back, then wait/download/parse each.
        Both jobs run concurrently on OpenAI's side. Resumable: any job whose
        output.jsonl already exists is skipped.
        """
        # Submit phase — returns immediately after batch creation.
        for spec in jobs:
            name = spec["name"]
            output_path = self.batch._output_path(name)
            meta_path = self.batch._meta_path(name)
            if output_path.exists() and meta_path.exists():
                self.logger.info(
                    f"[Batch:{name}] Output already present; will reuse."
                )
                continue
            if not self.batch._input_path(name).exists():
                self.batch.write_input_file(name, spec["requests"])
            self.batch.submit(name, endpoint=spec["endpoint"])

        # Wait + download + parse for each
        results: Dict[str, Dict[str, BatchResultEntry]] = {}
        for spec in jobs:
            name = spec["name"]
            endpoint = spec["endpoint"]
            output_path = self.batch._output_path(name)
            meta_path = self.batch._meta_path(name)
            already = output_path.exists() and meta_path.exists()
            if not already:
                meta: BatchJobMeta = self.batch.wait(name)
                if (
                    meta.status != "completed"
                    and not meta.output_file_id
                    and not meta.error_file_id
                ):
                    raise RuntimeError(
                        f"Batch '{name}' ended in status '{meta.status}' with no output."
                    )
                self.batch.download(name)
            results[name] = self.batch.parse_results(name, endpoint=endpoint)
        return results

    def _iter_parsed_json_results(
        self,
        results: Dict[str, BatchResultEntry],
        requests: Sequence[BatchRequest],
        retry_phase_label: str,
    ):
        """Yield (custom_id, parsed_payload) for each request; sync-retry on failure."""
        for req in requests:
            cid = req.custom_id
            entry = results.get(cid)
            payload: Optional[Dict[str, Any]] = None
            if entry is not None and entry.success and entry.content:
                try:
                    payload = json.loads(
                        LLMClient._clean_json_response(entry.content)
                    )
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
        try:
            body = req.body
            msgs = body.get("input") or body.get("messages") or []
            system = next(
                (m["content"] for m in msgs if m.get("role") == "system"), ""
            )
            user = next(
                (m["content"] for m in msgs if m.get("role") == "user"), ""
            )
            return self.llm.generate_json(
                system_prompt=system,
                user_prompt=user,
                max_output_tokens=body.get("max_output_tokens")
                or body.get("max_tokens"),
            )
        except Exception as exc:
            self.logger.error(f"   Sync retry failed for {req.custom_id}: {exc}")
            return None

    def _get_sync_embed(self) -> EmbeddingClient:
        if self._sync_embed is None:
            self._sync_embed = EmbeddingClient(
                model=self.config["embedding_model"],
                backend=self.config.get("embedding_backend", "openai"),
                logger=self.logger,
            )
        return self._sync_embed

    def _sync_embed_one(self, text: str) -> List[float]:
        return self._get_sync_embed().embed_query(text)

    # -------------------------------------------------------------------------
    # Dry-run reporting
    # -------------------------------------------------------------------------
    def _report_dry_run(self, states: Dict[str, DocState]) -> None:
        n_docs = len(states)
        total_sents = sum(len(st.sentences) for st in states.values())
        unique_sents = len({s for st in states.values() for s in st.sentences})
        lines = [
            "",
            "DRY RUN -- nothing submitted to OpenAI.",
            f"  Docs:                {n_docs}",
            f"  Total sentences:     {total_sents}",
            f"  Unique sentences:    {unique_sents}",
            f"  Batch S requests:    {unique_sents} (sentence embeddings)",
            f"  Batch N requests:    {total_sents} (per-sentence NER)",
            "  Batch E requests:    depends on raw entities (after Batch N)",
            "  Batch D requests:    depends on candidate pairs (after Batch E)",
            "  Batch Q requests:    depends on merged entities",
            "  Batch K requests:    depends on merged entities",
            f"  Input files written under: {self.batch_workdir}",
        ]
        for line in lines:
            self.logger.info(line)
