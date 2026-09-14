"""
Stage 2 — Per-document context retrieval.

For short documents (< 3000 words) we hand the judge the full source text:
the judge's window is large enough, and full-text removes any retrieval-error
confound from the faithfulness score.

For longer documents we chunk into ~300-word paragraph-aligned windows with
50-word overlap, embed each chunk once with all-MiniLM-L6-v2 (cached on disk
keyed by doc_id + a config hash), embed the verbalized statement, and pick
top-k chunks by cosine similarity. The retrieved chunks are concatenated in
document order so the judge sees them in their original narrative sequence.

Why these choices:
  * 300-word chunks: large enough that a self-contained fact rarely spans
    multiple chunks but small enough that embedding similarity is meaningful.
  * 50-word overlap: protects against facts that straddle chunk boundaries.
  * MiniLM (all-MiniLM-L6-v2): the same retrieval encoder used by MINE recall,
    so retrieval errors are comparable across the precision and recall sides
    of the evaluation.
  * Cache per (doc_id, chunker_config_hash): we re-embed chunks once per doc,
    not once per triple — ~85x speedup at typical KG sizes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


CHUNK_WORD_LIMIT = 300
CHUNK_OVERLAP_WORDS = 50
SHORT_DOC_WORD_THRESHOLD = 3000
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: int
    text: str


@dataclass
class RetrievedContext:
    doc_id: str
    statement_id: str          # the triple's verbalize custom_id
    used_full_text: bool
    chunk_ids: List[int]       # empty if used_full_text
    context_text: str


def _word_count(s: str) -> int:
    return len(s.split())


def _split_paragraph_aligned(
    text: str, max_words: int = CHUNK_WORD_LIMIT, overlap: int = CHUNK_OVERLAP_WORDS
) -> List[Chunk]:
    """
    Greedy paragraph-aligned chunker. Walks paragraphs in order, packing them
    into a chunk until the next paragraph would exceed `max_words`. If a single
    paragraph already exceeds max_words, it is split on word boundaries with
    `overlap` carry-forward.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    buf_words: List[str] = []
    buf_word_count = 0

    def flush():
        nonlocal buf_words, buf_word_count
        if buf_words:
            chunks.append(Chunk(chunk_id=len(chunks), text=" ".join(buf_words)))
            buf_words = []
            buf_word_count = 0

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            # Flush whatever we have, then window-split the long paragraph.
            flush()
            i = 0
            while i < len(words):
                end = min(i + max_words, len(words))
                chunks.append(
                    Chunk(chunk_id=len(chunks), text=" ".join(words[i:end]))
                )
                if end == len(words):
                    break
                i = max(i + max_words - overlap, i + 1)
            continue

        if buf_word_count + len(words) > max_words and buf_words:
            flush()
            # Carry forward last `overlap` words of previous chunk for continuity
            if chunks and overlap > 0:
                tail = chunks[-1].text.split()[-overlap:]
                buf_words.extend(tail)
                buf_word_count = len(tail)

        buf_words.extend(words)
        buf_word_count += len(words)

    flush()
    return chunks


def _config_hash(model_name: str) -> str:
    payload = f"{model_name}|{CHUNK_WORD_LIMIT}|{CHUNK_OVERLAP_WORDS}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


class _EncoderHolder:
    """Lazy singleton so we only load the model once per process."""
    _model = None

    @classmethod
    def get(cls, model_name: str):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer  # local import: heavy
            cls._model = SentenceTransformer(model_name)
        return cls._model


def _encode(texts: List[str], model_name: str) -> np.ndarray:
    model = _EncoderHolder.get(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _load_or_build_chunk_index(
    cache_dir: Path,
    doc_id: str,
    source_text: str,
    model_name: str,
    logger: logging.Logger,
) -> Tuple[List[Chunk], np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _config_hash(model_name)
    cache_path = cache_dir / f"{doc_id}__{h}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        chunks = [
            Chunk(chunk_id=int(cid), text=str(t))
            for cid, t in zip(data["chunk_ids"], data["texts"])
        ]
        embs = data["embeddings"]
        return chunks, embs

    chunks = _split_paragraph_aligned(source_text)
    if not chunks:
        # Defensive: empty source text -> single empty chunk so calling code
        # has something to return; will result in NOT_STATED for everything.
        chunks = [Chunk(chunk_id=0, text="")]
    embs = _encode([c.text for c in chunks], model_name)

    np.savez(
        cache_path,
        chunk_ids=np.array([c.chunk_id for c in chunks], dtype=np.int32),
        texts=np.array([c.text for c in chunks], dtype=object),
        embeddings=embs,
    )
    logger.debug(f"[retrieve] built chunk index for {doc_id}: {len(chunks)} chunks")
    return chunks, embs


def retrieve_context(
    doc_id: str,
    source_text: str,
    statement_id: str,
    statement: str,
    cache_dir: Path,
    logger: logging.Logger,
    model_name: str = DEFAULT_RETRIEVAL_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievedContext:
    """
    Return the context the judge should see for `statement` against
    document `doc_id`. Uses full text for short documents.
    """
    if _word_count(source_text) < SHORT_DOC_WORD_THRESHOLD:
        return RetrievedContext(
            doc_id=doc_id,
            statement_id=statement_id,
            used_full_text=True,
            chunk_ids=[],
            context_text=source_text.strip(),
        )

    chunks, chunk_embs = _load_or_build_chunk_index(
        cache_dir=cache_dir,
        doc_id=doc_id,
        source_text=source_text,
        model_name=model_name,
        logger=logger,
    )
    stmt_emb = _encode([statement], model_name)[0]
    sims = chunk_embs @ stmt_emb            # cosine since both are L2-normalized
    k = min(top_k, len(chunks))
    top_idx = np.argsort(-sims)[:k]
    top_idx_sorted = sorted(top_idx.tolist())   # restore document order

    selected = [chunks[i] for i in top_idx_sorted]
    context_text = "\n\n".join(c.text for c in selected)
    return RetrievedContext(
        doc_id=doc_id,
        statement_id=statement_id,
        used_full_text=False,
        chunk_ids=[c.chunk_id for c in selected],
        context_text=context_text,
    )


def retrieve_for_all_statements(
    statements_by_cid: Dict[str, str],          # {verb_cid -> statement}
    triple_doc_lookup: Dict[str, str],          # {verb_cid -> doc_id}
    source_texts: Dict[str, str],               # {doc_id -> source_text}
    cache_dir: Path,
    logger: logging.Logger,
    model_name: str = DEFAULT_RETRIEVAL_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, RetrievedContext]:
    """
    Retrieve context for every (statement, doc) pair. Embeddings are cached
    per doc_id so the encoder runs once per document, not once per triple.
    """
    out: Dict[str, RetrievedContext] = {}
    for cid, statement in statements_by_cid.items():
        doc_id = triple_doc_lookup[cid]
        src = source_texts.get(doc_id, "")
        out[cid] = retrieve_context(
            doc_id=doc_id,
            source_text=src,
            statement_id=cid,
            statement=statement,
            cache_dir=cache_dir,
            logger=logger,
            model_name=model_name,
            top_k=top_k,
        )
    n_full = sum(1 for r in out.values() if r.used_full_text)
    logger.info(
        f"[retrieve] retrieved_for={len(out)} stmts "
        f"(full_text={n_full}, top_k={len(out) - n_full})"
    )
    return out
