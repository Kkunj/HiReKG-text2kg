"""
Text Processing for Knowledge Graph Creation Pipeline. Includes text normalization and chunking based on sentences.
"""


import re
from functools import lru_cache
from typing import List

import spacy
from spacy.language import Language


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


@lru_cache(maxsize=1)
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


