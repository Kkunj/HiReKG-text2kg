"""
Append SciERC NER entities (excluding 'Generic' type) to the stage3_final
atomic-fact JSON files for the 100 sampled documents.

Pipeline
--------
1. Load combined.json (JSONL, 500 docs) from temp/ground_truth_entity/scierc/.
2. For each of the 100 sampled doc_<i>.txt files under datasets/scierc/texts/,
   match it to its source doc in combined.json via normalized-text comparison
   (handles PTB-style brackets like -LRB-/-RRB- and tokenization whitespace).
   Falls back to a probe-overlap fuzzy match for edge cases.
3. From the matched combined doc, extract every NER span whose type is NOT
   'Generic'. NER format in combined.json: [start_tok, end_tok, type] with
   token indices that are global across the document's flattened sentences.
   The entity surface form is recovered by slicing the flattened token list
   (after applying the PTB->punctuation map) and joining with a space.
4. Deduplicate entities while preserving first-seen order, then write them
   as a new `entities` field on the corresponding stage3_final JSON.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCIERC = ROOT / "datasets" / "scierc"
COMBINED_PATH = SCIERC / "ground_truth_entity" / "combined.json"
TEXTS_DIR = SCIERC / "texts"
STAGE_DIR = SCIERC / "atomic_facts" / "stage3_final"
NUM_DOCS = 100
EXCLUDED_NER_TYPES = {"Generic"}

PTB_MAP = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
    "``": '"', "''": '"', "`": "'",
}


def load_combined(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def flat_tokens(doc: dict) -> list[str]:
    return [PTB_MAP.get(t, t) for s in doc["sentences"] for t in s]


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_doc_mapping(combined: list[dict], texts_dir: Path, n: int) -> dict[int, dict]:
    """Map doc_<i> index -> combined doc record."""
    combined_norm = [(d, normalize(" ".join(flat_tokens(d)))) for d in combined]
    prefix_len = 250

    mapping: dict[int, dict] = {}
    fuzzy_fallbacks: list[tuple[int, str, int]] = []

    for i in range(n):
        txt_path = texts_dir / f"doc_{i}.txt"
        txt_norm = normalize(txt_path.read_text(encoding="utf-8"))

        # Exact prefix match
        match = None
        for d, dn in combined_norm:
            if dn[:prefix_len] == txt_norm[:prefix_len] and len(txt_norm) >= 100:
                match = d
                break

        # Probe-overlap fuzzy fallback
        if match is None:
            probes = [txt_norm[off:off + 80] for off in (0, 50, 100, 150) if len(txt_norm) > off + 80]
            best, best_score = None, 0
            for d, dn in combined_norm:
                score = sum(1 for p in probes if p in dn)
                if score > best_score:
                    best_score = score
                    best = d
            if best is not None and best_score >= 2:
                match = best
                fuzzy_fallbacks.append((i, best["doc_key"], best_score))

        if match is None:
            raise RuntimeError(f"Could not match doc_{i}.txt to any combined entry")
        mapping[i] = match

    if fuzzy_fallbacks:
        print(f"[info] {len(fuzzy_fallbacks)} doc(s) matched via fuzzy fallback:")
        for i, key, score in fuzzy_fallbacks:
            print(f"  doc_{i} -> {key} (probe-score {score}/4)")

    # Sanity check: unique mapping
    seen_keys: set[str] = set()
    for i, d in mapping.items():
        if d["doc_key"] in seen_keys:
            raise RuntimeError(f"Duplicate doc_key mapping detected at doc_{i}: {d['doc_key']}")
        seen_keys.add(d["doc_key"])
    return mapping


def extract_entities(doc: dict) -> list[str]:
    """Return ordered, deduplicated list of non-Generic NER entity surfaces."""
    tokens = flat_tokens(doc)
    entities: list[str] = []
    seen: set[str] = set()
    for sent_ner in doc.get("ner", []):
        for start, end, ner_type in sent_ner:
            if ner_type in EXCLUDED_NER_TYPES:
                continue
            # end index is inclusive in SciERC
            surface = " ".join(tokens[start:end + 1])
            surface = surface.strip()
            if not surface or surface in seen:
                continue
            seen.add(surface)
            entities.append(surface)
    return entities


def main() -> None:
    combined = load_combined(COMBINED_PATH)
    print(f"Loaded {len(combined)} docs from {COMBINED_PATH.name}")

    mapping = build_doc_mapping(combined, TEXTS_DIR, NUM_DOCS)
    print(f"Mapped {len(mapping)}/{NUM_DOCS} sampled docs to combined entries")

    total_entities = 0
    for i in range(NUM_DOCS):
        combined_doc = mapping[i]
        entities = extract_entities(combined_doc)
        total_entities += len(entities)

        stage_path = STAGE_DIR / f"doc_{i}.json"
        with stage_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["entities"] = entities
        with stage_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"doc_{i} <- {combined_doc['doc_key']}: {len(entities)} entities")

    print(f"\nDone. Wrote {total_entities} entities across {NUM_DOCS} docs.")


if __name__ == "__main__":
    main()
