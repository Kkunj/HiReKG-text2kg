"""Verify the GPT-5 atomic-fact entity classification using spaCy.

The GPT-5 labels in datasets/MINE/answers.json place each atomic fact in one of:
    - single-entity     (exactly one entity mentioned)
    - multiple-entity   (two or more entities mentioned)
    - ambiguous         (unclear)

This script re-classifies every atomic fact with spaCy and reports the
inter-annotator agreement (GPT-5 vs spaCy) using two entity-detection
strategies:

    NER     : spaCy named-entity recognition (proper-noun centric).
    CHUNKS  : distinct noun-chunk heads (lemmatised, lowercased, stop-words
              removed) -- approximates a broader notion of "entity" that
              includes common nouns like "butterfly" or "caterpillar".

Counts are mapped to labels as:
    0 distinct entities -> ambiguous
    1 distinct entity   -> single-entity
    >=2 distinct        -> multiple-entity
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import spacy

ANSWERS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "MINE" / "answers.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "mine_spacy_agreement.json"

LABELS = ("single-entity", "multiple-entity", "ambiguous")
BINARY_LABELS = ("single-entity", "multiple-entity")


def collapse(label: str) -> str:
    """Collapse the 3-way label to binary: ambiguous -> multiple-entity."""
    return "single-entity" if label == "single-entity" else "multiple-entity"


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gpt_labels_for_row(row: dict) -> dict[int, str]:
    """Map answer-index -> GPT-5 label for one row."""
    out: dict[int, str] = {}
    for label in LABELS:
        for idx in row.get(label, []):
            out[idx] = label
    return out


def count_to_label(n: int) -> str:
    """Binary mapping: 1 distinct entity -> single, otherwise multi
    (0 entities folds into multi, matching how we collapse GPT's ambiguous)."""
    return "single-entity" if n == 1 else "multiple-entity"


def ner_entities(doc) -> list[str]:
    seen: list[str] = []
    for ent in doc.ents:
        key = ent.text.strip().lower()
        if key and key not in seen:
            seen.append(key)
    return seen


def chunk_entities(doc) -> list[str]:
    """Distinct noun-chunk heads, lemmatised + stop-word filtered."""
    seen: list[str] = []
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.is_stop or head.is_punct or head.like_num:
            continue
        if head.pos_ == "PRON":
            continue
        key = head.lemma_.lower().strip()
        if not key or key in seen:
            continue
        seen.append(key)
    return seen


def cohen_kappa(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    pe = sum((a_counts.get(l, 0) / n) * (b_counts.get(l, 0) / n) for l in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(pairs, labels):
    cm = {gpt: {sp: 0 for sp in labels} for gpt in labels}
    for gpt, sp in pairs:
        cm[gpt][sp] += 1
    return cm


def print_cm(name: str, cm: dict, labels: tuple[str, ...]) -> None:
    col_w = max(len(l) for l in labels) + 2
    header = " " * (col_w + 4) + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n=== Confusion matrix [{name}]  (rows = GPT-5, cols = spaCy) ===")
    print(header)
    for gpt in labels:
        row = f"  {gpt:<{col_w + 2}}" + "".join(f"{cm[gpt][sp]:>{col_w}}" for sp in labels)
        print(row)


def summarise(name: str, pairs: list[tuple[str, str]]) -> dict:
    """Binary agreement (single-entity vs multiple-entity).

    pairs are already binary -- ambiguous has been collapsed into
    multiple-entity on the GPT-5 side.
    """
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    acc = agree / n if n else float("nan")
    kappa = cohen_kappa(pairs, BINARY_LABELS)

    # Per-class precision / recall, treating each label as positive in turn.
    per_class = {}
    for cls in BINARY_LABELS:
        tp = sum(1 for a, b in pairs if a == cls and b == cls)
        fp = sum(1 for a, b in pairs if a != cls and b == cls)
        fn = sum(1 for a, b in pairs if a == cls and b != cls)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        per_class[cls] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    cm = confusion_matrix(pairs, BINARY_LABELS)
    print(f"\n--- {name} ---")
    print(f"  Total facts compared : {n}")
    print(f"  Agreement rate       : {acc:.4f}  ({agree}/{n})")
    print(f"  Cohen's kappa        : {kappa:.4f}")
    print_cm(name, cm, BINARY_LABELS)
    for cls, m in per_class.items():
        print(f"  {cls:>16} : precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")

    sp_counts = Counter(b for _, b in pairs)
    gpt_counts = Counter(a for a, _ in pairs)
    print(f"  GPT-5 label counts   : {dict(gpt_counts)}")
    print(f"  spaCy label counts   : {dict(sp_counts)}")

    return {
        "method": name,
        "total": n,
        "agreement": acc,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "per_class": per_class,
        "gpt5_counts": dict(gpt_counts),
        "spacy_counts": dict(sp_counts),
    }


def main() -> None:
    print(f"Loading {ANSWERS_PATH} ...")
    data = load_dataset(ANSWERS_PATH)
    rows = data["rows"]
    print(f"Loaded {len(rows)} rows.")

    print("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    ner_pairs: list[tuple[str, str]] = []
    chunk_pairs: list[tuple[str, str]] = []

    flat_facts: list[tuple[int, int, str]] = []  # (row_idx, ans_idx, text)
    flat_gpt: list[str] = []
    for r in rows:
        row_idx = r["row_idx"]
        row = r["row"]
        labels_by_idx = gpt_labels_for_row(row)
        for i, ans in enumerate(row["answers"]):
            if i not in labels_by_idx:
                continue
            flat_facts.append((row_idx, i, ans["answer"]))
            # Collapse 3-class label to binary: ambiguous -> multiple-entity.
            flat_gpt.append(collapse(labels_by_idx[i]))

    print(f"Running spaCy over {len(flat_facts)} atomic facts ...")
    texts = [t for _, _, t in flat_facts]
    examples = []  # collect a handful of disagreements for the JSON report
    disagree_kept = 0
    for (row_idx, ans_idx, text), gpt, doc in zip(flat_facts, flat_gpt, nlp.pipe(texts, batch_size=64)):
        ner_count = len(ner_entities(doc))
        chunk_count = len(chunk_entities(doc))
        ner_label = count_to_label(ner_count)
        chunk_label = count_to_label(chunk_count)
        ner_pairs.append((gpt, ner_label))
        chunk_pairs.append((gpt, chunk_label))

        if ner_label != gpt and disagree_kept < 25:
            examples.append({
                "row_idx": row_idx,
                "ans_idx": ans_idx,
                "text": text,
                "gpt5_collapsed": gpt,
                "spacy_ner_label": ner_label,
                "spacy_chunk_label": chunk_label,
                "ner_entities": ner_entities(doc),
                "chunk_entities": chunk_entities(doc),
            })
            disagree_kept += 1

    ner_summary = summarise("spaCy NER (named entities only)", ner_pairs)
    chunk_summary = summarise("spaCy noun-chunk heads (broad entities)", chunk_pairs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "answers_file": str(ANSWERS_PATH),
        "model": "en_core_web_sm",
        "n_facts": len(flat_facts),
        "ner": ner_summary,
        "chunks": chunk_summary,
        "sample_disagreements_chunks": examples,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
