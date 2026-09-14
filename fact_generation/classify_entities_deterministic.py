"""
Deterministically reclassify each atomic fact as 'single-entity',
'multiple-entity', or 'ambiguous' based on how many entities from the
doc's `entities` list appear in the fact.

Rules (chosen to maximize the single-entity count)
--------------------------------------------------
- Matching is case-insensitive.
- Whitespace and spacing around punctuation are normalized on both sides
  before matching (so "( GMM )" in the entity list still matches "(GMM)"
  in the fact text).
- Matches must be bordered by non-alphanumeric characters (poor-man's
  word boundary that also works around punctuation), so the entity
  "Vivo" does NOT match inside "VivoTab".
- Overlapping matches are deduplicated greedily: when two entity matches
  cover overlapping spans in the fact, only the longer one is kept.
  This prevents "Windows" + "Microsoft Windows" from being counted as
  two distinct matches.

Output
------
For each doc JSON in the configured stage3_final directories, the keys
'single-entity', 'multiple-entity', and 'ambiguous' are OVERWRITTEN with
lists of fact indices. All other fields are preserved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"C:\<PROJECT_ROOT>\graph_rag")
STAGE_DIRS = [
    ROOT / "datasets" / "scierc" / "atomic_facts" / "stage3_final",
    ROOT / "datasets" / "windows_redocred" / "atomic_facts" / "stage3_final",
]


def normalize_for_match(s: str) -> str:
    """Lowercase, collapse whitespace, strip spaces around non-alphanumeric chars."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+(?=[^a-z0-9\s])", "", s)
    s = re.sub(r"(?<=[^a-z0-9\s])\s+", "", s)
    return s


def find_entity_spans(entity: str, fact_norm: str) -> list[tuple[int, int]]:
    """All non-overlapping spans of `entity` in normalized fact, with word-ish boundaries."""
    ent_norm = normalize_for_match(entity)
    if not ent_norm:
        return []
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(ent_norm)}(?![a-z0-9])")
    return [(m.start(), m.end()) for m in pattern.finditer(fact_norm)]


def count_distinct_entities(fact: str, entities: list[str]) -> int:
    """Greedy longest non-overlapping match count."""
    fact_norm = normalize_for_match(fact)

    candidates: list[tuple[int, int]] = []
    for ent in entities:
        candidates.extend(find_entity_spans(ent, fact_norm))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen: list[tuple[int, int]] = []
    for start, end in candidates:
        if all(end <= cs or start >= ce for cs, ce in chosen):
            chosen.append((start, end))
    return len(chosen)


def classify_doc(payload: dict) -> None:
    facts = payload.get("facts", [])
    entities = payload.get("entities", [])

    single: list[int] = []
    multiple: list[int] = []
    ambiguous: list[int] = []

    for idx, fact in enumerate(facts):
        n = count_distinct_entities(fact, entities)
        if n == 0:
            ambiguous.append(idx)
        elif n == 1:
            single.append(idx)
        else:
            multiple.append(idx)

    payload["single-entity"] = single
    payload["multiple-entity"] = multiple
    payload["ambiguous"] = ambiguous


def process_dir(stage_dir: Path) -> tuple[int, int, int, int]:
    files = sorted(stage_dir.glob("*.json"))
    n_single = n_multi = n_amb = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "entities" not in payload:
            print(f"  [warn] {path.name}: no 'entities' field, skipping")
            continue
        classify_doc(payload)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        n_single += len(payload["single-entity"])
        n_multi += len(payload["multiple-entity"])
        n_amb += len(payload["ambiguous"])
    return len(files), n_single, n_multi, n_amb


def main() -> None:
    for stage_dir in STAGE_DIRS:
        print(f"\n=== {stage_dir} ===")
        n_files, n_single, n_multi, n_amb = process_dir(stage_dir)
        total = n_single + n_multi + n_amb
        print(
            f"  files: {n_files} | total facts: {total} | "
            f"single: {n_single} | multiple: {n_multi} | ambiguous: {n_amb}"
        )


if __name__ == "__main__":
    main()
