"""
Adapter: convert KGGen's pipeline output (final_output.json) into the
{entities, edges, relations} dict shape that kg_gen.KGGen.from_dict expects.

KGGen pipeline output (baselines/kggen/pipeline.py):
    final_output["triples_final"]  = [
        {"subject": s, "relation": r, "object": o}, ...
    ]
    final_output["entities_refined"] = [e1, e2, ...]   # informational only
    final_output["edges"]            = [...]           # informational only

The triple schema matches OURS / RAKG, so this adapter is structurally
identical to RAKG/kg_adapter.py.

kg-gen format:
    {
        "entities":  [flat list of unique node names],
        "edges":     [flat list of unique relation labels],
        "relations": [[subj, rel, obj], ...]
    }

Notes:
    * kg-gen's `entities` must include EVERY node that appears in any relation
      (both subjects and objects), not just the refined-entities list.
    * Empty / whitespace-only strings are filtered out.
    * Duplicates are removed while preserving first-seen order for determinism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _clean(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _unique_preserve_order(items) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def triples_to_kggen_dict(triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Core conversion: KGGen triple dicts -> kg-gen dict format."""
    relations_tuples: List[List[str]] = []
    node_candidates: List[str] = []
    edge_candidates: List[str] = []

    for t in triples:
        subj = _clean(t.get("subject"))
        rel = _clean(t.get("relation"))
        obj = _clean(t.get("object"))
        if not (subj and rel and obj):
            continue
        relations_tuples.append([subj, rel, obj])
        node_candidates.append(subj)
        node_candidates.append(obj)
        edge_candidates.append(rel)

    return {
        "entities": _unique_preserve_order(node_candidates),
        "edges": _unique_preserve_order(edge_candidates),
        "relations": relations_tuples,
    }


def load_kg_for_doc(experiments_root: Path, doc_id: str) -> Dict[str, Any]:
    """
    Load KGGen's per-doc final_output.json and return the kg-gen dict.

    Args:
        experiments_root: path to .../baselines/kggen/experiments/<run>
        doc_id:           e.g. "doc_46"
    """
    final_path = Path(experiments_root) / doc_id / "final_output.json"
    if not final_path.exists():
        raise FileNotFoundError(
            f"Missing {final_path} -- was the KGGen pipeline run for {doc_id}?"
        )
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    triples = payload.get("triples_final", []) or []
    return triples_to_kggen_dict(triples)
