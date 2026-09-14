"""
Orchestrator — runs the four faithfulness stages end-to-end.

Stages:
    1. verbalize  (batch, /v1/responses, gpt-5-mini)
    2. retrieve   (local, MiniLM, cached per doc_id)
    3. judge      (batch, /v1/responses, gpt-5)
    4. assemble   (local, write per-doc audit + dataset summary)

The pipeline is resumable at every stage: each batch lives under
`workdir/<job_name>/` and is reused if its output.jsonl is already present.
Per-doc audit files are overwritten on each run of stage 4.

The entry point is `run_pipeline()`, which takes already-loaded inputs:
    * triples_per_doc:   {doc_id -> List[ {subject, relation, object} ]}
    * source_texts:      {doc_id -> str}
This keeps method-specific input loading (OURS, RAKG, etc.) out of pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .aggregate import (
    DatasetSummary,
    PerDocSummary,
    PerTripleAudit,
    audit_to_dict,
    dataset_to_dict,
    per_doc_to_dict,
    summarize_dataset,
    summarize_doc,
)
from .judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING,
    run_judge_stage,
)
from .retrieve import (
    DEFAULT_RETRIEVAL_MODEL,
    DEFAULT_TOP_K,
    retrieve_for_all_statements,
)
from .verbalize import (
    DEFAULT_VERBALIZE_MODEL,
    DEFAULT_VERBALIZE_REASONING,
    TripleRow,
    run_verbalize_stage,
    template_fallback,
)


def _build_triple_rows(
    triples_per_doc: Dict[str, List[Dict]],
) -> List[TripleRow]:
    rows: List[TripleRow] = []
    for doc_id, triples in triples_per_doc.items():
        for i, t in enumerate(triples):
            # Tolerate both (subject, predicate, object) and
            # (subject, relation, object) keyings.
            pred = t.get("predicate") if "predicate" in t else t.get("relation")
            rows.append(
                TripleRow(
                    doc_id=doc_id,
                    triple_idx=i,
                    subject=str(t.get("subject", "") or ""),
                    predicate=str(pred or ""),
                    object=str(t.get("object", "") or ""),
                )
            )
    return rows


def make_logger(workdir: Path, name: str = "Faithfulness") -> logging.Logger:
    import sys

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    workdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        workdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    logger.addHandler(ch)
    return logger


def run_pipeline(
    triples_per_doc: Dict[str, List[Dict]],
    source_texts: Dict[str, str],
    prompts_dir: Path,
    workdir: Path,
    results_dir: Path,
    cache_dir: Path,
    logger: Optional[logging.Logger] = None,
    verbalize_model: str = DEFAULT_VERBALIZE_MODEL,
    verbalize_reasoning: str = DEFAULT_VERBALIZE_REASONING,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_reasoning: str = DEFAULT_JUDGE_REASONING,
    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL,
    top_k: int = DEFAULT_TOP_K,
    use_template_for_verbalize_failures: bool = True,
    backend: str = "openai",
    local_max_workers: int = 8,
) -> Tuple[List[PerTripleAudit], List[PerDocSummary], DatasetSummary]:
    """
    End-to-end faithfulness scoring. Writes:
        results_dir/audit_<doc_id>.json    per-triple audit log
        results_dir/per_doc_summary.json   per-doc metrics
        results_dir/dataset_summary.json   macro mean / std + micro
    Returns the same artifacts in-memory.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or make_logger(workdir)

    # -------- inputs --------
    rows = _build_triple_rows(triples_per_doc)
    logger.info(
        f"[pipeline] docs={len(triples_per_doc)} total_triples={len(rows)}"
    )

    # -------- stage 1: verbalize --------
    statements_by_cid, malformed_rows = run_verbalize_stage(
        triples=rows,
        prompts_dir=prompts_dir,
        workdir=workdir,
        logger=logger,
        model=verbalize_model,
        reasoning_effort=verbalize_reasoning,
        backend=backend,
        local_max_workers=local_max_workers,
    )

    # Build verbalizer-failure fallbacks. Some triples may have a verb_cid in
    # `rows` but no entry in statements_by_cid (LLM/parse failure). Per the
    # plan, we don't want to silently drop them — apply a deterministic
    # template so they still reach the judge. The audit log records that the
    # fallback was used.
    cid_to_row: Dict[str, TripleRow] = {r.custom_id: r for r in rows}
    used_template_for: set = set()
    if use_template_for_verbalize_failures:
        for r in rows:
            if r.custom_id in statements_by_cid:
                continue
            if r in malformed_rows:
                continue
            statements_by_cid[r.custom_id] = template_fallback(r)
            used_template_for.add(r.custom_id)
        if used_template_for:
            logger.warning(
                f"[verbalize] used template fallback for {len(used_template_for)} triples"
            )

    # -------- stage 2: retrieve --------
    triple_doc_lookup = {cid: cid_to_row[cid].doc_id for cid in statements_by_cid}
    retrieved = retrieve_for_all_statements(
        statements_by_cid=statements_by_cid,
        triple_doc_lookup=triple_doc_lookup,
        source_texts=source_texts,
        cache_dir=cache_dir,
        logger=logger,
        model_name=retrieval_model,
        top_k=top_k,
    )

    # -------- stage 3: judge --------
    statement_to_context = {cid: retrieved[cid].context_text for cid in statements_by_cid}
    judge_outcomes = run_judge_stage(
        statement_to_context=statement_to_context,
        statement_to_text=statements_by_cid,
        prompts_dir=prompts_dir,
        workdir=workdir,
        logger=logger,
        model=judge_model,
        reasoning_effort=judge_reasoning,
        backend=backend,
        local_max_workers=local_max_workers,
    )

    # -------- stage 4: assemble --------
    audits: List[PerTripleAudit] = []
    audits_by_doc: Dict[str, List[PerTripleAudit]] = {
        d: [] for d in triples_per_doc
    }

    for r in rows:
        # MALFORMED triples never made it to verbalize / judge
        if r in malformed_rows:
            audit = PerTripleAudit(
                doc_id=r.doc_id,
                triple_idx=r.triple_idx,
                triple=[r.subject, r.predicate, r.object],
                statement=None,
                verdict="MALFORMED",
            )
            audits.append(audit)
            audits_by_doc[r.doc_id].append(audit)
            continue

        statement = statements_by_cid.get(r.custom_id)
        retrieved_ctx = retrieved.get(r.custom_id)
        judge_cid = "judge|" + r.custom_id.split("|", 1)[1]
        outcome = judge_outcomes.get(judge_cid)

        verdict: str
        judge_error: Optional[str] = None
        if outcome is None or outcome.verdict is None:
            verdict = "JUDGE_ERROR"
            judge_error = (outcome.error if outcome else None) or "no_outcome"
        else:
            verdict = outcome.verdict

        audit = PerTripleAudit(
            doc_id=r.doc_id,
            triple_idx=r.triple_idx,
            triple=[r.subject, r.predicate, r.object],
            statement=statement,
            used_full_text=retrieved_ctx.used_full_text if retrieved_ctx else None,
            retrieved_chunk_ids=retrieved_ctx.chunk_ids if retrieved_ctx else [],
            supporting_span=outcome.supporting_span if outcome else None,
            reasoning=outcome.reasoning if outcome else None,
            verdict=verdict,
            judge_error=judge_error,
        )
        audits.append(audit)
        audits_by_doc[r.doc_id].append(audit)

    # Write per-doc audits + summaries
    per_doc_summaries: List[PerDocSummary] = []
    for doc_id, doc_audits in audits_by_doc.items():
        out_path = results_dir / f"audit_{doc_id}.json"
        out_path.write_text(
            json.dumps([audit_to_dict(a) for a in doc_audits], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_doc_summaries.append(summarize_doc(doc_id, doc_audits))

    dataset_summary = summarize_dataset(per_doc_summaries)

    (results_dir / "per_doc_summary.json").write_text(
        json.dumps(
            [per_doc_to_dict(d) for d in per_doc_summaries],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (results_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_to_dict(dataset_summary), indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info(
        f"[dataset] faithfulness        mean={dataset_summary.faithfulness_mean:.4f} "
        f"std={dataset_summary.faithfulness_std:.4f}"
    )
    logger.info(
        f"[dataset] not_supported_rate  mean={dataset_summary.not_supported_rate_mean:.4f} "
        f"std={dataset_summary.not_supported_rate_std:.4f}"
    )
    logger.info(
        f"[dataset] micro_faithfulness={dataset_summary.overall_micro_faithfulness:.4f} "
        f"(supported={dataset_summary.n_supported_total} / "
        f"evaluated={dataset_summary.n_evaluated_total})"
    )
    logger.info(
        f"[dataset] errors: malformed={dataset_summary.n_malformed_total} "
        f"judge_errors={dataset_summary.n_judge_errors_total}"
    )
    logger.info("=" * 60)

    return audits, per_doc_summaries, dataset_summary
