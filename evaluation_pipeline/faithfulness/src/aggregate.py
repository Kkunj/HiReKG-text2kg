"""
Stage 4 — Aggregate per-triple verdicts into per-doc and per-dataset metrics.

Verdict scheme is binary: SUPPORTED vs NOT_SUPPORTED. Plus the two error
labels MALFORMED and JUDGE_ERROR which are excluded from the denominator.

Per-document scores (the unit reported in the paper):
    n_total          = number of predicted triples for this doc
    n_malformed      = triples with empty subject/predicate/object (skipped pre-flight)
    n_judge_errors   = triples whose judge call failed parse / API after one retry
    n_evaluated      = n_total - n_malformed - n_judge_errors
    n_supported, n_not_supported = counts of each verdict

    faithfulness        = n_supported     / max(n_evaluated, 1)
    not_supported_rate  = n_not_supported / max(n_evaluated, 1)

Per-dataset scores:
    faithfulness_mean,  faithfulness_std        — across documents (macro)
    not_supported_rate_mean / std

Why macro (mean of per-doc) and not micro (n_supported_total / n_evaluated_total):
  * micro biases toward documents with larger KGs (more triples = more weight).
    A noisier method that extracts 5x more triples per doc would dominate the
    score regardless of its per-doc faithfulness. Macro treats each document
    as one observation, which is the unit researchers actually compare.
  * std across documents is also a meaningful confidence signal: a method
    that is faithful on average but high-variance is a different artifact
    from one that is uniformly faithful.

For total transparency we also log the micro figure under
`overall_micro_faithfulness` so reviewers can see both numbers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


@dataclass
class PerTripleAudit:
    doc_id: str
    triple_idx: int
    triple: List[str]            # [subject, predicate, object]
    statement: Optional[str]     # None if MALFORMED
    used_full_text: Optional[bool] = None
    retrieved_chunk_ids: List[int] = field(default_factory=list)
    supporting_span: Optional[str] = None
    reasoning: Optional[str] = None
    verdict: Optional[str] = None  # SUPPORTED | NOT_SUPPORTED | MALFORMED | JUDGE_ERROR
    judge_error: Optional[str] = None


@dataclass
class PerDocSummary:
    doc_id: str
    n_total: int
    n_malformed: int
    n_judge_errors: int
    n_evaluated: int
    n_supported: int
    n_not_supported: int
    faithfulness: float
    not_supported_rate: float


@dataclass
class DatasetSummary:
    n_docs: int
    n_triples_total: int
    n_malformed_total: int
    n_judge_errors_total: int
    n_evaluated_total: int
    n_supported_total: int
    n_not_supported_total: int
    overall_micro_faithfulness: float
    faithfulness_mean: float
    faithfulness_std: float
    not_supported_rate_mean: float
    not_supported_rate_std: float


def _safe_div(num: float, den: int) -> float:
    return num / den if den > 0 else 0.0


def _std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_doc(doc_id: str, audits: List[PerTripleAudit]) -> PerDocSummary:
    n_total = len(audits)
    n_malformed = sum(1 for a in audits if a.verdict == "MALFORMED")
    n_judge_errors = sum(1 for a in audits if a.verdict == "JUDGE_ERROR")
    n_evaluated = n_total - n_malformed - n_judge_errors
    n_supported = sum(1 for a in audits if a.verdict == "SUPPORTED")
    n_not_supported = sum(1 for a in audits if a.verdict == "NOT_SUPPORTED")

    return PerDocSummary(
        doc_id=doc_id,
        n_total=n_total,
        n_malformed=n_malformed,
        n_judge_errors=n_judge_errors,
        n_evaluated=n_evaluated,
        n_supported=n_supported,
        n_not_supported=n_not_supported,
        faithfulness=_safe_div(n_supported, n_evaluated),
        not_supported_rate=_safe_div(n_not_supported, n_evaluated),
    )


def summarize_dataset(per_doc: List[PerDocSummary]) -> DatasetSummary:
    # Documents with zero evaluated triples can't contribute a meaningful
    # rate — exclude them from macro stats so they don't drag the mean to 0.
    valid = [d for d in per_doc if d.n_evaluated > 0]
    faithfulness_vals = [d.faithfulness for d in valid]
    not_supported_vals = [d.not_supported_rate for d in valid]

    n_supported_total = sum(d.n_supported for d in per_doc)
    n_evaluated_total = sum(d.n_evaluated for d in per_doc)

    return DatasetSummary(
        n_docs=len(per_doc),
        n_triples_total=sum(d.n_total for d in per_doc),
        n_malformed_total=sum(d.n_malformed for d in per_doc),
        n_judge_errors_total=sum(d.n_judge_errors for d in per_doc),
        n_evaluated_total=n_evaluated_total,
        n_supported_total=n_supported_total,
        n_not_supported_total=sum(d.n_not_supported for d in per_doc),
        overall_micro_faithfulness=_safe_div(n_supported_total, n_evaluated_total),
        faithfulness_mean=(
            statistics.fmean(faithfulness_vals) if faithfulness_vals else 0.0
        ),
        faithfulness_std=_std(faithfulness_vals),
        not_supported_rate_mean=(
            statistics.fmean(not_supported_vals) if not_supported_vals else 0.0
        ),
        not_supported_rate_std=_std(not_supported_vals),
    )


def audit_to_dict(a: PerTripleAudit) -> Dict:
    return asdict(a)


def per_doc_to_dict(d: PerDocSummary) -> Dict:
    return asdict(d)


def dataset_to_dict(d: DatasetSummary) -> Dict:
    return asdict(d)
