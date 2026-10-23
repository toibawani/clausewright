"""Metric primitives: span overlap, retrieval, abstention, citation quality.

Every function here is pure (no models, no I/O) so each has a hand-computed
unit test.  A metrics bug means fake numbers, so tests assert exact values.

Span matching
-------------
CUAD gold spans are text from *CUAD's own PDF extraction*, while our chunks
come from *our* extraction of the same PDF.  Offsets never line up, so a
chunk "hits" a gold span by text containment:

    overlap(gold, chunk) = matched gold tokens / total gold tokens

where matching is multiset (repeated words count repeatedly) after lowercasing
and whitespace collapse.  A hit requires ``overlap >= SPAN_OVERLAP_THRESHOLD``.
0.5 means most of the clause is present — enough to answer from, forgiving of
OCR/extraction noise that drops or mangles a few words.

Definitions
-----------
* ``recall@k`` (per item): fraction of the item's gold spans hit by *any* of
  the top-k chunks.  Mean over answerable items.  Absent-clause items have no
  spans and are excluded from retrieval metrics entirely.
* ``MRR``: 1 / rank of the first chunk that hits *any* gold span, 0.0 when
  nothing hits.
* abstention precision/recall: positives are abstentions on absent-clause
  items; a false abstention is an abstention on an answerable item.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Final

# A chunk counts as containing a gold span when at least this share of the
# span's tokens (multiset, normalized) appears in the chunk.
SPAN_OVERLAP_THRESHOLD: Final[float] = 0.5


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, collapse whitespace, split on whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower()).split()


def overlap_ratio(gold: str, candidate: str) -> float:
    """Share of *gold* tokens found in *candidate* (multiset containment)."""
    gold_tokens = normalize_tokens(gold)
    if not gold_tokens:
        return 0.0
    remaining = list(normalize_tokens(candidate))
    matched = 0
    for token in gold_tokens:
        try:
            remaining.remove(token)
        except ValueError:
            continue
        matched += 1
    return matched / len(gold_tokens)


def span_hit(gold: str, candidate: str, threshold: float = SPAN_OVERLAP_THRESHOLD) -> bool:
    """True when *candidate* contains enough of the gold span."""
    return overlap_ratio(gold, candidate) >= threshold


def first_hit_rank(
    gold_spans: Sequence[str],
    ranked_texts: Sequence[str],
    threshold: float = SPAN_OVERLAP_THRESHOLD,
) -> int | None:
    """1-based rank of the first chunk hitting any gold span; None if no hit."""
    for rank, text in enumerate(ranked_texts, start=1):
        if any(span_hit(gold, text, threshold) for gold in gold_spans):
            return rank
    return None


def recall_at_k(
    gold_spans: Sequence[str],
    ranked_texts: Sequence[str],
    k: int,
    threshold: float = SPAN_OVERLAP_THRESHOLD,
) -> float:
    """Fraction of gold spans hit by at least one of the top-k chunks."""
    if not gold_spans:
        raise ValueError("recall_at_k is undefined for items with no gold spans")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    top = ranked_texts[:k]
    hits = sum(1 for gold in gold_spans if any(span_hit(gold, t, threshold) for t in top))
    return hits / len(gold_spans)


def mrr_score(
    gold_spans: Sequence[str],
    ranked_texts: Sequence[str],
    threshold: float = SPAN_OVERLAP_THRESHOLD,
) -> float:
    """1 / rank of the first chunk hitting any gold span; 0.0 when no hit."""
    rank = first_hit_rank(gold_spans, ranked_texts, threshold)
    return 1.0 / rank if rank is not None else 0.0


def abstention_scores(
    *,
    true_absent: Sequence[bool],
    predicted_abstain: Sequence[bool],
) -> dict[str, float]:
    """Precision/recall of the abstain decision (positive = abstained).

    *precision* = of all abstentions, how many were on truly-absent items.
    *recall* = of all absent-clause items, how many were abstained on.
    *false_abstention_rate* = of all answerable items, how many were
    wrongly abstained on.
    """
    if len(true_absent) != len(predicted_abstain):
        raise ValueError("truth and prediction lists must have equal length")
    tp = sum(1 for absent, pred in zip(true_absent, predicted_abstain) if absent and pred)
    abstained_total = sum(1 for pred in predicted_abstain if pred)
    absent_total = sum(1 for absent in true_absent if absent)
    answerable_total = len(true_absent) - absent_total
    return {
        "precision": tp / abstained_total if abstained_total else 0.0,
        "recall": tp / absent_total if absent_total else 1.0,
        "false_abstention_rate": (abstained_total - tp) / answerable_total
        if answerable_total
        else 0.0,
    }


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; NaN over an empty sequence (explicit, not silent 0)."""
    if not values:
        return math.nan
    return sum(values) / len(values)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (p50/p95 latency style)."""
    if not values:
        raise ValueError("percentile is undefined over an empty sequence")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]
