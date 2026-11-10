"""Monotonicity / range invariants for retrieval metrics."""
from __future__ import annotations

from clausewright.eval.metrics import mrr_score, recall_at_k


def test_recall_at_k_is_non_decreasing_in_k() -> None:
    gold = ["notice period is thirty days", "governing law is new york"]
    chunks = [
        "payment terms net thirty",
        "this agreement is governed by the laws of new york",
        "notice period of thirty days applies",
    ]
    values = [recall_at_k(gold, chunks, k=k) for k in range(1, len(chunks) + 1)]
    assert values == sorted(values)


def test_mrr_lies_in_unit_interval() -> None:
    gold = ["indemnification obligation"]
    chunks = ["1", "2", "indemnification obligation covers losses"]
    score = mrr_score(gold, chunks)
    assert 0.0 <= score <= 1.0
