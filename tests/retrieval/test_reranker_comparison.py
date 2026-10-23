"""Unit tests for RerankerComparisonReport and run_reranker_comparison (issue #29)."""

from __future__ import annotations

import pytest

from clausewright.retrieval.reranker_comparison import (
    RerankerBenchResult,
    RerankerComparisonReport,
    _mrr,
    _recall_at_k,
    run_reranker_comparison,
)
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(cid: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc1",
        section_number="1.1",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


class _PerfectReranker:
    """Always moves the relevant chunk to the top of the list."""

    def __init__(self, model_name: str, relevant_id: str) -> None:
        self._name = model_name
        self._rid = relevant_id

    @property
    def model_name(self) -> str:
        return self._name

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], *, top_n: int | None = None
    ) -> list[tuple[Chunk, float]]:
        relevant = [(c, s) for c, s in candidates if c.chunk_id == self._rid]
        others = [(c, s) for c, s in candidates if c.chunk_id != self._rid]
        ordered = relevant + others
        if top_n is not None:
            ordered = ordered[:top_n]
        return ordered


class _WorstReranker:
    """Always pushes relevant chunk to the end (reverse order)."""

    def __init__(self, model_name: str) -> None:
        self._name = model_name

    @property
    def model_name(self) -> str:
        return self._name

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], *, top_n: int | None = None
    ) -> list[tuple[Chunk, float]]:
        ordered = list(reversed(candidates))
        if top_n is not None:
            ordered = ordered[:top_n]
        return ordered


# ---------------------------------------------------------------------------
# _mrr
# ---------------------------------------------------------------------------


def test_mrr_first_result_relevant():
    assert _mrr(["c1", "c2", "c3"], {"c1"}) == pytest.approx(1.0)


def test_mrr_second_result_relevant():
    assert _mrr(["c1", "c2", "c3"], {"c2"}) == pytest.approx(0.5)


def test_mrr_none_relevant_in_top10():
    assert _mrr(["c1"] * 10, {"c_missing"}) == pytest.approx(0.0)


def test_mrr_empty_relevant():
    assert _mrr(["c1", "c2"], set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_all_found():
    assert _recall_at_k(["c1", "c2", "c3"], {"c1", "c2"}, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial():
    assert _recall_at_k(["c1", "c2", "c3"], {"c1", "c4"}, k=3) == pytest.approx(0.5)


def test_recall_at_k_empty_relevant():
    assert _recall_at_k(["c1"], set(), k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_reranker_comparison
# ---------------------------------------------------------------------------


def test_comparison_perfect_beats_worst():
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    relevant_id = "c0"
    candidates = [(c, float(5 - i)) for i, c in enumerate(chunks)]
    queries = [("What is the notice period?", candidates, {relevant_id})]

    perfect = _PerfectReranker("perfect-model", relevant_id)
    worst = _WorstReranker("worst-model")

    report = run_reranker_comparison(
        queries_candidates=queries,
        rerankers=[perfect, worst],
        pool_sizes=(5,),
        top_n=5,
    )

    # Perfect reranker should win.
    assert report.winner_model == "perfect-model"

    # MRR for perfect should be 1.0.
    perfect_results = [r for r in report.results if r.model_name == "perfect-model"]
    assert len(perfect_results) == 1
    assert perfect_results[0].mrr == pytest.approx(1.0)


def test_comparison_sweeps_pool_sizes():
    chunks = [_make_chunk(f"c{i}") for i in range(50)]
    relevant_id = "c0"
    candidates = [(c, float(50 - i)) for i, c in enumerate(chunks)]
    queries = [("question", candidates, {relevant_id})]

    perfect = _PerfectReranker("model-a", relevant_id)
    report = run_reranker_comparison(
        queries_candidates=queries,
        rerankers=[perfect],
        pool_sizes=(10, 20, 50),
        top_n=5,
    )
    pool_sizes_seen = {r.pool_size for r in report.results}
    assert pool_sizes_seen == {10, 20, 50}


def test_comparison_empty_queries_no_crash():
    report = run_reranker_comparison(
        queries_candidates=[],
        rerankers=[_PerfectReranker("m", "c0")],
        pool_sizes=(10,),
    )
    assert report.results == []
    assert report.winner_model == ""


def test_format_table_non_empty():
    report = RerankerComparisonReport(
        results=[
            RerankerBenchResult(
                model_name="test-model",
                pool_size=10,
                mrr=0.75,
                recall_at_5=0.8,
                p50_latency_ms=12.3,
                mean_latency_ms=14.0,
                n_queries=5,
            )
        ],
        winner_model="test-model",
        winner_pool=10,
        rationale="Test rationale",
    )
    table = report.format_table()
    assert "test-model" in table
    assert "Winner" in table


def test_format_table_empty():
    report = RerankerComparisonReport()
    assert report.format_table() == "(no results)"
