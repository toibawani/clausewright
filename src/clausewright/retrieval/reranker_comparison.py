"""Reranker comparison harness: candidate pool size × model × CPU latency (closes #29).

Acceptance criteria
-------------------
* Compare at least 2 cross-encoder models.
* Sweep candidate pool sizes k = 10, 20, 50.
* Report MRR, Recall@5, and p50 CPU latency per configuration.
* Adopt the winner or keep current with documented rationale.

This module provides the comparison harness; the results are documented in
``docs/eval.md`` based on running ``python -m clausewright.eval reranker``.

Candidate models evaluated
--------------------------
1. ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — the current default.
   6-layer MiniLM, ~23 MB, very fast on CPU.

2. ``cross-encoder/ms-marco-MiniLM-L-12-v2`` — 12-layer variant.
   ~34 MB, ~1.8× slower than L-6 but typically +2–3 MRR points on MARCO.

The harness is intentionally retriever-agnostic: it accepts any list of
``(Chunk, float)`` candidate pools and any list of reranker instances.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


class _SupportsRerank(Protocol):
    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        *,
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]: ...


@dataclass(frozen=True)
class RerankerBenchResult:
    """Results for one (model, pool_size) configuration."""

    model_name: str
    pool_size: int
    mrr: float
    recall_at_5: float
    p50_latency_ms: float
    mean_latency_ms: float
    n_queries: int


@dataclass
class RerankerComparisonReport:
    """Aggregated comparison report across all configurations."""

    results: list[RerankerBenchResult] = field(default_factory=list)
    winner_model: str = ""
    winner_pool: int = 0
    rationale: str = ""

    def format_table(self) -> str:
        """Render an ASCII comparison table."""
        if not self.results:
            return "(no results)"
        header = (
            f"{'Model':<45} {'Pool':>4} {'MRR':>6} {'R@5':>6} "
            f"{'p50ms':>7} {'meanms':>8}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for r in sorted(self.results, key=lambda x: (-x.mrr, x.p50_latency_ms)):
            rows.append(
                f"{r.model_name:<45} {r.pool_size:>4} {r.mrr:>6.3f} "
                f"{r.recall_at_5:>6.3f} {r.p50_latency_ms:>7.1f} "
                f"{r.mean_latency_ms:>8.1f}"
            )
        rows.append(sep)
        rows.append(f"Winner: {self.winner_model}  pool={self.winner_pool}")
        rows.append(f"Rationale: {self.rationale}")
        return "\n".join(rows)


def _mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none in top-10)."""
    for rank, cid in enumerate(ranked_ids[:10], start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of relevant IDs found in the top-k ranked results."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for cid in ranked_ids[:k] if cid in relevant_ids)
    return found / len(relevant_ids)


def run_reranker_comparison(
    queries_candidates: list[tuple[str, list[tuple[Chunk, float]], set[str]]],
    rerankers: list[_SupportsRerank],
    pool_sizes: tuple[int, ...] = (10, 20, 50),
    top_n: int = 5,
) -> RerankerComparisonReport:
    """Benchmark multiple rerankers across pool sizes.

    Parameters
    ----------
    queries_candidates:
        List of ``(query, candidate_pool, relevant_chunk_ids)`` triples.
        The candidate pool must be large enough to satisfy the largest
        pool_size (i.e., ``len(candidate_pool) >= max(pool_sizes)``).
    rerankers:
        Reranker instances to compare.  Each must have a ``model_name``
        property and a ``rerank`` method.
    pool_sizes:
        Candidate pool sizes to sweep.  Each run uses only the first
        *pool_size* candidates from the provided pool.
    top_n:
        Number of top results to pass through after reranking (used for
        Recall@5 etc.).

    Returns
    -------
    RerankerComparisonReport
        Populated report with per-configuration results and a winner selection.
    """
    report = RerankerComparisonReport()

    for reranker in rerankers:
        for pool_size in pool_sizes:
            latencies_ms: list[float] = []
            mrrs: list[float] = []
            r_at_5s: list[float] = []

            for query, all_candidates, relevant_ids in queries_candidates:
                pool = all_candidates[:pool_size]
                if not pool:
                    continue

                t0 = time.perf_counter()
                ranked = reranker.rerank(query, pool, top_n=top_n)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                ranked_ids = [c.chunk_id for c, _ in ranked]
                mrrs.append(_mrr(ranked_ids, relevant_ids))
                r_at_5s.append(_recall_at_k(ranked_ids, relevant_ids, k=top_n))
                latencies_ms.append(elapsed_ms)

            if not latencies_ms:
                continue

            result = RerankerBenchResult(
                model_name=reranker.model_name,
                pool_size=pool_size,
                mrr=statistics.mean(mrrs),
                recall_at_5=statistics.mean(r_at_5s),
                p50_latency_ms=sorted(latencies_ms)[len(latencies_ms) // 2],
                mean_latency_ms=statistics.mean(latencies_ms),
                n_queries=len(latencies_ms),
            )
            report.results.append(result)
            logger.info(
                "Reranker=%s pool=%d MRR=%.3f R@5=%.3f p50=%.1fms",
                reranker.model_name,
                pool_size,
                result.mrr,
                result.recall_at_5,
                result.p50_latency_ms,
            )

    if report.results:
        # Select winner: highest MRR; break ties by lowest p50 latency.
        winner = min(
            report.results,
            key=lambda r: (-r.mrr, r.p50_latency_ms),
        )
        report.winner_model = winner.model_name
        report.winner_pool = winner.pool_size
        # Auto-rationale
        current = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if winner.model_name == current:
            report.rationale = (
                f"Current default model retained — highest MRR ({winner.mrr:.3f}) "
                f"at pool={winner.pool_size} with p50={winner.p50_latency_ms:.1f}ms."
            )
        else:
            report.rationale = (
                f"New model selected: MRR={winner.mrr:.3f} "
                f"pool={winner.pool_size} p50={winner.p50_latency_ms:.1f}ms. "
                "Update config.reranker_model and config.retrieval_top_k."
            )

    return report
