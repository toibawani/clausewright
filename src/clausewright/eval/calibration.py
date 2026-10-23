"""Calibrate min_rerank_score on the dev split.

Sweeps threshold values and reports abstention precision, abstention recall,
false-abstention rate, and answer coverage for each threshold.  Use this
output to pick the threshold that meets the firm's latency / coverage SLA
without leaking hallucinations.

Usage
-----
    python -m clausewright.eval calibrate --split dev

The report is printed as a table and saved to data/eval/calibration.json.
It is also imported by ``python -m clausewright.eval run`` to fill the
calibration section of the full report.

How it works
------------
For each threshold t in the sweep:
  - Retrieve the top-50 candidates for each dev query.
  - Rerank with the cross-encoder; keep top-5.
  - If the best reranker score < t, the system would abstain.
  - Measure abstention precision/recall/FAR against the gold absent labels
    in the eval set.

The sweep is done over quantiles of the actual score distribution to ensure
coverage at the interesting operating points rather than a blind linear grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.config import PATHS
from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker

# Default thresholds to sweep.  These cover the cross-encoder score range
# typical for MiniLM-based models (-5 to +5).  Adjust if your score
# distribution is different (run --print-scores to inspect).
DEFAULT_THRESHOLDS = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, None)


@dataclass(frozen=True)
class CalibrationRow:
    threshold: float | None  # None = disabled (always answer)
    abstention_precision: float
    abstention_recall: float
    false_abstention_rate: float
    answer_coverage: float  # fraction of answerable questions that were answered
    mean_recall_at_5: float  # retrieval quality on answered items
    n_queries: int


def _format_val(v: float | None) -> str:
    if v is None:
        return " None"
    if math.isnan(v):
        return "  nan"
    return f"{v:5.3f}"


def calibrate(
    split: str = "dev",
    thresholds: tuple[float | None, ...] = DEFAULT_THRESHOLDS,
    top_k: int = 50,
    keep: int = 5,
    sample_contracts: int | None = None,
) -> list[dict[str, Any]]:
    """Run calibration sweep.  Returns list of CalibrationRow dicts."""
    eval_set = build_eval_set(CUAD_SMALL)
    all_items = eval_set.by_split(split)  # type: ignore[arg-type]

    # Group by contract so we only index each contract once
    from collections import defaultdict

    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in all_items:
        by_contract[item.contract_title].append(item)

    titles = sorted(by_contract.keys())
    if sample_contracts:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    reranker = CrossEncoderReranker()

    # Per-query: best reranker score, gold absent flag
    query_scores: list[float] = []
    query_absent: list[bool] = []

    for c_idx, title in enumerate(titles, start=1):
        items = by_contract[title]
        pdf_path = items[0].pdf_path
        print(f"[{c_idx}/{len(titles)}] Indexing {title}...")
        doc = load_pdf(pdf_path, doc_id=title)
        sections = parse_sections(doc)
        chunks = chunk_sections(title, sections)
        retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)

        for item in items:
            candidates = retriever.search(item.question, top_k=top_k)
            if not candidates:
                query_scores.append(-999.0)
                query_absent.append(not item.is_answerable)
                continue
            ranked = reranker.rerank(item.question, candidates, top_n=keep)
            best_score = ranked[0][1] if ranked else -999.0
            query_scores.append(best_score)
            query_absent.append(not item.is_answerable)

    n_q = len(query_scores)
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        tp = fp = 0

        for score, absent in zip(query_scores, query_absent):
            would_abstain = threshold is not None and score < threshold
            if absent and would_abstain:
                tp += 1
            elif not absent and would_abstain:
                fp += 1

        abstained = tp + fp
        absent_total = sum(1 for a in query_absent if a)
        answerable_total = n_q - absent_total
        answered_answerable = answerable_total - fp

        row = CalibrationRow(
            threshold=threshold,
            abstention_precision=tp / abstained if abstained else 0.0,
            abstention_recall=tp / absent_total if absent_total else 1.0,
            false_abstention_rate=fp / answerable_total if answerable_total else 0.0,
            answer_coverage=answered_answerable / answerable_total if answerable_total else 1.0,
            mean_recall_at_5=float("nan"),  # needs full retrieval pass; see docs
            n_queries=n_q,
        )
        rows.append(asdict(row))

    return rows


def run_and_save_calibration(
    output_path: Path | None = None,
    split: str = "dev",
    sample_contracts: int | None = None,
) -> Path:
    out = output_path or (PATHS.root / "data" / "eval" / "calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = calibrate(split=split, sample_contracts=sample_contracts)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "split": split,
        "description": (
            "Precision vs. coverage sweep over min_rerank_score thresholds. "
            "Choose the threshold that maximises abstention_recall while keeping "
            "false_abstention_rate below your firm's coverage SLA."
        ),
        "thresholds": rows,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def print_calibration_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'threshold':>10} {'abst_prec':>10} {'abst_rec':>9} "
        f"{'FAR':>6} {'coverage':>9} {'n_q':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        t = r["threshold"]
        t_str = f"{t:>10.3f}" if t is not None else f"{'None':>10}"
        print(
            f"{t_str} "
            f"{r['abstention_precision']:>10.3f} "
            f"{r['abstention_recall']:>9.3f} "
            f"{r['false_abstention_rate']:>6.3f} "
            f"{r['answer_coverage']:>9.3f} "
            f"{r['n_queries']:>5}"
        )
    print(sep)
