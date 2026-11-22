"""Diagnose why hybrid fusion loses to BM25 alone, and sweep the fusion knobs.

The observation to explain
--------------------------
On dev, structure-aware hybrid scores R@5 0.7459 / R@1 0.4261 while BM25 alone
scores R@5 0.7965 / R@1 0.5067. A fusion that can only add evidence should not
lose to one of its own inputs unless the fusion rule is discarding it.

What this script prints
-----------------------
1. For up to N dev queries where BM25 hits in the top 5 and hybrid does not,
   the gold chunk's rank in the sparse channel, in the dense channel, and in
   the fused list. If the gold chunk is rank 1 for BM25 and rank 40 fused, the
   fusion rule is the problem, not retrieval.
2. The fusion configuration: candidate depth per channel, RRF constant, channel
   weights, tie-breaking, dedup, and chunk-id alignment.
3. A dev-only sweep over RRF constant (10, 30, 60, 100) and channel depth
   (20, 50, 100), reporting R@1, R@5, and MRR for each cell.

Everything here is retrieval-only: no model calls, no contract text printed
beyond section numbers and ranks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    first_hit_rank,
    mean,
    mrr_score,
    recall_at_k,
)
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import (
    HybridRetriever,
    _fused_order,
    _reciprocal_rank_fusion,
)
from clausewright.schemas import Chunk

RRF_CONSTANTS: tuple[int, ...] = (10, 30, 60, 100)
DEPTHS: tuple[int, ...] = (20, 50, 100)
SEED = CUAD_SMALL.seed


def _gold_rank(golds: list[str], ranked: list[Chunk]) -> int | None:
    return first_hit_rank(
        golds, [c.text for c in ranked], threshold=SPAN_OVERLAP_THRESHOLD
    )


def _fused_all(
    query: str,
    retriever: HybridRetriever,
    depth: int,
) -> list[Chunk]:
    """Fuse the full depth-N lists, not just the returned top 10."""
    sparse = [c.chunk_id for c, _ in retriever.search_sparse(query, top_k=depth)]
    dense = [c.chunk_id for c, _ in retriever.search_dense(query, top_k=depth)]
    scores = _reciprocal_rank_fusion([dense, sparse], rrf_k=retriever.rrf_k)
    return [retriever._chunk_map[cid] for cid in _fused_order(scores)]


def diagnose(
    *,
    split: str,
    limit: int,
    sample_contracts: int | None,
) -> list[dict[str, Any]]:
    """Collect the queries where BM25 wins and hybrid loses."""
    eval_set = build_eval_set(CUAD_SMALL)
    items = [i for i in eval_set.by_split(split) if i.is_answerable]  # type: ignore[arg-type]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_contract[item.contract_title].append(item)
    titles = sorted(by_contract)
    if sample_contracts is not None:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    findings: list[dict[str, Any]] = []

    for title in titles:
        doc = load_pdf(by_contract[title][0].pdf_path, doc_id=title)
        sections = parse_sections(doc)
        chunks = chunk_sections(title, sections)
        retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)

        for item in by_contract[title]:
            golds = list(item.gold_texts)
            sparse10 = [c for c, _ in retriever.search_sparse(item.question, top_k=10)]
            hybrid10 = [c for c, _ in retriever.search(item.question, top_k=10)]

            sparse_rank = _gold_rank(golds, sparse10)
            hybrid_rank = _gold_rank(golds, hybrid10)
            if sparse_rank is None or sparse_rank > 5:
                continue
            if hybrid_rank is not None:
                continue

            dense50 = [c for c, _ in retriever.search_dense(item.question, top_k=50)]
            fused_all = _fused_all(item.question, retriever, depth=50)
            findings.append(
                {
                    "item_id": item.item_id,
                    "contract": title,
                    "category": item.category,
                    "question": item.question,
                    "bm25_rank": sparse_rank,
                    "dense_rank": _gold_rank(golds, dense50),
                    "fused_rank": _gold_rank(golds, fused_all),
                    "fused_top_sections": [c.section_number for c in hybrid10[:5]],
                }
            )
            if len(findings) >= limit:
                return findings

    return findings


def sweep(*, split: str, sample_contracts: int | None) -> list[dict[str, Any]]:
    """R@1 / R@5 / MRR for every (rrf_k, depth) cell on dev."""
    eval_set = build_eval_set(CUAD_SMALL)
    items = [i for i in eval_set.by_split(split) if i.is_answerable]  # type: ignore[arg-type]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_contract[item.contract_title].append(item)
    titles = sorted(by_contract)
    if sample_contracts is not None:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    rows: list[dict[str, Any]] = []

    # Parse and embed each contract once. The sweep changes fusion parameters
    # only, so rebuilding the FAISS index per cell would multiply the run time
    # by twelve and measure nothing new.
    prepared: dict[str, tuple[list[Chunk], Any]] = {}
    for title in titles:
        doc = load_pdf(by_contract[title][0].pdf_path, doc_id=title)
        sections = parse_sections(doc)
        chunks = chunk_sections(title, sections)
        index = embedder.build_index(chunks)[0]
        prepared[title] = (chunks, index)

    for rrf_k in RRF_CONSTANTS:
        for depth in DEPTHS:
            r1: list[float] = []
            r5: list[float] = []
            mrrs: list[float] = []
            for title in titles:
                chunks, index = prepared[title]
                retriever = HybridRetriever(
                    embedder=embedder,
                    index=index,
                    chunks=chunks,
                    dense_candidates=depth,
                    sparse_candidates=depth,
                    rrf_k=rrf_k,
                )
                for item in by_contract[title]:
                    golds = list(item.gold_texts)
                    ranked = [c for c, _ in retriever.search(item.question, top_k=10)]
                    texts = [c.text for c in ranked]
                    r1.append(
                        recall_at_k(golds, texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD)
                    )
                    r5.append(
                        recall_at_k(golds, texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD)
                    )
                    mrrs.append(mrr_score(golds, texts, threshold=SPAN_OVERLAP_THRESHOLD))
            rows.append(
                {
                    "rrf_k": rrf_k,
                    "depth": depth,
                    "recall_at_1": round(mean(r1), 4),
                    "recall_at_5": round(mean(r5), 4),
                    "mrr": round(mean(mrrs), 4),
                    "n_queries": len(r1),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sweep", action="store_true", help="run the RRF/depth sweep")
    parser.add_argument("--out", default="data/eval/fusion_diagnosis.json")
    args = parser.parse_args(argv)

    print(f"split={args.split} seed={SEED}")
    print(
        "fusion config: equal channel weights, rrf_k=60, dense_candidates=50, "
        "sparse_candidates=50, ties broken by (-score, chunk_id) so channel "
        "order cannot change the result, dedup by chunk_id, identical chunk "
        "lists for both channels"
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "split": args.split,
        "seed": SEED,
        "cases": diagnose(
            split=args.split, limit=args.limit, sample_contracts=args.sample
        ),
    }

    print(f"\n=== {len(report['cases'])} queries where bm25 hits top 5 and hybrid misses top 10 ===")
    header = f"{'bm25':>5} {'dense':>6} {'fused':>6}  {'category':<28} contract"
    print(header)
    print("-" * len(header))
    for case in report["cases"]:
        dense = case["dense_rank"] if case["dense_rank"] is not None else "miss"
        fused = case["fused_rank"] if case["fused_rank"] is not None else "miss"
        print(
            f"{case['bm25_rank']:>5} {dense!s:>6} {fused!s:>6}  "
            f"{case['category']:<28} {case['contract'][:40]}"
        )

    if args.sweep:
        rows = sweep(split=args.split, sample_contracts=args.sample)
        report["sweep"] = rows
        print("\n=== RRF constant x channel depth (dev) ===")
        head = f"{'rrf_k':>6} {'depth':>6} {'R@1':>7} {'R@5':>7} {'MRR':>7} {'n':>5}"
        print(head)
        print("-" * len(head))
        for row in rows:
            print(
                f"{row['rrf_k']:>6} {row['depth']:>6} {row['recall_at_1']:>7.4f} "
                f"{row['recall_at_5']:>7.4f} {row['mrr']:>7.4f} {row['n_queries']:>5}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
