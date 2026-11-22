"""Sweep the fusion knobs on the fixed chunk pipeline and report intervals.

What is swept
-------------
* RRF constant: 10, 30, 60, 100.
* Channel depth: 20, 50, 100 candidates per channel before fusion.
* Chunking: structure-aware windows, and matched-size fixed windows at 300 and
  400 words, so a fusion setting cannot win by chunk size alone.
* Retriever: hybrid (fused list straight to the top 10) and hybrid_rerank
  (top 25 fused, cross-encoder reranked to the top 10).

Why per-query scores are dumped
-------------------------------
The decision rule is a paired bootstrap interval by contract against the
current default (rrf_k=60, depth=50). Intervals cannot be recovered from
means, so every query's score is written to the JSON.

Everything here is retrieval-only and dev-only: no model calls, no test split,
no contract text beyond section numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    mean,
    mrr_score,
    recall_at_k,
)
from clausewright.eval.stats import PairedDifference, paired_bootstrap
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import (
    HybridRetriever,
    _fused_order,
    _reciprocal_rank_fusion,
)
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk, ParsedDocument, Section

RRF_CONSTANTS: tuple[int, ...] = (10, 30, 60, 100)
DEPTHS: tuple[int, ...] = (20, 50, 100)
CHUNKINGS: tuple[str, ...] = ("struct", "win300", "win400")
RETRIEVERS: tuple[str, ...] = ("hybrid", "hybrid_rerank")

#: The current default the sweep is measured against.
DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 50

RERANK_CANDIDATES = 25
SEED = CUAD_SMALL.seed


def _build_chunks(
    chunking: str, doc_id: str, doc: ParsedDocument, sections: list[Section]
) -> list[Chunk]:
    """Materialise one chunking of one document."""
    if chunking == "struct":
        return chunk_sections(doc_id, sections)
    if chunking.startswith("win"):
        size = int(chunking[3:])
        return fixed_window_chunks(doc_id, doc, max_words=size, stride=size // 2)
    raise ValueError(f"unknown chunking: {chunking}")


def _cell(chunking: str, retriever: str, rrf_k: int, depth: int) -> str:
    return f"{chunking}:{retriever}:k{rrf_k}:d{depth}"


def _score(item: EvalItem, ranked: list[Chunk]) -> dict[str, float]:
    golds = list(item.gold_texts)
    texts = [c.text for c in ranked]
    return {
        "recall_at_1": recall_at_k(golds, texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD),
        "recall_at_5": recall_at_k(golds, texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD),
        "recall_at_10": recall_at_k(golds, texts, k=10, threshold=SPAN_OVERLAP_THRESHOLD),
        "mrr": mrr_score(golds, texts, threshold=SPAN_OVERLAP_THRESHOLD),
    }


def run(
    *,
    split: str,
    chunkings: tuple[str, ...],
    retrievers: tuple[str, ...],
    rrf_constants: tuple[int, ...],
    depths: tuple[int, ...],
    sample_contracts: int | None,
) -> dict[str, Any]:
    """Run the sweep and return a JSON-serialisable report."""
    started = time.time()

    eval_set = build_eval_set(CUAD_SMALL)
    items = [i for i in eval_set.by_split(split) if i.is_answerable]  # type: ignore[arg-type]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_contract[item.contract_title].append(item)
    titles = sorted(by_contract)
    if sample_contracts is not None:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    reranker = CrossEncoderReranker() if "hybrid_rerank" in retrievers else None
    max_depth = max(depths)

    per_query: dict[str, list[dict[str, Any]]] = {
        _cell(c, r, k, d): []
        for c in chunkings
        for r in retrievers
        for k in rrf_constants
        for d in depths
    }

    for index, title in enumerate(titles, start=1):
        contract_items = by_contract[title]
        print(f"[{index}/{len(titles)}] {title}", flush=True)
        doc = load_pdf(contract_items[0].pdf_path, doc_id=title)
        sections = parse_sections(doc)

        for chunking in chunkings:
            chunks = _build_chunks(chunking, title, doc, sections)
            retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)

            for item in contract_items:
                # Each channel is ranked once at maximum depth; every cell is a
                # prefix of these lists, so twelve cells cost one search.
                sparse_ids = [
                    c.chunk_id
                    for c, _ in retriever.search_sparse(item.question, top_k=max_depth)
                ]
                dense_ids = [
                    c.chunk_id
                    for c, _ in retriever.search_dense(item.question, top_k=max_depth)
                ]

                for depth in depths:
                    sparse = sparse_ids[:depth]
                    dense = dense_ids[:depth]
                    for rrf_k in rrf_constants:
                        scores = _reciprocal_rank_fusion([dense, sparse], rrf_k=rrf_k)
                        fused = [retriever._chunk_map[cid] for cid in _fused_order(scores)]

                        for name in retrievers:
                            if name == "hybrid":
                                ranked = fused[:10]
                            else:
                                assert reranker is not None
                                candidates = [(c, 0.0) for c in fused[:RERANK_CANDIDATES]]
                                ranked = [
                                    c
                                    for c, _ in reranker.rerank(
                                        item.question, candidates, top_n=10
                                    )
                                ]
                            metrics = _score(item, ranked)
                            per_query[_cell(chunking, name, rrf_k, depth)].append(
                                {
                                    "item_id": item.item_id,
                                    "contract": title,
                                    "category": item.category,
                                    **metrics,
                                }
                            )

    summary: dict[str, dict[str, float | int]] = {}
    for key, rows in per_query.items():
        summary[key] = {
            "n_queries": len(rows),
            "recall_at_1": round(mean([r["recall_at_1"] for r in rows]), 4),
            "recall_at_5": round(mean([r["recall_at_5"] for r in rows]), 4),
            "recall_at_10": round(mean([r["recall_at_10"] for r in rows]), 4),
            "mrr": round(mean([r["mrr"] for r in rows]), 4),
        }

    comparisons: list[PairedDifference] = []
    for chunking in chunkings:
        for name in retrievers:
            reference = per_query[_cell(chunking, name, DEFAULT_RRF_K, DEFAULT_DEPTH)]
            for rrf_k in rrf_constants:
                for depth in depths:
                    if (rrf_k, depth) == (DEFAULT_RRF_K, DEFAULT_DEPTH):
                        continue
                    left = per_query[_cell(chunking, name, rrf_k, depth)]
                    for metric in ("recall_at_5", "recall_at_1"):
                        pairs = [
                            (a["contract"], a[metric], b[metric])
                            for a, b in zip(left, reference)
                        ]
                        comparisons.append(
                            paired_bootstrap(
                                metric=metric,
                                left_label=_cell(chunking, name, rrf_k, depth),
                                right_label=_cell(
                                    chunking, name, DEFAULT_RRF_K, DEFAULT_DEPTH
                                ),
                                pairs=pairs,
                                seed=SEED,
                            )
                        )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "split": split,
            "chunkings": list(chunkings),
            "retrievers": list(retrievers),
            "rrf_constants": list(rrf_constants),
            "depths": list(depths),
            "default": {"rrf_k": DEFAULT_RRF_K, "depth": DEFAULT_DEPTH},
            "rerank_candidates": RERANK_CANDIDATES,
            "seed": SEED,
        },
        "dataset_hash": eval_set.dataset_hash,
        "n_contracts": len(titles),
        "n_queries": sum(len(by_contract[t]) for t in titles),
        "metrics": summary,
        "comparisons": [c.as_dict() for c in comparisons],
        "per_query": per_query,
        "elapsed_seconds": round(time.time() - started, 1),
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the sweep table and every comparison against the default."""
    print("\n=== RRF constant x channel depth ===")
    head = f"{'cell':<34} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'n':>4}"
    print(head)
    print("-" * len(head))
    for cell, metrics in report["metrics"].items():
        print(
            f"{cell:<34} {metrics['recall_at_1']:>7.4f} {metrics['recall_at_5']:>7.4f} "
            f"{metrics['recall_at_10']:>7.4f} {metrics['mrr']:>7.4f} "
            f"{metrics['n_queries']:>4}"
        )

    print("\n=== paired bootstrap vs the current default (rrf_k=60, depth=50) ===")
    print("(10000 resamples by contract, seed 20261018)")
    head2 = f"{'cell':<34} {'metric':<12} {'point':>8} {'ci_low':>8} {'ci_high':>8}  verdict"
    print(head2)
    print("-" * len(head2))
    for comp in report["comparisons"]:
        verdict = "excludes 0" if comp["excludes_zero"] else "includes 0"
        print(
            f"{comp['left']:<34} {comp['metric']:<12} {comp['point']:>+8.4f} "
            f"{comp['ci_low']:>+8.4f} {comp['ci_high']:>+8.4f}  {verdict}"
        )

    print(
        f"\n{report['n_queries']} queries, {report['n_contracts']} contracts, "
        f"{report['elapsed_seconds']}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--chunkings", default=",".join(CHUNKINGS))
    parser.add_argument("--retrievers", default=",".join(RETRIEVERS))
    parser.add_argument(
        "--rrf-constants", default=",".join(str(k) for k in RRF_CONSTANTS)
    )
    parser.add_argument("--depths", default=",".join(str(d) for d in DEPTHS))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default="data/eval/fusion_sweep.json")
    args = parser.parse_args(argv)

    chunkings = tuple(c.strip() for c in args.chunkings.split(",") if c.strip())
    unknown = [c for c in chunkings if c not in CHUNKINGS]
    if unknown:
        parser.error(f"unknown chunking(s): {unknown}")

    report = run(
        split=args.split,
        chunkings=chunkings,
        retrievers=tuple(r.strip() for r in args.retrievers.split(",") if r.strip()),
        rrf_constants=tuple(int(k) for k in args.rrf_constants.split(",")),
        depths=tuple(int(d) for d in args.depths.split(",")),
        sample_contracts=args.sample,
    )
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
