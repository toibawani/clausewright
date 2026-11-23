"""Three deterministic attempts at the date questions, measured on dev.

The date categories (Agreement Date, Effective Date, Expiration Date) are
missed because the annotator's question wording and the contract's wording
differ: contracts say "made and entered into as of", "dated as of", or give
the date on a cover page that no section heading covers.

Variants, each applied only to the date categories:

* ``baseline`` - the current path, hybrid fusion then cross-encoder rerank.
* ``expansion`` - append the date-clause synonyms to the query.
* ``prior`` - after fusion, add half a first-place RRF vote to chunks that
  start on page 1 or come from the preamble.
* ``both`` - expansion and prior together.

Every variant is also measured on the non-date categories, because a date fix
that quietly costs ordinary clauses is not a fix. Intervals are paired
bootstrap over contracts against the baseline.
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
from clausewright.schemas import Chunk

DATE_CATEGORIES: frozenset[str] = frozenset(
    {"Agreement Date", "Effective Date", "Expiration Date"}
)

#: Synonyms that appear in date clauses but rarely in the question.
DATE_SYNONYMS: tuple[str, ...] = (
    "made and entered into",
    "dated as of",
    "effective as of",
    "commencement date",
)

VARIANTS: tuple[str, ...] = ("baseline", "expansion", "prior", "both")

RERANK_CANDIDATES = 25
TOP_K = 10
SEED = CUAD_SMALL.seed


def _boost(retriever: HybridRetriever) -> float:
    """Half a first-place RRF vote, in the score units fusion actually uses."""
    return 0.5 / (retriever.rrf_k + 1)


def _query_for(variant: str, item: EvalItem) -> str:
    if variant in ("expansion", "both") and item.category in DATE_CATEGORIES:
        return item.question + " " + " ".join(DATE_SYNONYMS)
    return item.question


def _is_front_matter(chunk: Chunk) -> bool:
    return chunk.page_range[0] == 1 or chunk.section_number == "PREAMBLE"


def _ranked(
    variant: str,
    item: EvalItem,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> list[Chunk]:
    """Top-10 chunks for one variant."""
    query = _query_for(variant, item)
    use_prior = variant in ("prior", "both") and item.category in DATE_CATEGORIES
    # Channel depth before fusion, matching the retriever default.
    channel_depth = 50

    if use_prior:
        sparse = [c.chunk_id for c, _ in retriever.search_sparse(query, top_k=channel_depth)]
        dense = [c.chunk_id for c, _ in retriever.search_dense(query, top_k=channel_depth)]
        scores = _reciprocal_rank_fusion([dense, sparse], rrf_k=retriever.rrf_k)
        boost = _boost(retriever)
        for chunk_id in scores:
            if _is_front_matter(retriever._chunk_map[chunk_id]):
                scores[chunk_id] += boost
        candidates = [
            (retriever._chunk_map[cid], scores[cid]) for cid in _fused_order(scores)
        ][:RERANK_CANDIDATES]
    else:
        candidates = retriever.search(query, top_k=RERANK_CANDIDATES)

    return [c for c, _ in reranker.rerank(query, candidates, top_n=TOP_K)]


def _score(item: EvalItem, ranked: list[Chunk]) -> dict[str, float]:
    golds = list(item.gold_texts)
    texts = [c.text for c in ranked]
    return {
        "recall_at_1": recall_at_k(golds, texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD),
        "recall_at_5": recall_at_k(golds, texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD),
        "mrr": mrr_score(golds, texts, threshold=SPAN_OVERLAP_THRESHOLD),
    }


def run(*, split: str, sample_contracts: int | None) -> dict[str, Any]:
    eval_set = build_eval_set(CUAD_SMALL)
    items = [i for i in eval_set.by_split(split) if i.is_answerable]  # type: ignore[arg-type]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_contract[item.contract_title].append(item)
    titles = sorted(by_contract)
    if sample_contracts is not None:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    reranker = CrossEncoderReranker()

    # variant -> category group -> rows
    rows: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}

    for index, title in enumerate(titles, start=1):
        print(f"[{index}/{len(titles)}] {title}", flush=True)
        contract_items = by_contract[title]
        doc = load_pdf(contract_items[0].pdf_path, doc_id=title)
        retriever = HybridRetriever.from_chunks(
            chunk_sections(title, parse_sections(doc)), embedder=embedder
        )

        for item in contract_items:
            group = "date" if item.category in DATE_CATEGORIES else "other"
            for variant in VARIANTS:
                metrics = _score(item, _ranked(variant, item, retriever, reranker))
                rows[variant].append(
                    {
                        "item_id": item.item_id,
                        "contract": title,
                        "category": item.category,
                        "group": group,
                        **metrics,
                    }
                )

    def _summary(variant: str, group: str | None) -> dict[str, float | int]:
        subset = [r for r in rows[variant] if group is None or r["group"] == group]
        return {
            "n_queries": len(subset),
            "recall_at_1": round(mean([r["recall_at_1"] for r in subset]), 4),
            "recall_at_5": round(mean([r["recall_at_5"] for r in subset]), 4),
            "mrr": round(mean([r["mrr"] for r in subset]), 4),
        }

    summary = {
        variant: {
            "all": _summary(variant, None),
            "date": _summary(variant, "date"),
            "other": _summary(variant, "other"),
        }
        for variant in VARIANTS
    }

    comparisons: list[PairedDifference] = []
    base_by_item = {r["item_id"]: r for r in rows["baseline"]}
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        for group in ("date", "other"):
            subset = [r for r in rows[variant] if r["group"] == group]
            for metric in ("recall_at_5", "recall_at_1", "mrr"):
                pairs = [
                    (r["contract"], r[metric], base_by_item[r["item_id"]][metric])
                    for r in subset
                ]
                comparisons.append(
                    paired_bootstrap(
                        metric=f"{group}_{metric}",
                        left_label=variant,
                        right_label="baseline",
                        pairs=pairs,
                        seed=SEED,
                    )
                )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "split": split,
            "date_categories": sorted(DATE_CATEGORIES),
            "synonyms": list(DATE_SYNONYMS),
            "prior_boost": "half a first-place RRF vote: 0.5 / (rrf_k + 1)",
            "rerank_candidates": RERANK_CANDIDATES,
            "seed": SEED,
        },
        "dataset_hash": eval_set.dataset_hash,
        "n_contracts": len(titles),
        "n_queries": sum(len(by_contract[t]) for t in titles),
        "summary": summary,
        "comparisons": [c.as_dict() for c in comparisons],
        "per_query": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== date categories ===")
    head = f"{'variant':<10} {'R@1':>7} {'R@5':>7} {'MRR':>7} {'n':>4}"
    print(head)
    print("-" * len(head))
    for variant in VARIANTS:
        s = report["summary"][variant]["date"]
        print(
            f"{variant:<10} {s['recall_at_1']:>7.4f} {s['recall_at_5']:>7.4f} "
            f"{s['mrr']:>7.4f} {s['n_queries']:>4}"
        )

    print("\n=== every other category ===")
    print(head)
    print("-" * len(head))
    for variant in VARIANTS:
        s = report["summary"][variant]["other"]
        print(
            f"{variant:<10} {s['recall_at_1']:>7.4f} {s['recall_at_5']:>7.4f} "
            f"{s['mrr']:>7.4f} {s['n_queries']:>4}"
        )

    print("\n=== paired bootstrap vs baseline ===")
    print("(10000 resamples by contract, seed 20261018)")
    head2 = f"{'variant':<10} {'metric':<22} {'point':>8} {'ci_low':>8} {'ci_high':>8}  verdict"
    print(head2)
    print("-" * len(head2))
    for comp in report["comparisons"]:
        verdict = "excludes 0" if comp["excludes_zero"] else "includes 0"
        print(
            f"{comp['left']:<10} {comp['metric']:<22} {comp['point']:>+8.4f} "
            f"{comp['ci_low']:>+8.4f} {comp['ci_high']:>+8.4f}  {verdict}"
        )

    print(
        f"\n{report['n_queries']} queries, {report['n_contracts']} contracts"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default="data/eval/date_experiments.json")
    args = parser.parse_args(argv)

    report = run(split=args.split, sample_contracts=args.sample)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
