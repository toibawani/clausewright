"""Full factorial retrieval evaluation on dev.

What this answers
-----------------
The committed eval compares five configurations that each change two things at
once (chunking and retriever), so it cannot say whether the section prior helps
or whether a particular retriever just likes long chunks. This script crosses
the two variables:

    chunking  x  retriever
    struct    x  bm25, dense, hybrid, hybrid_rerank
    win150    x  bm25, dense, hybrid, hybrid_rerank
    win200    x  ...
    win300    x  ...
    win400    x  ...

``struct`` is the product chunker: sections parsed from the PDF, then 300-word
windows at 50 percent overlap. ``winN`` is the same windowing machinery run
over one pseudo-section, so text, tokenizer, and overlap are identical; only
the section boundary is removed. Matching ``win300`` against ``struct`` is
therefore a like-for-like test of the section prior.

Two metrics beyond recall@k
---------------------------
*Recall at an equal retrieved budget* answers the question a product actually
faces: with 1500 (or 3000) words of context, how much of the gold clause is
present? Chunks are taken in ranked order until the next one would exceed the
budget. This corrects a real confound in recall@k, where longer chunks win by
carrying more text per slot.

*Paired bootstrap by contract* gives a 95 percent interval for every
difference. Queries inside a contract are not independent; resampling
contracts is the conservative choice.

Usage
-----
    python scripts/factorial_eval.py --split dev
    python scripts/factorial_eval.py --chunkings struct,win300 --retrievers hybrid
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    first_hit_rank,
    mean,
    mrr_score,
    percentile,
    recall_at_k,
)
from clausewright.eval.stats import PairedDifference, paired_bootstrap
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk, ParsedDocument, Section

#: Chunkings under test. ``struct`` is the product chunker at its default
#: 300-word window; ``winN`` is the fixed-window baseline at N words.
CHUNKINGS: tuple[str, ...] = ("struct", "win150", "win200", "win300", "win400")
RETRIEVERS: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")

#: Equal retrieved budgets, in words.
WORD_BUDGETS: tuple[int, ...] = (1500, 3000)

#: Candidate depth for the reranker, matching the product path.
RERANK_CANDIDATES = 25

SEED = CUAD_SMALL.seed


@dataclass(frozen=True)
class QueryScore:
    """One query's outcome for one (chunking, retriever) cell."""

    item_id: str
    contract: str
    category: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    first_hit_rank: int | None
    recall_at_1500w: float
    recall_at_3000w: float
    top_sections: tuple[str, ...]


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def proof_header() -> dict[str, str]:
    """Everything needed to tell which code and interpreter produced a number."""
    import clausewright

    return {
        "branch": _git_branch(),
        "commit": _git_short_sha(),
        "python": sys.version.split()[0],
        "clausewright_path": str(Path(clausewright.__file__).resolve()),
    }


def build_chunks(
    chunking: str,
    doc_id: str,
    doc: ParsedDocument,
    sections: list[Section],
) -> list[Chunk]:
    """Materialise the chunk list for one chunking of one document."""
    if chunking == "struct":
        return chunk_sections(doc_id, sections)
    if not chunking.startswith("win"):
        raise ValueError(f"unknown chunking: {chunking}")
    size = int(chunking[3:])
    return fixed_window_chunks(doc_id, doc, max_words=size, stride=size // 2)


def chunks_within_word_budget(ranked: list[Chunk], budget: int) -> list[str]:
    """Texts of the ranked chunks that fit in *budget* words.

    The top chunk is always included even when it alone exceeds the budget:
    returning nothing would make the metric describe the budget rather than
    the retriever.
    """
    selected: list[str] = []
    used = 0
    for chunk in ranked:
        if selected and used + chunk.word_count > budget:
            break
        selected.append(chunk.text)
        used += chunk.word_count
        if used >= budget:
            break
    return selected


def score_query(
    item: EvalItem,
    ranked: list[Chunk],
) -> QueryScore:
    """Compute every retrieval metric for one query and one ranked list."""
    golds = list(item.gold_texts)
    texts = [c.text for c in ranked]
    within_1500 = chunks_within_word_budget(ranked, WORD_BUDGETS[0])
    within_3000 = chunks_within_word_budget(ranked, WORD_BUDGETS[1])
    return QueryScore(
        item_id=item.item_id,
        contract=item.contract_title,
        category=item.category,
        recall_at_1=recall_at_k(golds, texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD),
        recall_at_5=recall_at_k(golds, texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD),
        recall_at_10=recall_at_k(golds, texts, k=10, threshold=SPAN_OVERLAP_THRESHOLD),
        mrr=mrr_score(golds, texts, threshold=SPAN_OVERLAP_THRESHOLD),
        first_hit_rank=first_hit_rank(golds, texts, threshold=SPAN_OVERLAP_THRESHOLD),
        recall_at_1500w=recall_at_k(
            golds, within_1500, k=len(within_1500), threshold=SPAN_OVERLAP_THRESHOLD
        ),
        recall_at_3000w=recall_at_k(
            golds, within_3000, k=len(within_3000), threshold=SPAN_OVERLAP_THRESHOLD
        ),
        top_sections=tuple(c.section_number for c in ranked[:5]),
    )


def rank_for(
    retriever: str,
    query: str,
    struct_retriever: HybridRetriever,
    reranker: CrossEncoderReranker | None,
) -> list[Chunk]:
    """Produce the top-10 chunk list for one retriever."""
    if retriever == "bm25":
        return [c for c, _ in struct_retriever.search_sparse(query, top_k=10)]
    if retriever == "dense":
        return [c for c, _ in struct_retriever.search_dense(query, top_k=10)]
    if retriever == "hybrid":
        return [c for c, _ in struct_retriever.search(query, top_k=10)]
    if retriever == "hybrid_rerank":
        if reranker is None:
            raise ValueError("hybrid_rerank requires a reranker")
        candidates = struct_retriever.search(query, top_k=RERANK_CANDIDATES)
        return [c for c, _ in reranker.rerank(query, candidates, top_n=10)]
    raise ValueError(f"unknown retriever: {retriever}")


def _chunk_stats(word_counts: list[int]) -> dict[str, float | int]:
    if not word_counts:
        return {"n_chunks": 0}
    return {
        "n_chunks": len(word_counts),
        "mean_words": round(mean([float(w) for w in word_counts]), 1),
        "median_words": round(statistics.median(word_counts), 1),
        "p90_words": round(percentile([float(w) for w in word_counts], 90.0), 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }


def _summarise(scores: list[QueryScore]) -> dict[str, float | int]:
    return {
        "n_queries": len(scores),
        "recall_at_1": round(mean([s.recall_at_1 for s in scores]), 4),
        "recall_at_5": round(mean([s.recall_at_5 for s in scores]), 4),
        "recall_at_10": round(mean([s.recall_at_10 for s in scores]), 4),
        "mrr": round(mean([s.mrr for s in scores]), 4),
        "recall_at_1500_words": round(mean([s.recall_at_1500w for s in scores]), 4),
        "recall_at_3000_words": round(mean([s.recall_at_3000w for s in scores]), 4),
    }


def run_factorial(
    *,
    split: str,
    chunkings: tuple[str, ...],
    retrievers: tuple[str, ...],
    sample_contracts: int | None,
    dump_per_query: bool = False,
) -> dict[str, Any]:
    """Run the factorial and return a JSON-serialisable report."""
    started = time.time()
    header = proof_header()
    print(f"branch={header['branch']} commit={header['commit']} python={header['python']}")
    print(f"clausewright={header['clausewright_path']}")

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

    # (chunking, retriever) -> scores
    cells: dict[tuple[str, str], list[QueryScore]] = {
        (c, r): [] for c in chunkings for r in retrievers
    }
    chunk_words: dict[str, list[int]] = {c: [] for c in chunkings}

    for index, title in enumerate(titles, start=1):
        contract_items = by_contract[title]
        print(f"[{index}/{len(titles)}] {title}", flush=True)
        doc = load_pdf(contract_items[0].pdf_path, doc_id=title)
        sections = parse_sections(doc)

        retrievers_by_chunking: dict[str, HybridRetriever] = {}
        for chunking in chunkings:
            chunks = build_chunks(chunking, title, doc, sections)
            chunk_words[chunking].extend(c.word_count for c in chunks)
            retrievers_by_chunking[chunking] = HybridRetriever.from_chunks(
                chunks, embedder=embedder
            )

        for item in contract_items:
            for chunking in chunkings:
                retriever = retrievers_by_chunking[chunking]
                for name in retrievers:
                    ranked = rank_for(name, item.question, retriever, reranker)
                    cells[(chunking, name)].append(score_query(item, ranked))

    summary: dict[str, dict[str, float | int]] = {}
    for chunking in chunkings:
        for name in retrievers:
            summary[f"{chunking}:{name}"] = _summarise(cells[(chunking, name)])

    # Comparisons that hold size and retriever fixed.
    comparisons: list[PairedDifference] = []
    for name in retrievers:
        if "struct" in chunkings and "win300" in chunkings:
            left = cells[("struct", name)]
            right = cells[("win300", name)]
            for metric in ("recall_at_5", "recall_at_1", "mrr", "recall_at_1500w"):
                pairs = [
                    (a.contract, getattr(a, metric), getattr(b, metric))
                    for a, b in zip(left, right)
                ]
                comparisons.append(
                    paired_bootstrap(
                        metric=metric,
                        left_label=f"struct:{name}",
                        right_label=f"win300:{name}",
                        pairs=pairs,
                        seed=SEED,
                    )
                )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "header": header,
        "config": {
            "split": split,
            "chunkings": list(chunkings),
            "retrievers": list(retrievers),
            "rerank_candidates": RERANK_CANDIDATES,
            "word_budgets": list(WORD_BUDGETS),
            "span_overlap_threshold": SPAN_OVERLAP_THRESHOLD,
            "seed": SEED,
        },
        "dataset_hash": eval_set.dataset_hash,
        "n_contracts": len(titles),
        "n_queries": sum(len(by_contract[t]) for t in titles),
        "chunk_stats": {c: _chunk_stats(chunk_words[c]) for c in chunkings},
        "metrics": summary,
        "comparisons": [c.as_dict() for c in comparisons],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if dump_per_query:
        report["per_query"] = {
            f"{chunking}:{name}": [
                {
                    "item_id": s.item_id,
                    "contract": s.contract,
                    "recall_at_1": s.recall_at_1,
                    "recall_at_5": s.recall_at_5,
                    "recall_at_10": s.recall_at_10,
                    "mrr": s.mrr,
                    "recall_at_1500w": s.recall_at_1500w,
                    "recall_at_3000w": s.recall_at_3000w,
                }
                for s in cells[(chunking, name)]
            ]
            for chunking in chunkings
            for name in retrievers
        }
    return report


def print_report(report: dict[str, Any]) -> None:
    """Print the tables a human reads, in the order they matter."""
    print("\n=== chunk size statistics (all contracts) ===")
    header = f"{'chunking':<10} {'chunks':>7} {'mean':>7} {'median':>7} {'p90':>7} {'max':>6}"
    print(header)
    print("-" * len(header))
    for name, stats in report["chunk_stats"].items():
        if not stats.get("n_chunks"):
            continue
        print(
            f"{name:<10} {stats['n_chunks']:>7} {stats['mean_words']:>7} "
            f"{stats['median_words']:>7} {stats['p90_words']:>7} {stats['max_words']:>6}"
        )

    print("\n=== retrieval metrics ===")
    head = (
        f"{'config':<22} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} "
        f"{'R@1500w':>8} {'R@3000w':>8} {'n':>4}"
    )
    print(head)
    print("-" * len(head))
    for name, metrics in report["metrics"].items():
        print(
            f"{name:<22} {metrics['recall_at_1']:>6.4f} {metrics['recall_at_5']:>6.4f} "
            f"{metrics['recall_at_10']:>6.4f} {metrics['mrr']:>6.4f} "
            f"{metrics['recall_at_1500_words']:>8.4f} "
            f"{metrics['recall_at_3000_words']:>8.4f} {metrics['n_queries']:>4}"
        )

    if report["comparisons"]:
        print("\n=== paired bootstrap, structure vs fixed window at 300 words ===")
        print("(10000 resamples by contract, seed 20261018)")
        for comp in report["comparisons"]:
            verdict = "excludes 0" if comp["excludes_zero"] else "includes 0"
            print(
                f"{comp['left']:<20} - {comp['right']:<20} {comp['metric']:<16} "
                f"{comp['point']:+.4f} [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}] {verdict}"
            )

    print(
        f"\n{report['n_queries']} queries, {report['n_contracts']} contracts, "
        f"{report['elapsed_seconds']}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--chunkings", default=",".join(CHUNKINGS), help="comma-separated subset"
    )
    parser.add_argument(
        "--retrievers", default=",".join(RETRIEVERS), help="comma-separated subset"
    )
    parser.add_argument("--sample", type=int, default=None, help="first N contracts")
    parser.add_argument(
        "--out", default="data/eval/factorial.json", help="JSON output path"
    )
    parser.add_argument(
        "--dump-per-query",
        action="store_true",
        help="include per-query scores so intervals can be recomputed later",
    )
    args = parser.parse_args(argv)

    chunkings = tuple(c.strip() for c in args.chunkings.split(",") if c.strip())
    retrievers = tuple(r.strip() for r in args.retrievers.split(",") if r.strip())
    unknown = [c for c in chunkings if c not in CHUNKINGS]
    if unknown:
        parser.error(f"unknown chunking(s): {unknown}")
    unknown_r = [r for r in retrievers if r not in RETRIEVERS]
    if unknown_r:
        parser.error(f"unknown retriever(s): {unknown_r}")

    report = run_factorial(
        split=args.split,
        chunkings=chunkings,
        retrievers=retrievers,
        sample_contracts=args.sample,
        dump_per_query=args.dump_per_query,
    )
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
