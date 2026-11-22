"""Measure whether localizing a citation beats citing the whole window.

For every dev query where a retrieved window contains the gold clause, this
compares three citations:

* ``window`` - the whole retrieved chunk (what the product cites today);
* ``bm25`` - the best sentence of that window by BM25 within the window;
* ``cross_encoder`` - the best sentence by cross-encoder score.

Reported per method: span hit rate (the cited text still contains at least
half of a gold span's tokens, the existing threshold), mean cited length in
characters, and character precision/recall against the gold span, with paired
bootstrap intervals by contract for before/after.

Only queries whose retrieved window already contains the gold are measured:
localization cannot recover a span that retrieval never returned, and
including those would make retrieval misses look like localization failures.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.metrics import SPAN_OVERLAP_THRESHOLD, overlap_ratio
from clausewright.eval.stats import PairedDifference, paired_bootstrap
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.retrieval.span_localizer import SpanLocalizer
from clausewright.schemas import Chunk

RERANK_CANDIDATES = 25
TOP_K = 10
SEED = CUAD_SMALL.seed


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace for character comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _char_scores(gold: str, candidate: str) -> tuple[float, float]:
    """Character precision and recall of *candidate* against the best gold."""
    gold_n = _norm(gold)
    cand_n = _norm(candidate)
    if not gold_n or not cand_n:
        return 0.0, 0.0
    matched = sum(
        block.size
        for block in SequenceMatcher(None, gold_n, cand_n, autojunk=False).get_matching_blocks()
    )
    return matched / len(cand_n), matched / len(gold_n)


def _best_char_scores(
    golds: list[str], candidate: str
) -> tuple[float, float]:
    """Best character precision/recall over the item's gold spans."""
    scored = [_char_scores(gold, candidate) for gold in golds]
    best = max(scored, key=lambda p: (p[1], p[0]))
    return best


def _span_hit(golds: list[str], candidate: str) -> bool:
    return any(overlap_ratio(g, candidate) >= SPAN_OVERLAP_THRESHOLD for g in golds)


def _gold_window(item: EvalItem, ranked: list[Chunk]) -> Chunk | None:
    """First retrieved chunk that already contains a gold span."""
    golds = list(item.gold_texts)
    for chunk in ranked:
        if _span_hit(golds, chunk.text):
            return chunk
    return None


def evaluate(*, split: str, sample_contracts: int | None) -> dict[str, Any]:
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
    localizers = {
        "bm25": SpanLocalizer(method="bm25", model=reranker._model),
        "cross_encoder": SpanLocalizer(method="cross_encoder", model=reranker._model),
    }

    rows: list[dict[str, Any]] = []
    skipped_no_window = 0

    for index, title in enumerate(titles, start=1):
        print(f"[{index}/{len(titles)}] {title}", flush=True)
        contract_items = by_contract[title]
        doc = load_pdf(contract_items[0].pdf_path, doc_id=title)
        chunks = chunk_sections(title, parse_sections(doc))
        retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)

        for item in contract_items:
            candidates = retriever.search(item.question, top_k=RERANK_CANDIDATES)
            ranked = [c for c, _ in reranker.rerank(item.question, candidates, top_n=TOP_K)]
            window = _gold_window(item, ranked)
            if window is None:
                skipped_no_window += 1
                continue

            golds = list(item.gold_texts)
            row: dict[str, Any] = {
                "item_id": item.item_id,
                "contract": title,
                "category": item.category,
                "window_chars": len(window.text),
            }
            precision, recall = _best_char_scores(golds, window.text)
            row["window"] = {
                "hit": _span_hit(golds, window.text),
                "chars": len(window.text),
                "precision": precision,
                "recall": recall,
            }
            for name, localizer in localizers.items():
                span = localizer.localize(item.question, window.text)
                cited = span.text if span is not None else window.text
                precision, recall = _best_char_scores(golds, cited)
                row[name] = {
                    "hit": _span_hit(golds, cited),
                    "chars": len(cited),
                    "precision": precision,
                    "recall": recall,
                    "localized": span is not None,
                }
            rows.append(row)

    def _mean(key: str, field: str) -> float:
        return sum(1.0 if r[key][field] else 0.0 for r in rows) / len(rows) if rows else 0.0

    def _mean_num(key: str, field: str) -> float:
        return sum(float(r[key][field]) for r in rows) / len(rows) if rows else 0.0

    summary: dict[str, dict[str, float]] = {}
    for key in ("window", "bm25", "cross_encoder"):
        summary[key] = {
            "span_hit_rate": round(_mean(key, "hit"), 4),
            "mean_chars": round(_mean_num(key, "chars"), 1),
            "char_precision": round(_mean_num(key, "precision"), 4),
            "char_recall": round(_mean_num(key, "recall"), 4),
        }

    comparisons: list[PairedDifference] = []
    for key in ("bm25", "cross_encoder"):
        for metric in ("precision", "recall", "hit"):
            pairs = [
                (
                    r["contract"],
                    float(r[key][metric]),
                    float(r["window"][metric]),
                )
                for r in rows
            ]
            comparisons.append(
                paired_bootstrap(
                    metric=f"char_{metric}" if metric != "hit" else "span_hit",
                    left_label=key,
                    right_label="window",
                    pairs=pairs,
                    seed=SEED,
                )
            )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "split": split,
            "top_k": TOP_K,
            "rerank_candidates": RERANK_CANDIDATES,
            "span_overlap_threshold": SPAN_OVERLAP_THRESHOLD,
            "seed": SEED,
        },
        "dataset_hash": eval_set.dataset_hash,
        "n_contracts": len(titles),
        "n_items_measured": len(rows),
        "n_items_without_gold_window": skipped_no_window,
        "summary": summary,
        "comparisons": [c.as_dict() for c in comparisons],
        "rows": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== cited span, before and after localization ===")
    head = f"{'citation':<16} {'span hit':>9} {'mean chars':>11} {'char P':>8} {'char R':>8}"
    print(head)
    print("-" * len(head))
    for key, metrics in report["summary"].items():
        print(
            f"{key:<16} {metrics['span_hit_rate']:>9.4f} {metrics['mean_chars']:>11.1f} "
            f"{metrics['char_precision']:>8.4f} {metrics['char_recall']:>8.4f}"
        )

    print("\n=== paired bootstrap vs citing the whole window ===")
    print("(10000 resamples by contract, seed 20261018)")
    head2 = f"{'citation':<16} {'metric':<14} {'point':>8} {'ci_low':>8} {'ci_high':>8}  verdict"
    print(head2)
    print("-" * len(head2))
    for comp in report["comparisons"]:
        verdict = "excludes 0" if comp["excludes_zero"] else "includes 0"
        print(
            f"{comp['left']:<16} {comp['metric']:<14} {comp['point']:>+8.4f} "
            f"{comp['ci_low']:>+8.4f} {comp['ci_high']:>+8.4f}  {verdict}"
        )

    print(
        f"\n{report['n_items_measured']} queries with a gold window, "
        f"{report['n_items_without_gold_window']} without, "
        f"{report['n_contracts']} contracts"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default="data/eval/span_localization.json")
    args = parser.parse_args(argv)

    report = evaluate(split=args.split, sample_contracts=args.sample)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
