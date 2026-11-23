"""Classify date-category retrieval misses as extraction, chunking, or ranking.

Agreement Date and Effective Date dominate the worst dev queries: the gold span
is usually a date string of a few tokens, and the retrieved sections are real
sections that simply do not contain it. Guessing at the cause has produced two
rounds of wrong hypotheses, so this script produces the evidence instead.

For each miss it reports, without printing any contract text:

* whether the gold span's tokens appear in the parsed document at all — if
  they do not, the failure is PDF extraction, not retrieval;
* how many gold tokens survive extraction, as a fraction;
* the sections of the top three retrieved chunks, and whether the gold span
  sits in the same section as any of them;
* the rank of the first chunk that does contain the gold span, if there is one.

A gold span that is present in the document but absent from every chunk means
chunk assignment dropped or split it. A span that is in a chunk but ranked low
means the ranker is at fault. The three outcomes need different fixes, which is
the whole point of separating them.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.metrics import SPAN_OVERLAP_THRESHOLD, overlap_ratio
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever

DATE_CATEGORIES = ("Agreement Date", "Effective Date")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _present_in(text_tokens: list[str], gold: str) -> bool:
    """True when the gold span appears as a contiguous token run."""
    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return False
    n = len(gold_tokens)
    for start in range(len(text_tokens) - n + 1):
        if text_tokens[start : start + n] == gold_tokens:
            return True
    return False


def classify(
    *,
    split: str,
    limit: int,
    sample_contracts: int | None,
) -> list[dict[str, Any]]:
    eval_set = build_eval_set(CUAD_SMALL)
    items = [
        i
        for i in eval_set.by_split(split)  # type: ignore[arg-type]
        if i.is_answerable and i.category in DATE_CATEGORIES
    ]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_contract[item.contract_title].append(item)
    titles = sorted(by_contract)
    if sample_contracts is not None:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    rows: list[dict[str, Any]] = []

    for title in titles:
        doc = load_pdf(by_contract[title][0].pdf_path, doc_id=title)
        sections = parse_sections(doc)
        chunks = chunk_sections(title, sections)
        retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)
        doc_tokens = _tokens(" ".join(s.text for s in sections))

        for item in by_contract[title]:
            golds = list(item.gold_texts)
            ranked = [c for c, _ in retriever.search(item.question, top_k=10)]
            top_texts = [c.text for c in ranked]
            hit = any(
                overlap_ratio(g, t) >= SPAN_OVERLAP_THRESHOLD
                for g in golds
                for t in top_texts
            )
            if hit:
                continue

            for gold in golds:
                in_doc = _present_in(doc_tokens, gold)
                in_any_chunk = any(_present_in(_tokens(c.text), gold) for c in chunks)
                containing = [
                    (rank, c.section_number)
                    for rank, c in enumerate(ranked, start=1)
                    if _present_in(_tokens(c.text), gold)
                ]
                if not in_doc:
                    cause = "extraction"
                elif not in_any_chunk:
                    cause = "chunk-assignment"
                elif not containing:
                    cause = "ranking"
                else:
                    cause = "span-overlap"
                rows.append(
                    {
                        "item_id": item.item_id,
                        "contract": title,
                        "category": item.category,
                        "gold_tokens": len(_tokens(gold)),
                        "gold_in_parsed_doc": in_doc,
                        "gold_in_any_chunk": in_any_chunk,
                        "top3_sections": [c.section_number for c in ranked[:3]],
                        "cause": cause,
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default="data/eval/date_misses.json")
    args = parser.parse_args(argv)

    rows = classify(split=args.split, limit=args.limit, sample_contracts=args.sample)

    print(f"{len(rows)} date-category misses on {args.split}")
    header = f"{'cause':<17} {'tok':>3} {'in_doc':>6} {'in_chunk':>8}  {'top3 sections'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['cause']:<17} {row['gold_tokens']:>3} "
            f"{row['gold_in_parsed_doc']!s:>6} {row['gold_in_any_chunk']!s:>8}  "
            f"{row['top3_sections']}"
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["cause"]] = counts.get(row["cause"], 0) + 1
    print("\ncause counts:", json.dumps(counts, sort_keys=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "cases": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
