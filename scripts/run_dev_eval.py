"""Run the dev retrieval evaluation for every config and write per-config JSON.

Run from a worktree (or clone) so that ``clausewright`` resolves to that
commit's source via ``PYTHONPATH=<worktree>/src``.  Each output file is
``results/<config>-<short-hash>.json`` and contains the commit hash, Python
version, config, seed, and per-query outcomes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.dataset import CUAD_SMALL, build_eval_set
from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    mean,
    mrr_score,
    recall_at_k,
    span_hit,
)
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker

CONFIGS = ("bm25", "dense", "hybrid", "hybrid_rerank", "fixed_window")

# 1500 tokens ~= 2000 words at the project's 0.75 tokens/word estimate.
TOKEN_BUDGET = 1500
WORDS_PER_TOKEN = 0.75
WORD_BUDGET = int(TOKEN_BUDGET / WORDS_PER_TOKEN)


def git_short_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def char_precision_at_budget(
    gold_spans: list[str], ranked_texts: list[str], ranked_words: list[int]
) -> float:
    """Gold characters covered by chunks within the token budget / retrieved chars."""
    selected: list[str] = []
    selected_words = 0
    for text, words in zip(ranked_texts, ranked_words):
        if selected and selected_words + words > WORD_BUDGET:
            break
        selected.append(text)
        selected_words += words
        if selected_words >= WORD_BUDGET:
            break
    if not selected:
        if ranked_texts:
            selected = [ranked_texts[0]]
        else:
            return 0.0

    retrieved_chars = sum(len(t) for t in selected)
    if retrieved_chars == 0:
        return 0.0

    covered_gold_chars = 0
    for gold in gold_spans:
        if any(gold in text or text in gold for text in selected):
            # Count this gold span once if any selected chunk hits it.
            covered_gold_chars += len(gold)
    covered_gold_chars = min(covered_gold_chars, retrieved_chars)
    return covered_gold_chars / retrieved_chars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--config", default="all")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path.cwd()
    sha = git_short_sha(root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config == "all":
        configs = CONFIGS
    else:
        configs = tuple(c.strip() for c in args.config.split(","))

    eval_set = build_eval_set(CUAD_SMALL)
    items = [i for i in eval_set.by_split(args.split) if i.is_answerable]  # type: ignore[arg-type]
    contracts: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        contracts[item.contract_title].append(item)
    contract_titles = sorted(contracts.keys())

    embedder = Embedder()
    reranker = CrossEncoderReranker() if any("rerank" in c for c in configs) else None

    # per config -> per query records
    per_config_queries: dict[str, list[dict[str, Any]]] = {cfg: [] for cfg in configs}
    # per config -> chunk word counts (for chunk size stats)
    per_config_chunk_words: dict[str, list[int]] = {cfg: [] for cfg in configs}

    for c_idx, title in enumerate(contract_titles, start=1):
        contract_items = contracts[title]
        pdf_path = contract_items[0].pdf_path
        print(f"[{c_idx}/{len(contract_titles)}] Indexing {title}...", flush=True)

        doc = load_pdf(pdf_path, doc_id=title)
        sections = parse_sections(doc)
        struct_chunks = chunk_sections(title, sections)
        window_chunks = fixed_window_chunks(title, doc)

        struct_retriever = HybridRetriever.from_chunks(struct_chunks, embedder=embedder)
        window_retriever = HybridRetriever.from_chunks(window_chunks, embedder=embedder)

        for cfg in configs:
            if cfg == "fixed_window":
                per_config_chunk_words[cfg].extend(c.word_count for c in window_chunks)
            else:
                per_config_chunk_words[cfg].extend(c.word_count for c in struct_chunks)

        for item in contract_items:
            golds = list(item.gold_texts)
            q = item.question

            for cfg in configs:
                if cfg == "bm25":
                    ranked = [c for c, _ in struct_retriever.search_sparse(q, top_k=10)]
                elif cfg == "dense":
                    ranked = [c for c, _ in struct_retriever.search_dense(q, top_k=10)]
                elif cfg == "hybrid":
                    ranked = [c for c, _ in struct_retriever.search(q, top_k=10)]
                elif cfg == "hybrid_rerank":
                    assert reranker is not None
                    cands = struct_retriever.search(q, top_k=25)
                    ranked = [c for c, _ in reranker.rerank(q, cands, top_n=10)]
                elif cfg == "fixed_window":
                    ranked = [c for c, _ in window_retriever.search(q, top_k=10)]
                else:
                    raise ValueError(cfg)

                ranked_texts = [c.text for c in ranked]
                ranked_words = [c.word_count for c in ranked]
                r1 = recall_at_k(golds, ranked_texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD)
                r3 = recall_at_k(golds, ranked_texts, k=3, threshold=SPAN_OVERLAP_THRESHOLD)
                r5 = recall_at_k(golds, ranked_texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD)
                r10 = recall_at_k(golds, ranked_texts, k=10, threshold=SPAN_OVERLAP_THRESHOLD)
                mrr = mrr_score(golds, ranked_texts, threshold=SPAN_OVERLAP_THRESHOLD)
                fhr = None
                for rank, t in enumerate(ranked_texts, start=1):
                    if any(
                        span_hit(g, t, SPAN_OVERLAP_THRESHOLD)
                        for g in golds
                    ):
                        fhr = rank
                        break

                cp = char_precision_at_budget(golds, ranked_texts, ranked_words)

                per_config_queries[cfg].append(
                    {
                        "item_id": item.item_id,
                        "contract_title": title,
                        "category": item.category,
                        "n_gold_spans": len(golds),
                        "recall_at_1": r1,
                        "recall_at_3": r3,
                        "recall_at_5": r5,
                        "recall_at_10": r10,
                        "mrr": mrr,
                        "first_hit_rank": fhr,
                        "char_precision_at_1500_tokens": cp,
                        "top_sections": [c.section_number for c in ranked[:5]],
                        "top_chunk_ids": [c.chunk_id for c in ranked[:5]],
                    }
                )

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    for cfg in configs:
        qs = per_config_queries[cfg]
        r1s = [q["recall_at_1"] for q in qs]
        r3s = [q["recall_at_3"] for q in qs]
        r5s = [q["recall_at_5"] for q in qs]
        r10s = [q["recall_at_10"] for q in qs]
        mrrs = [q["mrr"] for q in qs]
        cps = [q["char_precision_at_1500_tokens"] for q in qs]
        words = per_config_chunk_words[cfg]
        record = {
            "commit": sha,
            "python_version": python_version,
            "config": cfg,
            "seed": CUAD_SMALL.seed,
            "split": args.split,
            "timestamp": datetime.now(UTC).isoformat(),
            "n_contracts": len(contract_titles),
            "n_queries": len(qs),
            "metrics": {
                "recall_at_1": round(mean(r1s), 4),
                "recall_at_3": round(mean(r3s), 4),
                "recall_at_5": round(mean(r5s), 4),
                "recall_at_10": round(mean(r10s), 4),
                "mrr": round(mean(mrrs), 4),
                "char_precision_at_1500_tokens": round(mean(cps), 4),
            },
            "chunk_stats": {
                "n_chunks": len(words),
                "mean_chunk_words": round(mean(words), 2) if words else 0.0,
                "median_chunk_words": round(statistics.median(words), 2) if words else 0.0,
            },
            "per_query": qs,
        }
        out_path = out_dir / f"{cfg}-{sha}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
