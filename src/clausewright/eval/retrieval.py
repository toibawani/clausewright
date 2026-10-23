"""Retrieval evaluation harness across ablation configurations.

Evaluates 5 retrieval configurations on the evaluation set:
1. bm25: BM25 sparse search over structure-aware chunks
2. dense: BGE dense vector search over structure-aware chunks
3. hybrid: Hybrid RRF search over structure-aware chunks
4. hybrid_rerank: Hybrid RRF + CrossEncoder reranking over structure-aware chunks
5. fixed_window: Hybrid RRF over fixed-size window chunks (no section structure)

Computes Recall@1, Recall@3, Recall@5, Recall@10, and MRR.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.config import PATHS
from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.dataset import CUAD_SMALL, EvalItem, EvalSet, build_eval_set
from clausewright.eval.metrics import SPAN_OVERLAP_THRESHOLD, mean, mrr_score, recall_at_k
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk, ParsedDocument

CONFIGS = ("bm25", "dense", "hybrid", "hybrid_rerank", "fixed_window")


@dataclass(frozen=True)
class RetrievalScores:
    config: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int


def _build_indexes_for_contract(
    doc: ParsedDocument,
    doc_id: str,
    embedder: Embedder,
) -> tuple[HybridRetriever, HybridRetriever]:
    """Build structure-aware and fixed-window retrievers for a document."""
    # 1. Structure-aware chunks
    sections = parse_sections(doc)
    struct_chunks = chunk_sections(doc_id, sections)
    struct_retriever = HybridRetriever.from_chunks(struct_chunks, embedder=embedder)

    # 2. Fixed-window baseline chunks
    window_chunks = fixed_window_chunks(doc_id, doc)
    window_retriever = HybridRetriever.from_chunks(window_chunks, embedder=embedder)

    return struct_retriever, window_retriever


def evaluate_retrieval(
    eval_set: EvalSet,
    split: str = "dev",
    configs: tuple[str, ...] = CONFIGS,
    sample_contracts: int | None = None,
) -> dict[str, Any]:
    """Run retrieval evaluation across configurations on the given split."""
    items = [i for i in eval_set.by_split(split) if i.is_answerable]  # type: ignore[arg-type]
    if not items:
        raise ValueError(f"No answerable items in split '{split}'")

    # Group items by contract
    contracts: dict[str, list[EvalItem]] = {}
    for item in items:
        contracts.setdefault(item.contract_title, []).append(item)

    contract_titles = sorted(contracts.keys())
    if sample_contracts and sample_contracts < len(contract_titles):
        contract_titles = contract_titles[:sample_contracts]

    print(f"Evaluating {len(contract_titles)} contracts ({sum(len(contracts[t]) for t in contract_titles)} queries)...")

    embedder = Embedder()
    reranker = CrossEncoderReranker() if any("rerank" in c for c in configs) else None

    # Storage for per-config per-query metrics
    results_by_config: dict[str, dict[str, list[float]]] = {
        cfg: {"r1": [], "r3": [], "r5": [], "r10": [], "mrr": []}
        for cfg in configs
    }

    for c_idx, title in enumerate(contract_titles, start=1):
        contract_items = contracts[title]
        pdf_path = contract_items[0].pdf_path
        print(f"[{c_idx}/{len(contract_titles)}] Indexing {title}...")

        doc = load_pdf(pdf_path, doc_id=title)
        struct_retriever, window_retriever = _build_indexes_for_contract(doc, title, embedder)

        for item in contract_items:
            golds = item.gold_texts
            q = item.question

            # Retrieve top candidates for each requested config
            for cfg in configs:
                ranked_chunks: list[Chunk] = []
                if cfg == "bm25":
                    ranked_chunks = [c for c, _ in struct_retriever.search_sparse(q, top_k=10)]
                elif cfg == "dense":
                    ranked_chunks = [c for c, _ in struct_retriever.search_dense(q, top_k=10)]
                elif cfg == "hybrid":
                    ranked_chunks = [c for c, _ in struct_retriever.search(q, top_k=10)]
                elif cfg == "hybrid_rerank":
                    assert reranker is not None
                    cands = struct_retriever.search(q, top_k=25)
                    ranked_chunks = [c for c, _ in reranker.rerank(q, cands, top_n=10)]
                elif cfg == "fixed_window":
                    ranked_chunks = [c for c, _ in window_retriever.search(q, top_k=10)]
                else:
                    raise ValueError(f"Unknown config: {cfg}")

                ranked_texts = [c.text for c in ranked_chunks]
                r1 = recall_at_k(golds, ranked_texts, k=1, threshold=SPAN_OVERLAP_THRESHOLD)
                r3 = recall_at_k(golds, ranked_texts, k=3, threshold=SPAN_OVERLAP_THRESHOLD)
                r5 = recall_at_k(golds, ranked_texts, k=5, threshold=SPAN_OVERLAP_THRESHOLD)
                r10 = recall_at_k(golds, ranked_texts, k=10, threshold=SPAN_OVERLAP_THRESHOLD)
                mrr = mrr_score(golds, ranked_texts, threshold=SPAN_OVERLAP_THRESHOLD)

                res = results_by_config[cfg]
                res["r1"].append(r1)
                res["r3"].append(r3)
                res["r5"].append(r5)
                res["r10"].append(r10)
                res["mrr"].append(mrr)

    summary_scores: dict[str, Any] = {}
    for cfg in configs:
        res = results_by_config[cfg]
        n_q = len(res["r1"])
        scores = RetrievalScores(
            config=cfg,
            recall_at_1=round(mean(res["r1"]), 4),
            recall_at_3=round(mean(res["r3"]), 4),
            recall_at_5=round(mean(res["r5"]), 4),
            recall_at_10=round(mean(res["r10"]), 4),
            mrr=round(mean(res["mrr"]), 4),
            n_queries=n_q,
        )
        summary_scores[cfg] = asdict(scores)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "split": split,
        "span_overlap_threshold": SPAN_OVERLAP_THRESHOLD,
        "n_contracts_evaluated": len(contract_titles),
        "n_queries_evaluated": sum(len(contracts[t]) for t in contract_titles),
        "configs": summary_scores,
    }
    return report


def run_and_save_retrieval_eval(
    output_path: Path | None = None,
    split: str = "dev",
    configs: tuple[str, ...] = CONFIGS,
    sample_contracts: int | None = None,
) -> Path:
    out = output_path or (PATHS.root / "data" / "eval" / "retrieval_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    eval_set = build_eval_set(CUAD_SMALL)
    results = evaluate_retrieval(
        eval_set,
        split=split,
        configs=configs,
        sample_contracts=sample_contracts,
    )
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out
