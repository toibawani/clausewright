"""Run the answer pipeline end to end with a fake model and print its table.

No provider call is made. The retriever, the reranker, and the model are stubs,
so this exercises the parts that do not need a network: prompt assembly, the
verifier, abstention, the metrics, and the provenance block. Answer quality is
**not** measured here and the printed table says so.

    python scripts/part_c_dry_run.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clausewright.eval.answer_metrics import (
    answer_quality_report,
    faithfulness_rate,
    qualifier_guard_rate,
)
from clausewright.eval.dataset import CUAD_SMALL, EvalItem, build_eval_set
from clausewright.eval.part_c import build_answer_eval_set, estimate_answer_cost
from clausewright.eval.provenance import provenance
from clausewright.qa.engine import AnswerEngine
from clausewright.qa.schemas import Answer, AnswerResult, EvidenceItem
from clausewright.schemas import Chunk


class StubRetriever:
    """Returns one chunk for answerable items and nothing for absent ones."""

    def __init__(self, payload: dict[str, list[tuple[Chunk, float]]]) -> None:
        self._payload = payload

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        return self._payload.get(query, [])[:top_k]


class StubReranker:
    """Keeps the retriever's order; scores are passed through."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_n: int = 5,
    ) -> list[tuple[Chunk, float]]:
        return list(candidates)[:top_n]


class FakeLLM:
    """Returns a canned claim that quotes the first sentence of the evidence."""

    model_name = "fake-llm-dry-run"

    def __init__(self, quotes: dict[str, str]) -> None:
        self._quotes = quotes

    def complete(self, *, system: str, user: str) -> str:
        question = next(
            (line[len("Question: ") :] for line in user.splitlines() if line.startswith("Question: ")),
            "",
        )
        quote = self._quotes.get(question, "")
        if not quote:
            return json.dumps(
                {
                    "answerable": False,
                    "reason_if_unanswerable": "the evidence does not contain this clause",
                    "claims": [],
                }
            )
        return json.dumps(
            {
                "answerable": True,
                "reason_if_unanswerable": None,
                "claims": [
                    {
                        "text": "The contract states the point quoted above.",
                        "quote": quote,
                        "evidence_ids": [1],
                    }
                ],
            }
        )


def _chunk_for(item: EvalItem) -> Chunk:
    text = item.gold_texts[0] if item.gold_texts else "No clause of this kind appears in the contract."
    return Chunk(
        chunk_id=f"{item.contract_title}-1-0",
        doc_id=item.contract_title,
        section_number="1",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=max(1, len(text.split())),
    )


def _first_sentence(text: str, limit: int = 120) -> str:
    sentence = text.split(". ")[0]
    return sentence[:limit]


def run(limit: int, evidence_blocks: int = 5) -> dict[str, Any]:
    eval_set = build_eval_set(CUAD_SMALL)
    items = build_answer_eval_set(eval_set)[:limit]

    payload: dict[str, list[tuple[Chunk, float]]] = {}
    quotes: dict[str, str] = {}
    for item in items:
        if not item.is_answerable:
            payload[item.question] = []
            continue
        chunk = _chunk_for(item)
        payload[item.question] = [(chunk, 1.0)]
        quotes[item.question] = _first_sentence(chunk.text)

    engine = AnswerEngine(
        retriever=StubRetriever(payload),
        reranker=StubReranker(),
        llm=FakeLLM(quotes),
        top_k=5,
        keep_after_rerank=1,
    )

    results: list[AnswerResult] = [engine.answer(item.question) for item in items]
    true_absent = [item.clause_absent for item in items]

    report = answer_quality_report(results, true_absent=true_absent)

    claims_and_quotes = [
        (claim.text, claim.quote)
        for result in results
        if isinstance(result, Answer)
        for claim in result.claims
    ]
    report["qualifier_guard_rate"] = _round(qualifier_guard_rate(claims_and_quotes))
    report["qualifier_violation_rate"] = _round(1.0 - qualifier_guard_rate(claims_and_quotes))
    report["faithfulness_rate"] = _round(faithfulness_rate(claims_and_quotes))

    # The engine sends ``keep_after_rerank`` blocks, so the estimate uses that
    # many copies: the prompt size, not the number of distinct chunks, is what
    # costs tokens.
    pairs: list[tuple[str, list[EvidenceItem]]] = [
        (
            item.question,
            [
                EvidenceItem(evidence_id=index, chunk=chunk, score=score)
                for index in range(1, evidence_blocks + 1)
                for chunk, score in payload[item.question]
            ],
        )
        for item in items
        if payload[item.question]
    ]

    return {
        "dry_run": True,
        "provenance": provenance(),
        "config": {"limit": limit, "split": "dev", "llm": "fake-llm-dry-run"},
        "metrics": report,
        "cost": estimate_answer_cost(pairs),
    }


def _round(value: float, ndigits: int = 4) -> Any:
    if math.isnan(value):
        return None
    return round(value, ndigits)


def render_table(result: dict[str, Any]) -> str:
    """Markdown table of the dry-run metrics and the call estimate."""
    metrics = result["metrics"]
    cost = result["cost"]
    lines = [
        "**Dry run: stub retriever, stub reranker, fake model. Not a measurement.**",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in (
        "n_questions",
        "n_answered",
        "n_abstained",
        "abstention_precision",
        "abstention_recall",
        "false_abstention_rate",
        "qualifier_guard_rate",
        "qualifier_violation_rate",
        "faithfulness_rate",
    ):
        if key in metrics:
            lines.append(f"| `{key}` | {metrics[key]} |")
    lines.append("")
    lines.append(f"Provider calls for {cost['questions']} questions: {cost['total_calls']}")
    lines.append(f"Prompt tokens (estimate): {cost['prompt_tokens_estimate']}")
    lines.append(f"Completion tokens (estimate): {cost['completion_tokens_estimate']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--evidence", type=int, default=5, help="evidence blocks per prompt")
    parser.add_argument("--out", type=Path, default=Path("data/eval/part_c_dry_run.json"))
    args = parser.parse_args(argv)

    result = run(args.limit, args.evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(render_table(result))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
