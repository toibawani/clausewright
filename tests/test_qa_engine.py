"""Tests for clausewright.qa.engine — the full pipeline with fakes.

Fake retriever, reranker, and LLM; no ML models or network access anywhere.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from clausewright.qa.engine import AnswerEngine
from clausewright.qa.llm import LLMError
from clausewright.qa.schemas import AbstentionReason, Answer
from clausewright.schemas import Chunk

NOTICE_TEXT = (
    "The Provider shall maintain commercial general liability insurance with "
    "limits of not less than two million dollars per occurrence."
)


def _chunk(n: int, section: str = "8.4", text: str | None = None) -> tuple[Chunk, float]:
    t = text or NOTICE_TEXT
    c = Chunk(
        chunk_id=f"doc-{section}-{n}",
        doc_id="doc",
        section_number=section,
        chunk_index=n,
        text=t,
        page_range=(5, 5),
        word_count=len(t.split()),
    )
    return c, 1.0 - n * 0.1


class FakeRetriever:
    def __init__(self, results: list[tuple[Chunk, float]] | None = None) -> None:
        self._results = results if results is not None else [_chunk(0), _chunk(1)]
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        self.calls.append({"query": query, "top_k": top_k})
        return self._results


class FakeReranker:
    def __init__(self, results: list[tuple[Chunk, float]] | None = None) -> None:
        self._results = results

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], *, top_n: int | None = None
    ) -> list[tuple[Chunk, float]]:
        out = self._results if self._results is not None else candidates
        return out[:top_n] if top_n else out


class FakeLLM:
    def __init__(self, output: str) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._output


def _good_output(quote: str = "two million dollars per occurrence") -> str:
    return json.dumps(
        {
            "answerable": True,
            "reason_if_unanswerable": None,
            "claims": [
                {
                    "text": "Insurance coverage must be at least $2M per occurrence.",
                    "quote": quote,
                    "evidence_ids": [1],
                }
            ],
        }
    )


def _engine(
    llm_output: str = "",
    retriever: Any = None,
    reranker: Any = None,
    llm: Any = None,
    **kwargs: Any,
) -> tuple[AnswerEngine, Any, Any, Any]:
    r = retriever if retriever is not None else FakeRetriever()
    rr = reranker if reranker is not None else FakeReranker()
    l = llm if llm is not None else FakeLLM(_good_output() if llm_output == "" else llm_output)
    return AnswerEngine(r, rr, l, **kwargs), r, rr, l


class TestAnsweredPath:
    def test_returns_answer_with_verified_citations(self) -> None:
        engine, _, _, llm = _engine()
        result = engine.answer("What insurance is required?")
        assert isinstance(result, Answer)
        assert result.status == "answered"
        assert "$2M" in result.text
        assert result.model == "fake-model"
        claim = result.claims[0]
        assert claim.citations[0].section_number == "8.4"
        assert claim.citations[0].doc_id == "doc"
        assert "not legal advice" in result.notice
        # System/user separation: untrusted-data rule lives in the system half.
        assert "untrusted" in llm.calls[0]["system"]
        assert "<<<EVIDENCE 1" in llm.calls[0]["user"]

    def test_question_is_passed_to_retriever(self) -> None:
        engine, retriever, _, _ = _engine()
        engine.answer("What insurance is required?")
        assert retriever.calls[0]["query"] == "What insurance is required?"


class TestAbstentions:
    def test_no_candidates_from_retriever(self) -> None:
        engine, _, _, llm = _engine(retriever=FakeRetriever(results=[]))
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert result.reason == AbstentionReason.NO_EVIDENCE
        assert llm.calls == []  # no LLM call when there is nothing to send

    def test_weak_evidence_gate_blocks_llm_call(self) -> None:
        weak = [_chunk(0)]
        engine, _, _, llm = _engine(reranker=FakeReranker(results=weak), min_rerank_score=1.5)
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert result.reason == AbstentionReason.WEAK_EVIDENCE
        assert "below the acceptance floor" in result.detail
        assert result.searched_sections == ("8.4",)
        assert llm.calls == []

    def test_invalid_llm_output_abstains(self) -> None:
        engine, _, _, _ = _engine(llm_output="Here is your answer, trust me.")
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert result.reason == AbstentionReason.INVALID_OUTPUT

    def test_model_says_unanswerable(self) -> None:
        raw = json.dumps(
            {"answerable": False, "reason_if_unanswerable": "no insurance clause", "claims": []}
        )
        engine, _, _, _ = _engine(llm_output=raw)
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert result.reason == AbstentionReason.NO_EVIDENCE
        assert "no insurance clause" in result.detail

    def test_all_claims_fail_verification(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Liability is capped at ten million.",
                        "quote": "ten million dollars",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        engine, _, _, _ = _engine(llm_output=raw)
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert result.reason == AbstentionReason.NO_VERIFIABLE_CLAIMS
        assert "quote_not_found" in result.detail

    def test_llm_error_propagates(self) -> None:
        engine, _, _, _ = _engine(llm=_RaisingLLM())
        with pytest.raises(LLMError):
            engine.answer("q?")

    def test_abstention_states_sections_searched(self) -> None:
        raw = json.dumps({"answerable": False, "reason_if_unanswerable": "x", "claims": []})
        engine, _, _, _ = _engine(llm_output=raw)
        result = engine.answer("q?")
        assert result.status == "abstained"
        assert "8.4" in result.detail


class _RaisingLLM:
    @property
    def model_name(self) -> str:
        return "boom"

    def complete(self, *, system: str, user: str) -> str:
        raise LLMError("model down")


class TestEngineConfig:
    def test_invalid_top_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            _engine(top_k=0)

    def test_invalid_keep_rejected(self) -> None:
        with pytest.raises(ValueError, match="keep_after_rerank"):
            _engine(keep_after_rerank=0)

    def test_blank_question_rejected(self) -> None:
        engine, _, _, _ = _engine()
        with pytest.raises(ValueError, match="question"):
            engine.answer("   ")

