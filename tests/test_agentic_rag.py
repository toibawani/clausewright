"""Tests for clausewright.generation.agentic_rag (CRAG pipeline).

All external dependencies (Anthropic client, HybridRetriever, CrossEncoderReranker)
are fully mocked so tests run without network access or model weights.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from clausewright.generation.agentic_rag import AgenticRAG, RAGResponse, _ABSTAIN_SENTINEL
from clausewright.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8


def _chunk(n: int, section: str = "1", text: str | None = None) -> Chunk:
    t = text or f"Chunk {n}: The indemnifying party shall hold harmless all officers."
    return Chunk(
        chunk_id=f"doc-{section}-{n}",
        doc_id="doc",
        section_number=section,
        chunk_index=n,
        text=t,
        page_range=(1, 1),
        word_count=len(t.split()),
    )


def _chunks(count: int) -> list[Chunk]:
    return [_chunk(i) for i in range(count)]


def _mock_llm_message(text: str) -> MagicMock:
    """Build a fake anthropic.Message."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _build_pipeline(
    *,
    retriever_chunks: list[Chunk] | None = None,
    reranker_chunks: list[Chunk] | None = None,
    grader_verdicts: list[str] | None = None,
    generator_answer: str = "The indemnification obligation is in [Section 1].",
    variant_json: str | None = None,
    max_retries: int = 2,
    relevance_threshold: float = 0.4,
) -> tuple[AgenticRAG, MagicMock]:
    """Build an AgenticRAG with all dependencies mocked.

    side_effect ordering (matches actual pipeline call order):
      1. N grader calls (one per reranked chunk, per retrieval attempt)
      2. Variant generation call (only on retry due to low relevance)
      3. Generator call (final answer)

    On the *happy path* (all relevant, no retry) the call order is:
      [grade_0, grade_1, ..., grade_N, generate_answer]
    so we place the generator answer immediately after the grader verdicts.
    Variant messages are inserted only when the test supplies irrelevant verdicts.
    """
    r_chunks = retriever_chunks if retriever_chunks is not None else _chunks(3)
    rr_chunks = reranker_chunks if reranker_chunks is not None else r_chunks

    retriever = MagicMock()
    retriever.search.return_value = [(c, 0.9) for c in r_chunks]

    reranker = MagicMock()
    reranker.rerank.return_value = [(c, 0.8) for c in rr_chunks]

    verdicts = grader_verdicts if grader_verdicts is not None else ["relevant"] * len(rr_chunks)
    variant_text = variant_json or json.dumps(
        ["variant A", "variant B", "variant C"]
    )

    # Build side_effect sequence:
    # grader messages first, then interleave variant + grader for retries,
    # finally the generator answer.
    #
    # We partition verdicts into rounds: first len(rr_chunks) per attempt.
    # If grader_verdicts is longer than one round it means retries are expected;
    # insert a variant message between rounds.
    n_per_round = len(rr_chunks) if rr_chunks else 1
    side_effects: list[MagicMock] = []
    idx = 0
    round_num = 0
    while idx < len(verdicts):
        if round_num > 0:
            # Variant call that triggers the new round
            side_effects.append(_mock_llm_message(variant_text))
        for v in verdicts[idx : idx + n_per_round]:
            side_effects.append(_mock_llm_message(v))
        idx += n_per_round
        round_num += 1

    # Generator answer — appended after all grading rounds
    side_effects.append(_mock_llm_message(generator_answer))
    # Extra fallbacks
    for _ in range(3):
        side_effects.append(_mock_llm_message(generator_answer))

    client = MagicMock()
    client.messages.create.side_effect = side_effects

    pipeline = AgenticRAG(
        client=client,
        retriever=retriever,
        reranker=reranker,
        max_retries=max_retries,
        relevance_threshold=relevance_threshold,
    )
    return pipeline, client


# ---------------------------------------------------------------------------
# RAGResponse schema
# ---------------------------------------------------------------------------


def test_ragresponse_default_not_abstained() -> None:
    r = RAGResponse(question="q?")
    assert not r.abstained
    assert r.answer is None
    assert r.citations == []


def test_ragresponse_abstained_flag() -> None:
    r = RAGResponse(question="q?", abstained=True, abstain_reason="no evidence")
    assert r.abstained
    assert r.abstain_reason == "no evidence"


# ---------------------------------------------------------------------------
# AgenticRAG.answer — happy path
# ---------------------------------------------------------------------------


def test_answer_returns_ragresponse() -> None:
    pipeline, _ = _build_pipeline()
    result = pipeline.answer("What are the indemnification obligations?")
    assert isinstance(result, RAGResponse)


def test_answer_not_abstained_when_relevant_context() -> None:
    pipeline, _ = _build_pipeline()
    result = pipeline.answer("What are the payment terms?")
    assert not result.abstained
    assert result.answer is not None


def test_answer_preserves_question() -> None:
    q = "What are the termination provisions?"
    pipeline, _ = _build_pipeline()
    result = pipeline.answer(q)
    assert result.question == q


def test_answer_extracts_citations() -> None:
    pipeline, _ = _build_pipeline(
        generator_answer="The clause in [Section 3.2] and [Section 5] applies."
    )
    result = pipeline.answer("liability caps?")
    assert "3.2" in result.citations
    assert "5" in result.citations


def test_answer_retrieval_attempts_is_one_on_first_success() -> None:
    pipeline, _ = _build_pipeline()
    result = pipeline.answer("payment terms?")
    assert result.retrieval_attempts == 1


def test_answer_chunks_used_populated() -> None:
    chunks = _chunks(3)
    pipeline, _ = _build_pipeline(reranker_chunks=chunks)
    result = pipeline.answer("indemnification?")
    assert len(result.chunks_used) == 3


# ---------------------------------------------------------------------------
# AgenticRAG.answer — abstain paths
# ---------------------------------------------------------------------------


def test_answer_abstains_on_insufficient_evidence_sentinel() -> None:
    pipeline, _ = _build_pipeline(generator_answer=_ABSTAIN_SENTINEL)
    result = pipeline.answer("When does the contract end?")
    assert result.abstained
    assert result.answer is None


def test_answer_abstains_when_no_chunks_retrieved() -> None:
    pipeline, _ = _build_pipeline(
        retriever_chunks=[],
        reranker_chunks=[],
        max_retries=0,
    )
    result = pipeline.answer("payment terms?")
    assert result.abstained


def test_answer_abstain_reason_is_set_on_no_chunks() -> None:
    pipeline, _ = _build_pipeline(retriever_chunks=[], reranker_chunks=[], max_retries=0)
    result = pipeline.answer("q?")
    assert result.abstain_reason is not None
    assert len(result.abstain_reason) > 0


def test_answer_abstained_response_has_no_answer() -> None:
    pipeline, _ = _build_pipeline(generator_answer=_ABSTAIN_SENTINEL)
    result = pipeline.answer("q?")
    assert result.answer is None


# ---------------------------------------------------------------------------
# AgenticRAG.answer — CRAG retry loop
# ---------------------------------------------------------------------------


def test_answer_retries_on_low_relevance() -> None:
    """With all chunks graded irrelevant, pipeline should retry."""
    irrelevant_verdicts = ["irrelevant"] * 3  # 3 chunks, all irrelevant
    # Second attempt: all relevant
    all_verdicts = irrelevant_verdicts + ["relevant"] * 3

    pipeline, client = _build_pipeline(
        grader_verdicts=all_verdicts,
        max_retries=2,
        relevance_threshold=0.5,
    )
    result = pipeline.answer("What are the indemnification obligations?")
    # Should have made at least 2 retrieval attempts
    assert result.retrieval_attempts >= 2


def test_answer_uses_query_variant_on_retry() -> None:
    """After low relevance, the pipeline should request query variants."""
    irrelevant_verdicts = ["irrelevant"] * 3
    # Second attempt: relevant
    all_verdicts = irrelevant_verdicts + ["relevant"] * 3
    pipeline, client = _build_pipeline(
        grader_verdicts=all_verdicts,
        max_retries=2,
        relevance_threshold=0.5,
    )
    pipeline.answer("payment terms?")
    # Variant generation call should appear among client.messages.create calls
    # (calls after grader calls)
    total_calls = client.messages.create.call_count
    # Grader (3) + variant call (1) + generator (1) = 5 minimum
    assert total_calls >= 5


def test_answer_stops_after_max_retries() -> None:
    """Even with all irrelevant grades, must stop at max_retries."""
    all_irrelevant = ["irrelevant"] * 6  # enough for 2 retrieval rounds
    pipeline, _ = _build_pipeline(
        grader_verdicts=all_irrelevant,
        max_retries=1,
        relevance_threshold=1.0,  # impossible to meet
    )
    result = pipeline.answer("payment terms?")
    assert result.retrieval_attempts <= 2  # 1 base + 1 retry


# ---------------------------------------------------------------------------
# AgenticRAG.answer — error handling
# ---------------------------------------------------------------------------


def test_answer_empty_question_raises() -> None:
    pipeline, _ = _build_pipeline()
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.answer("   ")


def test_answer_abstains_on_generator_exception() -> None:
    r_chunks = _chunks(3)
    retriever = MagicMock()
    retriever.search.return_value = [(c, 0.9) for c in r_chunks]
    reranker = MagicMock()
    reranker.rerank.return_value = [(c, 0.8) for c in r_chunks]

    client = MagicMock()
    # Grader calls succeed (relevant), generator call raises
    grader_msg = _mock_llm_message("relevant")
    client.messages.create.side_effect = [grader_msg, grader_msg, grader_msg,
                                           RuntimeError("LLM down")]

    pipeline = AgenticRAG(
        client=client,
        retriever=retriever,
        reranker=reranker,
        max_retries=0,
    )
    result = pipeline.answer("liability?")
    assert result.abstained


# ---------------------------------------------------------------------------
# _extract_citations (static helper)
# ---------------------------------------------------------------------------


def test_extract_citations_finds_section_labels() -> None:
    text = "See [Section 3.2] and also refer to [Section 10.1(b)]."
    cits = AgenticRAG._extract_citations(text)
    assert "3.2" in cits
    assert "10.1(b)" in cits


def test_extract_citations_empty_on_no_labels() -> None:
    assert AgenticRAG._extract_citations("No citations here.") == []


def test_extract_citations_multiple_same_section() -> None:
    text = "[Section 5] is referenced twice. [Section 5] again."
    cits = AgenticRAG._extract_citations(text)
    assert cits.count("5") == 2
