"""Tests for clausewright.generation.query_optimizer.

The Anthropic client is fully mocked so tests run without an API key or
network access.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from clausewright.generation.query_optimizer import QueryOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str) -> MagicMock:
    """Build a fake anthropic.Message with a single text content block."""
    block = MagicMock()
    block.text = text
    message = MagicMock()
    message.content = [block]
    return message


def _make_client(response_text: str) -> MagicMock:
    """Return a fake Anthropic client whose messages.create returns *response_text*."""
    client = MagicMock()
    client.messages.create.return_value = _make_response(response_text)
    return client


def _make_optimizer(response_text: str = "dummy", model: str = "mock-model") -> QueryOptimizer:
    return QueryOptimizer(client=_make_client(response_text), model=model)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_optimizer_stores_model() -> None:
    opt = _make_optimizer(model="claude-haiku-4-5")
    assert opt.model == "claude-haiku-4-5"


def test_optimizer_model_property() -> None:
    opt = QueryOptimizer(client=MagicMock(), model="my-model")
    assert opt.model == "my-model"


# ---------------------------------------------------------------------------
# generate_hyde
# ---------------------------------------------------------------------------


def test_hyde_returns_string() -> None:
    opt = _make_optimizer("The indemnification clause requires the party to hold harmless...")
    result = opt.generate_hyde("What are the indemnification obligations?")
    assert isinstance(result, str)


def test_hyde_returns_llm_text() -> None:
    expected = "The payment terms require net-30 settlement of all invoices."
    opt = _make_optimizer(expected)
    result = opt.generate_hyde("What are the payment terms?")
    assert result == expected


def test_hyde_strips_whitespace() -> None:
    opt = _make_optimizer("  Some clause text.  ")
    result = opt.generate_hyde("question?")
    assert result == "Some clause text."


def test_hyde_empty_question_raises() -> None:
    opt = _make_optimizer("anything")
    with pytest.raises(ValueError, match="non-empty"):
        opt.generate_hyde("   ")


def test_hyde_calls_client_once() -> None:
    client = _make_client("answer text")
    opt = QueryOptimizer(client=client, model="mock")
    opt.generate_hyde("payment terms?")
    client.messages.create.assert_called_once()


def test_hyde_passes_question_to_client() -> None:
    client = _make_client("answer")
    opt = QueryOptimizer(client=client, model="mock")
    question = "What is the termination clause?"
    opt.generate_hyde(question)
    call_kwargs: dict[str, Any] = client.messages.create.call_args[1]
    assert call_kwargs["messages"][0]["content"] == question


def test_hyde_fallback_on_empty_llm_response() -> None:
    """If LLM returns empty string, should fall back to original question."""
    opt = _make_optimizer("")
    question = "What are the liabilities?"
    result = opt.generate_hyde(question)
    assert result == question


def test_hyde_fallback_on_exception() -> None:
    """If LLM raises, should fall back to original question."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("API error")
    opt = QueryOptimizer(client=client, model="mock")
    question = "When does the contract expire?"
    result = opt.generate_hyde(question)
    assert result == question


# ---------------------------------------------------------------------------
# generate_query_variants
# ---------------------------------------------------------------------------


def _variants_json(variants: list[str]) -> str:
    return json.dumps(variants)


def test_variants_returns_list() -> None:
    opt = _make_optimizer(_variants_json(["q1", "q2", "q3"]))
    result = opt.generate_query_variants("question?", n_variants=3)
    assert isinstance(result, list)


def test_variants_returns_correct_count() -> None:
    opt = _make_optimizer(_variants_json(["a", "b", "c"]))
    result = opt.generate_query_variants("question?", n_variants=3)
    assert len(result) == 3


def test_variants_contains_strings() -> None:
    opt = _make_optimizer(_variants_json(["variant 1", "variant 2"]))
    result = opt.generate_query_variants("q?", n_variants=2)
    for v in result:
        assert isinstance(v, str)


def test_variants_preserves_content() -> None:
    expected = ["What are the liability caps?", "How are damages limited?"]
    opt = _make_optimizer(_variants_json(expected))
    result = opt.generate_query_variants("liability question", n_variants=2)
    assert result == expected


def test_variants_handles_markdown_fences() -> None:
    """JSON wrapped in code fences should still parse correctly."""
    raw = '```json\n["v1", "v2"]\n```'
    opt = _make_optimizer(raw)
    result = opt.generate_query_variants("q?", n_variants=2)
    assert result == ["v1", "v2"]


def test_variants_empty_question_raises() -> None:
    opt = _make_optimizer("[]")
    with pytest.raises(ValueError, match="non-empty"):
        opt.generate_query_variants("  ", n_variants=2)


def test_variants_n_variants_zero_raises() -> None:
    opt = _make_optimizer("[]")
    with pytest.raises(ValueError, match="n_variants"):
        opt.generate_query_variants("q?", n_variants=0)


def test_variants_n_variants_too_large_raises() -> None:
    opt = _make_optimizer("[]")
    with pytest.raises(ValueError, match="n_variants"):
        opt.generate_query_variants("q?", n_variants=11)


def test_variants_calls_client_once() -> None:
    client = _make_client(_variants_json(["q1", "q2"]))
    opt = QueryOptimizer(client=client, model="mock")
    opt.generate_query_variants("question", n_variants=2)
    client.messages.create.assert_called_once()


def test_variants_fallback_on_invalid_json() -> None:
    """If LLM returns non-JSON, fall back to [original_question]."""
    opt = _make_optimizer("This is not JSON at all.")
    question = "What are the payment terms?"
    result = opt.generate_query_variants(question, n_variants=3)
    assert result == [question]


def test_variants_fallback_on_wrong_count() -> None:
    """If LLM returns wrong number of variants, fall back to [original_question]."""
    opt = _make_optimizer(_variants_json(["only one"]))  # asked for 3
    question = "liability?"
    result = opt.generate_query_variants(question, n_variants=3)
    assert result == [question]


def test_variants_fallback_on_exception() -> None:
    client = MagicMock()
    client.messages.create.side_effect = ConnectionError("network error")
    opt = QueryOptimizer(client=client, model="mock")
    question = "termination conditions?"
    result = opt.generate_query_variants(question, n_variants=2)
    assert result == [question]


def test_variants_n_variants_one() -> None:
    opt = _make_optimizer(_variants_json(["single rephrasing"]))
    result = opt.generate_query_variants("q?", n_variants=1)
    assert len(result) == 1
    assert result[0] == "single rephrasing"
