"""Tests for span localization: offsets, selection, disabled path."""

from __future__ import annotations

import pytest

from clausewright.retrieval.span_localizer import (
    SpanLocalizer,
    split_sentences,
)

WINDOW = (
    "The parties entered into this Agreement on March 3, 2019. "
    "The Effective Date is the date of the last signature. "
    "Either party may terminate for convenience on thirty days notice."
)


def test_split_sentences_offsets_slice_the_text_exactly() -> None:
    spans = split_sentences(WINDOW)

    assert len(spans) == 3
    for start, end in spans:
        assert WINDOW[start:end].strip() == WINDOW[start:end]
    assert WINDOW[spans[0][0] : spans[0][1]].startswith("The parties entered")


def test_bm25_selects_the_sentence_sharing_query_terms() -> None:
    localizer = SpanLocalizer(method="bm25")

    span = localizer.localize("effective date of the agreement", WINDOW)

    assert span is not None
    assert "Effective Date" in span.text
    assert WINDOW[span.char_start : span.char_end] == span.text


def test_termination_query_selects_the_termination_sentence() -> None:
    localizer = SpanLocalizer(method="bm25")

    span = localizer.localize("terminate for convenience notice", WINDOW)

    assert span is not None
    assert "terminate for convenience" in span.text


def test_disabled_localizer_returns_none() -> None:
    localizer = SpanLocalizer(method="bm25", enabled=False)

    assert localizer.localize("effective date", WINDOW) is None


def test_single_sentence_window_is_not_narrowed() -> None:
    localizer = SpanLocalizer(method="bm25")

    assert localizer.localize("effective date", "Only one sentence here, long enough.") is None


def test_blank_query_is_rejected() -> None:
    localizer = SpanLocalizer(method="bm25")

    with pytest.raises(ValueError):
        localizer.localize("   ", WINDOW)
