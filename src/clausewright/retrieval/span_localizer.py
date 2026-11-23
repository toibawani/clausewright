"""Localize the minimal supporting span inside a retrieved window.

Why
---
Retrieval returns a 300 to 400 word window because that is what embeds well.
A reader does not want the window: they want the sentence that answers the
question, and a citation that points at forty words is a citation nobody
checks. This module picks one sentence (or clause) out of the window and keeps
its character offsets, so the citation can be rendered on the exact text while
the window stays available as context.

Two scorers, same contract
--------------------------
* ``bm25`` scores each candidate sentence against the query with a BM25 index
  built over the sentences of that one window. Cheap, deterministic, no model,
  no network.
* ``cross_encoder`` scores ``(query, sentence)`` pairs with the same
  cross-encoder family used for reranking. Better at paraphrase, far slower.

Offsets
-------
``LocalizedSpan.char_start`` and ``char_end`` index the *window* string, so
``window[char_start:char_end] == span.text`` always holds. Anything that
renders a citation can highlight the span without re-extracting text.

The localizer is opt-in: callers pass it explicitly and the answer engine only
uses it when configured, because narrowing a citation is a product decision,
not a retrieval default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

#: Sentence boundary: after terminal punctuation, or a blank line.
_BOUNDARY: Final = re.compile(r"(?<=[.;:])\s+|\n+")

#: Candidates shorter than this are headings, numbers, or noise.
MIN_CANDIDATE_CHARS: Final[int] = 20

#: Default cross-encoder used when the caller does not supply one.
DEFAULT_CE_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Method = Literal["bm25", "cross_encoder"]


@dataclass(frozen=True)
class LocalizedSpan:
    """One sentence of a window, with offsets into the window text."""

    text: str
    char_start: int
    char_end: int
    method: str


class SupportsPairScoring(Protocol):
    """The surface needed from a cross-encoder."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> Any: ...


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of sentence-like units in *text*.

    Offsets are exact slices of *text* (trailing whitespace is excluded by the
    boundary match itself, not by stripping the slice).
    """
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _BOUNDARY.finditer(text):
        if match.start() > position:
            spans.append((position, match.start()))
        position = match.end()
    if position < len(text):
        spans.append((position, len(text)))
    return [(start, end) for start, end in spans if end > start]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SpanLocalizer:
    """Select the sentence of a window that best supports the query.

    Parameters
    ----------
    method:
        ``"bm25"`` or ``"cross_encoder"``.
    model_name:
        Cross-encoder model, used only by the ``cross_encoder`` method.
    model:
        An already-loaded cross-encoder (anything with ``predict``); avoids
        reloading a model per query when the caller already has one.
    enabled:
        When ``False``, :meth:`localize` returns ``None`` and the window is
        used unchanged. This is the product default.
    """

    def __init__(
        self,
        *,
        method: Method = "bm25",
        model_name: str = DEFAULT_CE_MODEL,
        model: SupportsPairScoring | None = None,
        enabled: bool = True,
    ) -> None:
        if method not in ("bm25", "cross_encoder"):
            raise ValueError(f"unknown method: {method}")
        self._method = method
        self._enabled = enabled
        self._model = model
        if method == "cross_encoder" and model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name, device="cpu")

    @property
    def method(self) -> str:
        return self._method

    def localize(self, query: str, window: str) -> LocalizedSpan | None:
        """Return the best supporting sentence of *window* for *query*.

        Returns ``None`` when the localizer is disabled, the window has no
        candidate long enough, or the window has a single sentence (then the
        window already is the minimal span and narrowing it buys nothing).
        """
        if not self._enabled:
            return None
        if not query.strip():
            raise ValueError("query must be non-empty")

        candidates = [
            (start, end)
            for start, end in split_sentences(window)
            if end - start >= MIN_CANDIDATE_CHARS
        ]
        if len(candidates) <= 1:
            return None

        if self._method == "bm25":
            best_index = self._best_bm25(query, window, candidates)
        else:
            best_index = self._best_cross_encoder(query, window, candidates)

        start, end = candidates[best_index]
        return LocalizedSpan(
            text=window[start:end],
            char_start=start,
            char_end=end,
            method=self._method,
        )

    def _best_bm25(
        self, query: str, window: str, candidates: list[tuple[int, int]]
    ) -> int:
        sentences = [window[start:end] for start, end in candidates]
        bm25 = BM25Okapi([_tokenize(s) for s in sentences])
        scores = bm25.get_scores(_tokenize(query))
        best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
        return best

    def _best_cross_encoder(
        self, query: str, window: str, candidates: list[tuple[int, int]]
    ) -> int:
        assert self._model is not None
        pairs = [(query, window[start:end]) for start, end in candidates]
        raw = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        scores = [float(score) for score in raw]
        return max(range(len(scores)), key=lambda i: (scores[i], -i))
