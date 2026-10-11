"""Query optimizer: HyDE + Multi-Query Expansion for pre-retrieval query enrichment.

Background
----------
Raw user questions are often too short or colloquial to match the terminology
in contract documents.  Two techniques dramatically improve retrieval recall:

1. **HyDE (Hypothetical Document Embeddings)** — Gao et al. 2022.
   The LLM generates a *hypothetical answer* to the question.  That answer is
   denser in domain vocabulary and closer in vector space to relevant passages
   than the short question itself.  We embed the hypothetical answer and use it
   as a search vector instead of the question.

2. **Multi-Query Expansion** — Generate *N* rephrased variants of the user
   question (different perspective, vocabulary, granularity).  Run parallel
   retrievals and merge results.  Covers lexical gaps that any single phrasing
   misses.

Architecture
------------
``QueryOptimizer`` calls the Anthropic Claude API (``claude-haiku-*`` or any
model the caller supplies) to generate:

- A single hypothetical document snippet (HyDE).
- A list of *n_variants* alternative query phrasings.

Both outputs are returned as plain strings — the caller decides which retrieval
path to use.  This keeps the optimizer stateless and easily testable (the
Anthropic client is injected / mockable).

Prompt engineering
------------------
Prompts are short, system-role injected, and ask for JSON output so parsing
is reliable.  We validate the JSON and fall back gracefully to the original
query if the model misbehaves.

Usage
-----
::

    from clausewright.generation.query_optimizer import QueryOptimizer
    import anthropic

    client = anthropic.Anthropic()
    opt = QueryOptimizer(client=client)

    # HyDE: get an ideal hypothetical answer to embed
    hyde_text = opt.generate_hyde("What are the indemnification obligations?")

    # Multi-query: get 3 rephrased variants
    variants = opt.generate_query_variants(
        "What are the indemnification obligations?", n_variants=3
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Final

import anthropic

logger = logging.getLogger(__name__)

# Default model: Claude Haiku is fast and cheap — ideal for query pre-processing.
_DEFAULT_MODEL: Final[str] = "claude-haiku-4-5"
_MAX_TOKENS: Final[int] = 512


class QueryOptimizer:
    """Generate HyDE snippets and multi-query variants using an Anthropic LLM.

    Parameters
    ----------
    client:
        An instantiated ``anthropic.Anthropic`` client.  Pass a mock in tests.
    model:
        Anthropic model name.  Defaults to ``claude-haiku-4-5`` (fast/cheap).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """Name of the underlying LLM."""
        return self._model

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------

    def generate_hyde(self, question: str) -> str:
        """Generate a hypothetical document passage that would answer *question*.

        The returned text is intended to be embedded (not shown to users) —
        it is a *search vector* richer in domain vocabulary than the bare question.

        Parameters
        ----------
        question:
            The user's question string.

        Returns
        -------
        str
            A short passage (≤ 200 words) representing an ideal answer, or
            *question* itself if the LLM response cannot be parsed.

        Raises
        ------
        ValueError
            If *question* is blank.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")

        system = (
            "You are a legal contract analysis assistant. "
            "Your task is to write a short (≤ 200 word) passage from a contract "
            "that directly answers the given question. "
            "Respond with ONLY the passage text — no preamble, no labels."
        )
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": question}],
            )
            block = message.content[0]
            text = str(getattr(block, "text", "")).strip()
            if not text:
                raise ValueError("empty LLM response")
            logger.debug("HyDE generated (%d chars) for: %s", len(text), question[:60])
            return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE generation failed (%s); falling back to original query", exc)
            return question

    # ------------------------------------------------------------------
    # Multi-query expansion
    # ------------------------------------------------------------------

    def generate_query_variants(
        self,
        question: str,
        n_variants: int = 3,
    ) -> list[str]:
        """Produce *n_variants* rephrased versions of *question*.

        Parameters
        ----------
        question:
            The original user question.
        n_variants:
            How many variants to generate (1–10).

        Returns
        -------
        list[str]
            A list of length *n_variants* containing alternative phrasings.
            Falls back to ``[question]`` if parsing fails.

        Raises
        ------
        ValueError
            If *question* is blank or *n_variants* is out of range.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")
        if not (1 <= n_variants <= 10):
            raise ValueError(f"n_variants must be between 1 and 10, got {n_variants}")

        system = (
            "You are a query expansion assistant for a legal contract retrieval system. "
            f"Rephrase the user's question into {n_variants} alternative search queries "
            "that cover different angles (legal terminology, broader scope, specific sub-questions). "
            f"Respond with a JSON array of exactly {n_variants} strings and nothing else. "
            'Example: ["variant 1", "variant 2", "variant 3"]'
        )

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": question}],
            )
            block = message.content[0]
            raw = str(getattr(block, "text", "")).strip()

            # Extract JSON array — handle markdown code fences defensively.
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array found in response: {raw[:200]}")

            variants: list[str] = json.loads(match.group())
            if not isinstance(variants, list) or len(variants) != n_variants:
                raise ValueError(f"Expected list of {n_variants} strings, got: {variants}")
            # Ensure all elements are non-empty strings
            variants = [str(v).strip() for v in variants if str(v).strip()]
            if len(variants) != n_variants:
                raise ValueError("Some variant strings were empty")

            logger.debug("Generated %d query variants for: %s", n_variants, question[:60])
            return variants

        except Exception as exc:  # noqa: BLE001
            logger.warning("Query variant generation failed (%s); returning original query", exc)
            return [question]
