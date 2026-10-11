"""Agentic Corrective RAG (CRAG) pipeline for contract question answering.

Pipeline Overview
-----------------
This module implements a multi-step, self-correcting retrieval-augmented
generation loop:

::

    [ User Query ]
         │
         ▼
    [ QueryOptimizer ]          ← HyDE + multi-query expansion (M8)
         │  3-5 query variants + HyDE vector
         ▼
    [ HybridRetriever ]         ← BM25 + Dense via RRF (M6)
         │  top-50 candidates
         ▼
    [ CrossEncoderReranker ]    ← Re-score pairs (M7)
         │  top-5 hyper-relevant chunks
         ▼
    [ RelevanceGrader ]         ← LLM evaluates chunk relevance (CRAG)
         │
         ├──(all relevant)──────────► [ Generator ] → answer + citations
         │
         └──(low relevance)─────────► [ query rewrite + retry, max 2 loops ]
                                              │
                                    (if still poor after retries)
                                              │
                                              ▼
                                      ABSTAIN with explanation

Design decisions
----------------
* **Relevance grading** uses a lightweight Claude call that scores each chunk
  as ``relevant`` or ``irrelevant``.  If the fraction of relevant chunks is
  below ``relevance_threshold`` (default 0.4), the pipeline re-expands the
  query and retries.
* **Hallucination guard** — the generator is instructed via system prompt to
  cite only sections that are explicitly in the retrieved context and to
  respond with ``INSUFFICIENT_EVIDENCE`` if the context doesn't support the
  answer.  The caller receives a structured ``RAGResponse`` distinguishing
  answered from abstained states.
* **Max retries** default to 2 so the agentic loop is bounded.
* All LLM interactions use the same Anthropic client supplied at construction
  time — making the entire pipeline testable with a single mock.

Schemas
-------
``RAGResponse`` is a Pydantic model capturing:
- ``answer`` (str | None): The generated answer, or None if abstained.
- ``citations``: Section numbers cited.
- ``abstained`` (bool): True when evidence was insufficient.
- ``retrieval_attempts`` (int): How many retrieval rounds were used.
- ``chunks_used``: The final set of chunks passed to the generator.
"""

from __future__ import annotations

import logging
import re
from typing import Final

import anthropic
from pydantic import BaseModel, Field

from clausewright.generation.query_optimizer import QueryOptimizer
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_DEFAULT_GENERATOR_MODEL: Final[str] = "claude-sonnet-4-5"
_DEFAULT_GRADER_MODEL: Final[str] = "claude-haiku-4-5"
_DEFAULT_TOP_K_CANDIDATES: Final[int] = 50
_DEFAULT_TOP_N_RERANKED: Final[int] = 5
_DEFAULT_RELEVANCE_THRESHOLD: Final[float] = 0.4
_DEFAULT_MAX_RETRIES: Final[int] = 2
_MAX_TOKENS_GRADER: Final[int] = 64
_MAX_TOKENS_GENERATOR: Final[int] = 1024

_ABSTAIN_SENTINEL: Final[str] = "INSUFFICIENT_EVIDENCE"


# ── Response schema ────────────────────────────────────────────────────────


class RAGResponse(BaseModel):
    """Structured output from the CRAG pipeline."""

    question: str = Field(description="Original user question")
    answer: str | None = Field(default=None, description="Generated answer; None when abstained")
    citations: list[str] = Field(
        default_factory=list, description="Section numbers cited in the answer"
    )
    abstained: bool = Field(default=False, description="True when evidence was insufficient")
    abstain_reason: str | None = Field(default=None, description="Why the system abstained")
    retrieval_attempts: int = Field(default=1, ge=1, description="Number of retrieval rounds used")
    chunks_used: list[Chunk] = Field(
        default_factory=list,
        description="Final chunks passed to the generator",
    )


# ── Pipeline ───────────────────────────────────────────────────────────────


class AgenticRAG:
    """Corrective RAG pipeline with self-reflective retrieval loops.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` instance (real or mocked).
    retriever:
        A :class:`~clausewright.retrieval.hybrid_search.HybridRetriever`
        built from the document corpus.
    reranker:
        A :class:`~clausewright.retrieval.reranker.CrossEncoderReranker`.
    generator_model:
        Anthropic model for answer generation.
    grader_model:
        Anthropic model for relevance grading (fast/cheap Haiku by default).
    top_k_candidates:
        Candidates fetched from the hybrid retriever before reranking.
    top_n_reranked:
        Chunks passed to the generator after reranking.
    relevance_threshold:
        Fraction of reranked chunks that must be graded relevant to skip retry.
    max_retries:
        Maximum number of additional retrieval attempts before abstaining.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        *,
        generator_model: str = _DEFAULT_GENERATOR_MODEL,
        grader_model: str = _DEFAULT_GRADER_MODEL,
        top_k_candidates: int = _DEFAULT_TOP_K_CANDIDATES,
        top_n_reranked: int = _DEFAULT_TOP_N_RERANKED,
        relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._client = client
        self._retriever = retriever
        self._reranker = reranker
        self._generator_model = generator_model
        self._grader_model = grader_model
        self._top_k = top_k_candidates
        self._top_n = top_n_reranked
        self._threshold = relevance_threshold
        self._max_retries = max_retries
        self._optimizer = QueryOptimizer(client=client, model=grader_model)

    # ── Private: relevance grading ─────────────────────────────────────────

    def _grade_chunk(self, question: str, chunk: Chunk) -> bool:
        """Return True if the chunk is relevant to the question."""
        system = (
            "You are a relevance grader for a legal contract retrieval system. "
            "Given a question and a contract excerpt, respond with exactly one word: "
            "'relevant' if the excerpt helps answer the question, or 'irrelevant' if not."
        )
        prompt = f"Question: {question}\n\nExcerpt:\n{chunk.text}"
        try:
            msg = self._client.messages.create(
                model=self._grader_model,
                max_tokens=_MAX_TOKENS_GRADER,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            block = msg.content[0]
            verdict = str(getattr(block, "text", "")).strip().lower()
            return verdict.startswith("relevant")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relevance grading failed for chunk %s: %s", chunk.chunk_id, exc)
            return True  # Optimistic default: don't throw away chunks on API error

    def _relevance_fraction(self, question: str, chunks: list[Chunk]) -> float:
        """Return the fraction of *chunks* graded relevant for *question*."""
        if not chunks:
            return 0.0
        grades = [self._grade_chunk(question, c) for c in chunks]
        return sum(grades) / len(grades)

    # ── Private: retrieval round ───────────────────────────────────────────

    def _retrieve_and_rerank(self, query: str) -> list[Chunk]:
        """Run one full retrieve → rerank cycle and return top chunks."""
        candidates = self._retriever.search(query, top_k=self._top_k)
        if not candidates:
            return []
        reranked = self._reranker.rerank(query, candidates, top_n=self._top_n)
        return [c for c, _ in reranked]

    # ── Private: answer generation ─────────────────────────────────────────

    def _build_context(self, chunks: list[Chunk]) -> str:
        """Format chunks into a labelled context block for the LLM."""
        parts: list[str] = []
        for chunk in chunks:
            label = f"[Section {chunk.section_number}, p.{chunk.page_range[0]}]"
            parts.append(f"{label}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, question: str, chunks: list[Chunk]) -> str:
        """Call the generator LLM and return its raw text response."""
        context = self._build_context(chunks)
        system = (
            "You are a precise contract analysis assistant. "
            "Answer the question using ONLY the contract excerpts provided below. "
            "Always cite the section label (e.g. [Section 3.2]) when you use it. "
            f"If the excerpts do not contain enough information, respond with exactly: "
            f"{_ABSTAIN_SENTINEL}"
        )
        user_content = f"Contract excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"
        msg = self._client.messages.create(
            model=self._generator_model,
            max_tokens=_MAX_TOKENS_GENERATOR,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        block = msg.content[0]
        return str(getattr(block, "text", "")).strip()

    @staticmethod
    def _extract_citations(answer: str) -> list[str]:
        """Extract all [Section X.Y] labels from the answer string."""
        return re.findall(r"\[Section\s+([^\]]+)\]", answer)

    # ── Public API ─────────────────────────────────────────────────────────

    def answer(self, question: str) -> RAGResponse:
        """Run the full agentic CRAG pipeline and return a structured response.

        Parameters
        ----------
        question:
            The user's question in plain English.

        Returns
        -------
        RAGResponse
            Structured result with answer, citations, abstain flag, and
            provenance metadata.

        Raises
        ------
        ValueError
            If *question* is blank.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")

        attempts = 0
        current_query = question

        while attempts <= self._max_retries:
            attempts += 1
            logger.info(
                "CRAG loop attempt %d/%d for question: %s",
                attempts,
                self._max_retries + 1,
                question[:60],
            )

            # ── Step 1: Retrieve + Rerank ──────────────────────────────────
            chunks = self._retrieve_and_rerank(current_query)
            if not chunks:
                logger.warning("No chunks retrieved on attempt %d", attempts)
                if attempts > self._max_retries:
                    break
                # Rewrite the query and retry
                variants = self._optimizer.generate_query_variants(question, n_variants=3)
                current_query = variants[0]
                continue

            # ── Step 2: Relevance grading (CRAG) ──────────────────────────
            rel_fraction = self._relevance_fraction(question, chunks)
            logger.info("Relevance fraction: %.2f (threshold %.2f)", rel_fraction, self._threshold)

            if rel_fraction >= self._threshold or attempts > self._max_retries:
                # Acceptable relevance — proceed to generation
                break

            # Poor relevance → query rewrite and retry
            logger.info("Low relevance; rewriting query (attempt %d)", attempts)
            variants = self._optimizer.generate_query_variants(question, n_variants=3)
            current_query = variants[min(attempts - 1, len(variants) - 1)]

        if not chunks:
            return RAGResponse(
                question=question,
                abstained=True,
                abstain_reason="No relevant chunks found after retrieval attempts.",
                retrieval_attempts=attempts,
            )

        # ── Step 3: Answer generation ──────────────────────────────────────
        try:
            raw_answer = self._generate_answer(question, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error("Answer generation failed: %s", exc)
            return RAGResponse(
                question=question,
                abstained=True,
                abstain_reason=f"Generation error: {exc}",
                retrieval_attempts=attempts,
                chunks_used=chunks,
            )

        # ── Step 4: Abstain check ──────────────────────────────────────────
        if _ABSTAIN_SENTINEL in raw_answer:
            return RAGResponse(
                question=question,
                abstained=True,
                abstain_reason="Model determined that context is insufficient.",
                retrieval_attempts=attempts,
                chunks_used=chunks,
            )

        citations = self._extract_citations(raw_answer)
        return RAGResponse(
            question=question,
            answer=raw_answer,
            citations=citations,
            abstained=False,
            retrieval_attempts=attempts,
            chunks_used=chunks,
        )
