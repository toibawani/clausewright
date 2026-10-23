"""Answer engine: retrieve, rerank, generate, verify, answer or abstain.

Pipeline for one question:

1. **Retrieve** — hybrid BM25+dense search returns candidates.
2. **Rerank** — cross-encoder scores them; keep the top N.
3. **Gate** — if the best reranked score is below the acceptance floor,
   abstain before spending an LLM call.
4. **Generate** — the LLM answers strictly from numbered evidence blocks.
5. **Verify** — the verifier drops any claim with an unresolvable citation or
   a quote not found verbatim in the cited chunk.  If nothing survives,
   abstain.

The engine takes its dependencies as duck-typed protocols, so tests inject
fakes and no ML model or network access is needed to exercise the full
pipeline.  Retrieval thresholds are constructor parameters, deliberately not
guessed in config: they must be calibrated on the gold eval set.
"""

from __future__ import annotations

import logging
from typing import Protocol

from clausewright.qa.llm import LLMClient
from clausewright.qa.prompts import SYSTEM_PROMPT, build_user_message
from clausewright.qa.schemas import (
    Abstention,
    AbstentionReason,
    Answer,
    AnswerResult,
    Claim,
    EvidenceItem,
)
from clausewright.qa.verifier import verify
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


class SupportsSearch(Protocol):
    """The retrieval surface the engine needs (HybridRetriever satisfies it)."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


class SupportsRerank(Protocol):
    """The reranking surface the engine needs (CrossEncoderReranker satisfies it)."""

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        *,
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]: ...



class AnswerEngine:
    """Grounded question answering over a contract index.

    Parameters
    ----------
    retriever:
        Anything with ``search(query, top_k) -> list[(Chunk, score)]``.
    reranker:
        Anything with ``rerank(query, candidates, top_n=...) -> list[(Chunk, score)]``.
    llm:
        Anything satisfying :class:`~clausewright.qa.llm.LLMClient`.
    top_k:
        Candidates pulled from retrieval before reranking.
    keep_after_rerank:
        Evidence blocks handed to the model.
    min_rerank_score:
        Acceptance floor: if the best reranker score is below this, abstain
        without calling the LLM.  ``None`` disables the gate.  Calibrate on
        the gold set — do not guess.
    """

    def __init__(
        self,
        retriever: SupportsSearch,
        reranker: SupportsRerank,
        llm: LLMClient,
        *,
        top_k: int = 50,
        keep_after_rerank: int = 5,
        min_rerank_score: float | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if keep_after_rerank < 1:
            raise ValueError(f"keep_after_rerank must be >= 1, got {keep_after_rerank}")
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._top_k = top_k
        self._keep = keep_after_rerank
        self._min_rerank_score = min_rerank_score

    def answer(self, question: str) -> AnswerResult:
        """Answer *question* with verified citations, or abstain usefully."""
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        # 1. Retrieve.
        candidates = self._retriever.search(question, top_k=self._top_k)
        if not candidates:
            return Abstention(
                question=question,
                reason=AbstentionReason.NO_EVIDENCE,
                detail="No clauses matched the question in this contract. "
                "The question may reference terms the contract does not use.",
            )

        # 2. Rerank.
        ranked = self._reranker.rerank(question, candidates, top_n=self._keep)
        if not ranked:
            return Abstention(
                question=question,
                reason=AbstentionReason.NO_EVIDENCE,
                detail="Retrieved clauses were all filtered out during reranking.",
            )
        top_score = ranked[0][1]
        searched_sections = tuple(dict.fromkeys(c.section_number for c, _ in ranked))

        # 3. Gate on weak evidence before spending an LLM call.
        if self._min_rerank_score is not None and top_score < self._min_rerank_score:
            return Abstention(
                question=question,
                reason=AbstentionReason.WEAK_EVIDENCE,
                detail=(
                    f"The strongest retrieved clause (section {ranked[0][0].section_number}) "
                    f"scored {top_score:.3f}, below the acceptance floor "
                    f"{self._min_rerank_score:.3f}. Searched sections: "
                    f"{', '.join(searched_sections)}. No clause in this contract "
                    "clearly addresses the question."
                ),
                searched_sections=searched_sections,
                top_score=top_score,
            )


        # 4. Generate.
        evidence = [
            EvidenceItem(evidence_id=i, chunk=chunk, score=score)
            for i, (chunk, score) in enumerate(ranked, start=1)
        ]
        user_message = build_user_message(question, evidence)
        raw_output = self._llm.complete(system=SYSTEM_PROMPT, user=user_message)

        # 5. Verify; fail closed.
        outcome = verify(raw_output, evidence)

        if outcome.invalid_output:
            return Abstention(
                question=question,
                reason=AbstentionReason.INVALID_OUTPUT,
                detail=(
                    "The model's response could not be validated "
                    f"({outcome.invalid_detail}). No answer is shown because it "
                    "could not be checked against the retrieved clauses."
                ),
                searched_sections=searched_sections,
                top_score=top_score,
            )

        if outcome.model_says_unanswerable and not outcome.valid_claims:
            model_reason = outcome.model_reason or "no reason given"
            return Abstention(
                question=question,
                reason=AbstentionReason.NO_EVIDENCE,
                detail=(
                    f"Searched sections {', '.join(searched_sections)}; the "
                    f"retrieved clauses did not answer the question. Model note: "
                    f"{model_reason}"
                ),
                searched_sections=searched_sections,
                top_score=top_score,
            )

        if not outcome.valid_claims:
            dropped_summary = (
                ", ".join(d.reason.value for d in outcome.dropped) or "none reported"
            )
            return Abstention(
                question=question,
                reason=AbstentionReason.NO_VERIFIABLE_CLAIMS,
                detail=(
                    "The model produced claims, but none could be verified "
                    f"against the retrieved clauses (drop reasons: "
                    f"{dropped_summary}). Searched sections: "
                    f"{', '.join(searched_sections)}."
                ),
                searched_sections=searched_sections,
                top_score=top_score,
            )

        # Answered: join claim texts; every claim carries verified citations.
        claims: tuple[Claim, ...] = outcome.valid_claims
        logger.info(
            "Answered question with %d verified claims (%d dropped)",
            len(claims),
            len(outcome.dropped),
        )
        return Answer(
            question=question,
            text=" ".join(c.text for c in claims),
            claims=claims,
            model=self._llm.model_name,
        )
