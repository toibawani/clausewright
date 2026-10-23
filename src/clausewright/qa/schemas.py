"""Schemas for grounded question answering: citations, claims, answers, abstentions.

The QA layer fails closed: an :class:`Answer` may only exist when every claim
carries citations that verified against retrieved chunks.  When verification
fails or evidence is weak, the result is an :class:`Abstention` that states
what was searched and what was missing.

``EvidenceItem`` is internal plumbing (id + chunk + retrieval score) shared by
the prompt builder and the verifier; it is not persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from clausewright.schemas import Chunk

# One clear legal notice, attached once per result — not repeated per line.
NOTICE: Final[str] = (
    "Clausewright locates and summarizes contract text. It is not legal advice."
)


class Citation(BaseModel):
    """A verified pointer from a claim to one retrieved chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: int = Field(..., ge=1, description="1-based position in the evidence list shown to the model")
    doc_id: str = Field(..., min_length=1)
    section_number: str = Field(..., min_length=1)
    page_range: tuple[int, int] = Field(..., description="(start_page, end_page) inclusive")
    score: float = Field(..., description="Reranker relevance score for this chunk")


class Claim(BaseModel):
    """One model-written statement that survived citation verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., min_length=1)
    quote: str = Field(
        ...,
        min_length=1,
        description="Verbatim excerpt the model based the claim on; verified against the cited chunk",
    )
    citations: tuple[Citation, ...] = Field(..., min_length=1)


class Answer(BaseModel):
    """A grounded answer: every claim maps to verified retrieved clauses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["answered"] = "answered"
    question: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, description="Claims joined for display")
    claims: tuple[Claim, ...] = Field(..., min_length=1)
    model: str = Field(..., min_length=1, description="LLM that produced the answer")
    notice: str = NOTICE


class AbstentionReason(str, Enum):
    """Why the system refused to answer."""

    NO_EVIDENCE = "no_evidence"            # retrieval returned nothing
    WEAK_EVIDENCE = "weak_evidence"        # best reranked chunk below the acceptance floor
    NO_VERIFIABLE_CLAIMS = "no_verifiable_claims"  # model answered but nothing verified
    INVALID_OUTPUT = "invalid_output"      # model output was not parseable


class Abstention(BaseModel):
    """A useful refusal: states what was searched and what is missing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["abstained"] = "abstained"
    question: str = Field(..., min_length=1)
    reason: AbstentionReason
    detail: str = Field(..., min_length=1, description="Human-readable: what was searched, what was missing")
    searched_sections: tuple[str, ...] = Field(
        default_factory=tuple, description="Section numbers of the reranked evidence, in order"
    )
    top_score: float | None = Field(default=None, description="Best reranker score, if any chunks were ranked")
    notice: str = NOTICE


# Discriminated on ``status`` so API layers can serialize/validate exactly one shape.
AnswerResult = Answer | Abstention


@dataclass(frozen=True)
class EvidenceItem:
    """Internal: one chunk offered to the model, with its evidence number."""

    evidence_id: int
    chunk: Chunk
    score: float
