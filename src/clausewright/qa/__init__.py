"""Grounded question answering: retrieve, verify citations, answer or abstain.

Public surface::

    from clausewright.qa import AnswerEngine, AnthropicClient

    engine = AnswerEngine(retriever, reranker, AnthropicClient())
    result = engine.answer("What is the notice period for termination?")
    if result.status == "answered":
        ...
    else:
        print(result.detail)  # what was searched, what was missing
"""

from clausewright.qa.engine import AnswerEngine
from clausewright.qa.llm import AnthropicClient, LLMClient, LLMError
from clausewright.qa.schemas import (
    NOTICE,
    Abstention,
    AbstentionReason,
    Answer,
    AnswerResult,
    Citation,
    Claim,
    EvidenceItem,
)
from clausewright.qa.verifier import DropReason, VerificationOutcome, verify

__all__ = [
    "NOTICE",
    "Abstention",
    "AbstentionReason",
    "Answer",
    "AnswerEngine",
    "AnswerResult",
    "AnthropicClient",
    "Citation",
    "Claim",
    "DropReason",
    "EvidenceItem",
    "LLMClient",
    "LLMError",
    "VerificationOutcome",
    "verify",
]
