"""Key-terms structured extractor for legal contracts (closes #30).

Seven core fields
-----------------
Every standard commercial contract contains a predictable set of commercially
important terms.  Lawyers spend significant time locating them manually.  This
extractor produces a structured ``KeyTermsReport`` with citations and per-field
abstention.

Fields extracted:
1. **parties** — contracting parties and their roles
2. **term** — contract duration
3. **renewal** — auto-renewal or notice-to-renew provisions
4. **termination_notice** — required notice period before terminating
5. **liability_cap** — maximum liability in monetary terms
6. **governing_law** — governing law / jurisdiction
7. **payment_terms** — payment schedule, amounts, or milestones

Design
------
* Each field is extracted independently via a targeted question to the
  ``AnswerEngine`` or directly to the LLM with a focused prompt.
* A field abstains (returns ``None``) when the engine's answer is an
  ``Abstention`` — no hallucinated defaults.
* The verifier inside ``AnswerEngine`` already enforces citation validity and
  quote fidelity, so citation guarantees flow through automatically.
* Output is a Pydantic model with ``Optional[str]`` fields and matching
  citation lists so downstream code can render highlighted PDFs.

CLI usage
---------
::

    python -m clausewright.extract key-terms --contract data/raw/cuad/sample.pdf

Eval usage
----------
::

    from clausewright.extraction.key_terms import KeyTermsExtractor, KeyTermsReport
    report = extractor.extract(chunks)
    # report.parties, report.term, report.liability_cap, …
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from clausewright.schemas import Chunk  # noqa: F401 – used in Protocol type stubs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """Result for a single extracted key term.

    Attributes
    ----------
    value:
        The extracted text, or ``None`` if the field was not found.
    citation_chunk_ids:
        Chunk IDs that support the value.  Empty when abstaining.
    abstained:
        True when the field could not be found or verified.
    """

    value: str | None = None
    citation_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


@dataclass
class KeyTermsReport:
    """Structured extraction of seven core commercial contract fields.

    All fields are individually abstainable; ``None`` means "not found in the
    contract" — never a hallucinated default.
    """

    parties: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    term: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    renewal: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    termination_notice: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    liability_cap: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    governing_law: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))
    payment_terms: FieldResult = field(default_factory=lambda: FieldResult(abstained=True))

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize to a JSON-safe dict."""
        out: dict[str, dict[str, Any]] = {}
        for fname in (
            "parties", "term", "renewal", "termination_notice",
            "liability_cap", "governing_law", "payment_terms",
        ):
            fr: FieldResult = getattr(self, fname)
            out[fname] = {
                "value": fr.value,
                "citation_chunk_ids": fr.citation_chunk_ids,
                "abstained": fr.abstained,
            }
        return out

    def format_table(self) -> str:
        """Render an ASCII summary table."""
        lines = [
            f"{'Field':<24} {'Value (truncated)':<55} {'Abstained'}",
            "-" * 85,
        ]
        for fname in (
            "parties", "term", "renewal", "termination_notice",
            "liability_cap", "governing_law", "payment_terms",
        ):
            fr: FieldResult = getattr(self, fname)
            display = (fr.value or "—")[:54]
            abstained_str = "YES" if fr.abstained else "no"
            lines.append(f"{fname:<24} {display:<55} {abstained_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_FIELD_QUESTIONS: dict[str, str] = {
    "parties": (
        "Who are the contracting parties in this agreement and what is each party's role?"
    ),
    "term": (
        "What is the duration or term of this agreement, including the start date and end date?"
    ),
    "renewal": (
        "Does this agreement auto-renew or require a notice to renew? "
        "If so, what are the renewal conditions and notice period?"
    ),
    "termination_notice": (
        "What notice period is required to terminate this agreement? "
        "How many days notice must be given, and to whom?"
    ),
    "liability_cap": (
        "What is the cap on liability or maximum aggregate liability under this agreement? "
        "State the monetary limit if specified."
    ),
    "governing_law": (
        "What is the governing law and jurisdiction for this agreement?"
    ),
    "payment_terms": (
        "What are the payment terms, including amounts, schedule, due dates, "
        "and any late payment consequences?"
    ),
}


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class _AnswerEngineProtocol(Protocol):
    """Duck-typed interface for AnswerEngine (or any compatible engine)."""

    def answer(self, question: str) -> Any:  # returns AnswerResult
        ...


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class KeyTermsExtractor:
    """Extract the seven core key terms from a pre-indexed contract.

    Parameters
    ----------
    engine:
        An ``AnswerEngine`` (or compatible stub) used for each field question.
    """

    def __init__(self, engine: _AnswerEngineProtocol) -> None:
        self._engine = engine

    def extract(self) -> KeyTermsReport:
        """Run extraction for all seven fields and return a :class:`KeyTermsReport`.

        Each field is independently queried; an ``Abstention`` result causes
        that field to remain abstained.  Errors in one field do not abort
        others.
        """
        report = KeyTermsReport()

        for field_name, question in _FIELD_QUESTIONS.items():
            try:
                result_field = self._extract_field(question)
            except Exception:
                logger.exception("KeyTermsExtractor: error extracting field '%s'", field_name)
                result_field = FieldResult(abstained=True)
            setattr(report, field_name, result_field)

        return report

    def _extract_field(self, question: str) -> FieldResult:
        """Ask the engine a single field question and parse the result."""
        from clausewright.qa.schemas import Abstention, Answer  # local import to avoid circular

        result = self._engine.answer(question)

        if isinstance(result, Abstention) or type(result).__name__.lower().endswith("abstention"):
            return FieldResult(abstained=True)

        if isinstance(result, Answer) or hasattr(result, "claims"):
            claims = getattr(result, "claims", ())
            chunk_ids = list(
                dict.fromkeys(
                    str(
                        getattr(
                            citation,
                            "chunk_id",
                            f"{getattr(citation, 'doc_id', '')}:{getattr(citation, 'section_number', '')}",
                        )
                    )
                    for claim in claims
                    for citation in getattr(claim, "citations", ())
                )
            )
            chunk_ids = [cid for cid in chunk_ids if cid and cid != ":"]
            return FieldResult(
                value=getattr(result, "text", None),
                citation_chunk_ids=chunk_ids,
                abstained=False,
            )

        # Unknown result type — abstain safely
        logger.warning("KeyTermsExtractor: unexpected result type %s", type(result))
        return FieldResult(abstained=True)
