"""Citation verifier: the fail-closed gate between model output and answers.

Every model claim must survive three checks before it can appear in an
:class:`~clausewright.qa.schemas.Answer`:

1. **Output parses** as the exact JSON schema we asked for (no code fences,
   no trailing prose).  Otherwise: invalid output, engine abstains.
2. **Evidence ids resolve.**  Every id in ``evidence_ids`` must be one of the
   blocks we actually sent.  A claim citing an unknown id is dropped.
3. **Quote is verbatim.**  The claim's quote must appear (whitespace-
   normalised, case-sensitive) in at least one of the chunks it cites.  A
   claim whose quote is not found is dropped — that is the anti-hallucination
   check that makes citations trustworthy.

Claims that fail any check are recorded in ``dropped`` with a machine-readable
reason, so the eval harness can measure citation validity and abstention
precision.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clausewright.qa.schemas import Citation, Claim, EvidenceItem

logger = logging.getLogger(__name__)

# Max characters accepted from the model before we even try to parse;
# a runaway completion is invalid output, not something to feed to json.loads.
_MAX_OUTPUT_CHARS: Final[int] = 32_000


class DropReason(str, Enum):
    """Why a single claim was dropped."""

    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    QUOTE_NOT_FOUND = "quote_not_found"


class DropRecord(BaseModel):
    """Audit record for one dropped claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_preview: str = Field(..., max_length=140)
    reason: DropReason


class _ModelClaim(BaseModel):
    """Schema we asked the model for — one claim."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1)
    evidence_ids: list[int] = Field(..., min_length=1)


class _ModelResponse(BaseModel):
    """Schema we asked the model for — full response."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    reason_if_unanswerable: str | None = None
    claims: list[_ModelClaim] = Field(default_factory=list)


class VerificationOutcome(BaseModel):
    """Result of verifying one model completion."""

    model_config = ConfigDict(frozen=True)

    valid_claims: tuple[Claim, ...] = ()
    dropped: tuple[DropRecord, ...] = ()
    model_says_unanswerable: bool = False
    model_reason: str | None = None
    invalid_output: bool = False
    invalid_detail: str | None = None


def _normalise(text: str) -> str:
    """Collapse all whitespace runs to single spaces; used for quote matching."""
    return re.sub(r"\s+", " ", text).strip()


def _find_char_offset(quote: str, text: str) -> tuple[int, int] | None:
    """Locate the exact character span of quote within chunk text."""
    clean_q = _normalise(quote)
    clean_t = _normalise(text)
    idx = clean_t.find(clean_q)
    if idx == -1:
        return None
    return (idx, idx + len(clean_q))


def _strip_code_fence(raw: str) -> str:
    """Remove a surrounding ```json ... ``` fence if the model added one."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_model_output(raw: str) -> tuple[_ModelResponse | None, str | None]:
    """Parse the model's completion into the expected schema.

    Returns ``(response, None)`` on success or ``(None, detail)`` on failure.
    The detail is safe to show in an abstention; it never contains contract
    text (only a short description of the parse failure).
    """
    if len(raw) > _MAX_OUTPUT_CHARS:
        return None, f"model output exceeded {_MAX_OUTPUT_CHARS} characters"
    text = _strip_code_fence(raw)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"model output is not valid JSON ({exc.msg})"
    try:
        return _ModelResponse.model_validate(data), None
    except ValidationError as exc:
        # Report only error locations/messages, not values (they may be
        # contract text).
        issues = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        return None, f"model output does not match the schema: {issues}"


def verify_claims(
    response: _ModelResponse, evidence: Sequence[EvidenceItem]
) -> VerificationOutcome:
    """Apply the evidence-id and verbatim-quote checks to every claim."""
    by_id = {item.evidence_id: item for item in evidence}
    valid: list[Claim] = []
    dropped: list[DropRecord] = []

    for mc in response.claims:
        preview = mc.text[:140]
        ids = sorted(set(mc.evidence_ids))
        if any(i not in by_id for i in ids):
            dropped.append(
                DropRecord(text_preview=preview, reason=DropReason.UNKNOWN_EVIDENCE_ID)
            )
            continue
        normalised_quote = _normalise(mc.quote)
        cited_items = [by_id[i] for i in ids]
        if not any(normalised_quote in _normalise(it.chunk.text) for it in cited_items):
            dropped.append(
                DropRecord(text_preview=preview, reason=DropReason.QUOTE_NOT_FOUND)
            )
            continue
        citations = tuple(
            Citation(
                evidence_id=it.evidence_id,
                doc_id=it.chunk.doc_id,
                section_number=it.chunk.section_number,
                page_range=it.chunk.page_range,
                score=it.score,
                page_number=it.chunk.page_range[0],
                char_offset=_find_char_offset(mc.quote, it.chunk.text),
            )
            for it in cited_items
        )
        valid.append(
            Claim(text=mc.text.strip(), quote=mc.quote.strip(), citations=citations)
        )

    return VerificationOutcome(
        valid_claims=tuple(valid),
        dropped=tuple(dropped),
        model_says_unanswerable=not response.answerable,
        model_reason=response.reason_if_unanswerable,
    )


def verify(raw_output: str, evidence: Sequence[EvidenceItem]) -> VerificationOutcome:
    """Full pipeline: parse then verify.  Never raises on bad model output."""
    response, parse_detail = parse_model_output(raw_output)
    if response is None:
        logger.info("Model output rejected: %s", parse_detail)
        return VerificationOutcome(invalid_output=True, invalid_detail=parse_detail)
    outcome = verify_claims(response, evidence)
    logger.info(
        "Verification: %d valid, %d dropped, unanswerable=%s",
        len(outcome.valid_claims),
        len(outcome.dropped),
        outcome.model_says_unanswerable,
    )
    return outcome

