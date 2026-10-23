"""Answer-quality metrics: citation validity, abstention, qualifier guard, faithfulness.

All functions here are pure (no models, no I/O).  Every function has a unit
test with a hand-computed expected value.

Metric definitions
------------------
citation_validity_rate
    Fraction of model-generated claims (verified or dropped) whose evidence
    ids all resolved to real evidence blocks.  Measures whether the model cited
    real ids vs hallucinated ones.

citation_quote_rate
    Fraction of claims (that passed id resolution) whose verbatim quote was
    actually found in the cited chunk.  This is the anti-hallucination check.

abstention_precision / abstention_recall
    Treating "abstain" as the positive class:
    - precision = of all abstentions, how many were on truly-absent questions
    - recall    = of all absent questions, how many were correctly abstained on
    Mirrors the retrieval-level ``abstention_scores`` but operates on
    AnswerResult objects so the answer layer can be evaluated end-to-end.

false_abstention_rate
    Of answerable questions, what fraction was wrongly abstained on.  The
    primary latency-quality tradeoff knob: higher min_rerank_score → more
    false abstentions.

qualifier_guard_rate
    Fraction of claims whose qualifier words (shall/must/may/shall not/…) all
    appear in the verbatim quote.  A claim that says "shall" about something
    the quote says "may" about is a faithfulness violation.  This is the
    deterministic sub-signal; NLI-based judged faithfulness is separate.

faithfulness_rate
    Fraction of claims with zero literal violations (numbers, dates, qualifier
    words).  A claim is faithful when every hard element it asserts appears
    verbatim in its cited quote.  This uses the token-level check in
    :mod:`clausewright.eval.faithfulness`.

mean_claims_per_answer
    Of answered questions, average number of verified claims.  Useful for
    monitoring over-answering (many short claims) vs under-answering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from clausewright.qa.schemas import AnswerResult

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_abstained(result: AnswerResult) -> bool:
    return result.status == "abstained"


def _is_answered(result: AnswerResult) -> bool:
    return result.status == "answered"


# ---------------------------------------------------------------------------
# Citation-level metrics (operate on VerificationOutcome counts)
# ---------------------------------------------------------------------------


def citation_validity_rate(
    *,
    n_claims_total: int,
    n_claims_unknown_id: int,
) -> float:
    """Fraction of claims whose evidence ids were all valid.

    A claim failing id resolution is a strong hallucination signal: the model
    cited a source that was never in the evidence list.

    Parameters
    ----------
    n_claims_total:
        Total model-generated claims (valid + all drop reasons).
    n_claims_unknown_id:
        Claims dropped specifically because one or more ids were not in the
        evidence list.
    """
    if n_claims_total < 0 or n_claims_unknown_id < 0:
        raise ValueError("claim counts must be >= 0")
    if n_claims_unknown_id > n_claims_total:
        raise ValueError("unknown-id count cannot exceed total claim count")
    if n_claims_total == 0:
        return math.nan  # undefined: model abstained or produced no claims
    return (n_claims_total - n_claims_unknown_id) / n_claims_total


def citation_quote_rate(
    *,
    n_claims_valid_id: int,
    n_claims_quote_not_found: int,
) -> float:
    """Fraction of id-valid claims whose verbatim quote was found in the chunk.

    Parameters
    ----------
    n_claims_valid_id:
        Claims that passed the id-resolution check (valid + quote-not-found).
    n_claims_quote_not_found:
        Claims that had valid ids but whose quote string was not found verbatim
        in any cited chunk.
    """
    if n_claims_valid_id < 0 or n_claims_quote_not_found < 0:
        raise ValueError("claim counts must be >= 0")
    if n_claims_quote_not_found > n_claims_valid_id:
        raise ValueError("quote-not-found cannot exceed valid-id count")
    if n_claims_valid_id == 0:
        return math.nan
    return (n_claims_valid_id - n_claims_quote_not_found) / n_claims_valid_id


# ---------------------------------------------------------------------------
# Abstention metrics (operate on AnswerResult sequences)
# ---------------------------------------------------------------------------


def abstention_metrics(
    results: Sequence[AnswerResult],
    *,
    true_absent: Sequence[bool],
) -> dict[str, float]:
    """Precision/recall/FAR of the abstain decision over a sequence of results.

    Parameters
    ----------
    results:
        One AnswerResult per question (same order as *true_absent*).
    true_absent:
        ``True`` when the correct answer is "this clause is absent from the
        contract", ``False`` when the contract has an answerable clause.

    Returns
    -------
    dict with keys:
        ``precision`` — of all abstentions, fraction on absent questions.
        ``recall``    — of all absent questions, fraction abstained on.
        ``false_abstention_rate`` — of answerable questions, fraction wrong.
    """
    if len(results) != len(true_absent):
        raise ValueError(
            f"results ({len(results)}) and true_absent ({len(true_absent)}) "
            "must have the same length"
        )

    tp = 0  # truly absent, correctly abstained
    fp = 0  # answerable, wrongly abstained
    fn = 0  # truly absent, wrongly answered
    absent_total = sum(1 for a in true_absent if a)
    answerable_total = len(true_absent) - absent_total
    abstained_total = sum(1 for r in results if _is_abstained(r))

    for result, absent in zip(results, true_absent):
        abstained = _is_abstained(result)
        if absent and abstained:
            tp += 1
        elif not absent and abstained:
            fp += 1
        elif absent and not abstained:
            fn += 1

    return {
        "precision": tp / abstained_total if abstained_total else 0.0,
        "recall": tp / absent_total if absent_total else 1.0,
        "false_abstention_rate": fp / answerable_total if answerable_total else 0.0,
        "abstention_rate": abstained_total / len(results) if results else math.nan,
    }


# ---------------------------------------------------------------------------
# Qualifier guard (pure, no model)
# ---------------------------------------------------------------------------


def qualifier_guard_rate(
    claims_and_quotes: Sequence[tuple[str, str]],
) -> float:
    """Fraction of (claim_text, quote) pairs with no qualifier violations.

    Uses :func:`clausewright.eval.faithfulness.violations` under the hood so
    that the qualifier check and the full faithfulness check share the same
    regex bank.

    Parameters
    ----------
    claims_and_quotes:
        Sequence of ``(claim_text, verbatim_quote)`` pairs.
    """
    from clausewright.eval.faithfulness import QUALIFIER_WORDS, violations

    if not claims_and_quotes:
        return math.nan

    def _qualifier_only_violations(claim: str, quote: str) -> list[str]:
        """Violations restricted to qualifier words only."""
        all_v = violations(claim, quote)
        qual_set = set(QUALIFIER_WORDS)
        return [v for v in all_v if v.lower() in qual_set]

    clean = sum(
        1
        for claim, quote in claims_and_quotes
        if not _qualifier_only_violations(claim, quote)
    )
    return clean / len(claims_and_quotes)


# ---------------------------------------------------------------------------
# Faithfulness rate (pure, no model)
# ---------------------------------------------------------------------------


def faithfulness_rate(
    claims_and_quotes: Sequence[tuple[str, str]],
) -> float:
    """Fraction of (claim_text, quote) pairs with zero literal violations.

    A claim is *faithful* when every number, date, currency amount, and
    qualifier word it states appears in the verbatim quote it cites.  This
    is the deterministic component of faithfulness; it catches the most
    dangerous hallucinations (wrong numbers, flipped qualifiers) without
    any NLI model.

    Parameters
    ----------
    claims_and_quotes:
        Sequence of ``(claim_text, verbatim_quote)`` pairs.
    """
    from clausewright.eval.faithfulness import violations

    if not claims_and_quotes:
        return math.nan

    clean = sum(
        1
        for claim, quote in claims_and_quotes
        if not violations(claim, quote)
    )
    return clean / len(claims_and_quotes)


# ---------------------------------------------------------------------------
# Answer-level summary
# ---------------------------------------------------------------------------


def mean_claims_per_answer(results: Sequence[AnswerResult]) -> float:
    """Average number of verified claims per *answered* question.

    Answered questions with 0 claims are impossible by schema (``min_length=1``
    on the claims tuple), but we handle the edge case gracefully.
    """
    answered = [r for r in results if _is_answered(r)]
    if not answered:
        return math.nan
    total_claims = sum(len(getattr(r, "claims", ())) for r in answered)
    return total_claims / len(answered)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def answer_quality_report(
    results: Sequence[AnswerResult],
    *,
    true_absent: Sequence[bool],
    verification_stats: Sequence[dict[str, int]] | None = None,
) -> dict[str, object]:
    """Aggregate all answer-quality metrics into one report dict.

    Parameters
    ----------
    results:
        One AnswerResult per question.
    true_absent:
        True when the correct answer for that question is "clause absent".
    verification_stats:
        Optional per-question dicts with keys ``n_total``, ``n_unknown_id``,
        ``n_valid_id``, ``n_quote_not_found``.  When provided, citation-level
        metrics are included.

    Returns
    -------
    dict suitable for ``json.dumps``.
    """
    n = len(results)
    if n == 0:
        raise ValueError("results must be non-empty")

    abst = abstention_metrics(results, true_absent=true_absent)

    report: dict[str, object] = {
        "n_questions": n,
        "n_answered": sum(1 for r in results if _is_answered(r)),
        "n_abstained": sum(1 for r in results if _is_abstained(r)),
        "abstention_precision": round(abst["precision"], 4),
        "abstention_recall": round(abst["recall"], 4),
        "false_abstention_rate": round(abst["false_abstention_rate"], 4),
        "abstention_rate": round(abst["abstention_rate"], 4),
        "mean_claims_per_answer": _round_or_nan(mean_claims_per_answer(results)),
    }

    if verification_stats:
        tot = sum(s["n_total"] for s in verification_stats)
        unk = sum(s["n_unknown_id"] for s in verification_stats)
        vld = sum(s["n_valid_id"] for s in verification_stats)
        qnf = sum(s["n_quote_not_found"] for s in verification_stats)
        report["citation_validity_rate"] = _round_or_nan(
            citation_validity_rate(n_claims_total=tot, n_claims_unknown_id=unk)
        )
        report["citation_quote_rate"] = _round_or_nan(
            citation_quote_rate(n_claims_valid_id=vld, n_claims_quote_not_found=qnf)
        )

    return report


def _round_or_nan(v: float, ndigits: int = 4) -> float:
    return round(v, ndigits) if not math.isnan(v) else v
