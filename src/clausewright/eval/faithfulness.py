"""Deterministic, token-level faithfulness check for one claim vs its quote.

A claim is literal-faithful when every "hard" element it states is literally
present in the quote it cites:

* numbers and currency amounts (``30``, ``1,000``, ``$2,000,000``),
* dates (``January 5, 2024``, ``2024-01-05``, ``5th day of June``),
* qualifier words that change legal meaning (``may``, ``shall``, ``must``,
  ``shall not``, ``unless``, ``except``, ``subject to``, ``provided that``).

Numbers are normalized before comparison (commas stripped, so ``$1,000``
matches ``$1000``; leading zeros preserved as-is since ``007`` rarely means
``7`` in a contract).  Qualifiers are matched case-insensitively on word
boundaries.  Ordinary prose is NOT checked — paraphrase is allowed; inventing
facts is not.

Limits, stated plainly: this catches claims that restate a number or flip a
qualifier.  It does not catch new facts expressed without numbers, nor
spelled-out amounts ("two million") that differ textually from the quote's
"2,000,000".  The judged-faithfulness step covers what this cannot.
"""

from __future__ import annotations

import re
from typing import Final

# Phrases whose presence or absence changes what a clause permits.
QUALIFIER_WORDS: Final[tuple[str, ...]] = (
    "shall not",
    "subject to",
    "provided that",
    "shall",
    "must",
    "may",
    "unless",
    "except",
)

_MONTHS: Final[str] = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)

_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DATE_NUMERIC_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"
)
_DATE_MONTH_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTHS})(?:\s*,?\s*\d{{4}})?\b",
    re.IGNORECASE,
)
_CURRENCY_RE: Final[re.Pattern[str]] = re.compile(
    r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?"
    r"|\b(?:USD|EUR|GBP)\s?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)


def _word_boundary(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)


def _normalize_number(raw: str) -> str:
    """Compare numbers by digits: commas are formatting, not meaning."""
    digits = raw.replace(",", "")
    # "$2,000,000" and "USD 2000000" should both reduce to "2000000"
    digits = re.sub(r"^[$€£¥]", "", digits).strip()
    digits = re.sub(r"^(?:USD|EUR|GBP)\s*", "", digits, flags=re.IGNORECASE)
    return digits


def extract_literals(text: str) -> list[str]:
    """Return the checkable literals in claim text, in first-appearance order.

    Duplicates are kept (a claim stating "30 ... 30" twice backs two facts).
    Currency amounts normalize to their digit sequence; dates and qualifier
    phrases keep their written form for substring comparison against quotes.
    """
    found: list[tuple[int, str]] = []

    for match in _DATE_NUMERIC_RE.finditer(text):
        found.append((match.start(), match.group(0)))
    for match in _DATE_MONTH_RE.finditer(text):
        found.append((match.start(), match.group(0)))
    for match in _CURRENCY_RE.finditer(text):
        found.append((match.start(), _normalize_number(match.group(0))))
    covered: list[tuple[int, int]] = [
        (m.start(), m.end()) for rx in (_DATE_NUMERIC_RE, _DATE_MONTH_RE, _CURRENCY_RE) for m in rx.finditer(text)
    ]
    for match in _NUMBER_RE.finditer(text):
        start = match.start()
        if any(s <= start < e for s, e in covered):
            continue  # digits already claimed by a date or currency match
        found.append((start, _normalize_number(match.group(0))))

    # Process qualifiers longest-first so multi-word phrases shadow sub-words.
    # e.g. "shall not" must shadow the single-word "shall" match inside it.
    qual_covered: list[tuple[int, int]] = []
    for word in sorted(QUALIFIER_WORDS, key=len, reverse=True):
        for match in _word_boundary(word).finditer(text):
            ms, me = match.start(), match.end()
            if any(s <= ms < e for s, e in qual_covered):
                continue  # already claimed by a longer phrase
            found.append((ms, word))
            qual_covered.append((ms, me))

    found.sort(key=lambda item: item[0])
    return [value for _, value in found]


def violations(claim_text: str, quote: str) -> list[str]:
    """Return literals from *claim_text* missing from *quote* (empty = clean).

    Numbers are compared as digit tokens with commas stripped, so the quote's
    ``$2,000,000`` agrees with the claim's ``$2,000,000``.  Words are matched
    case-insensitively on whitespace-normalised text.
    """
    lowered = " ".join(quote.lower().split())
    quote_digits = [tok.replace(",", "") for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", lowered)]
    quote_words = lowered.split()
    missing: list[str] = []
    for literal in extract_literals(claim_text):
        needle = " ".join(literal.lower().split())
        if re.fullmatch(r"[\d.]+", needle):
            if needle not in quote_digits:
                missing.append(literal)
        elif needle not in lowered:
            if " " in needle:
                words = needle.split()
                if not any(
                    words[i] in quote_words and words[i + 1] in quote_words
                    for i in range(len(words) - 1)
                ):
                    missing.append(literal)
            else:
                missing.append(literal)
    return missing


def is_faithful(claim_text: str, quote: str) -> bool:
    """True when *claim_text* has zero literal violations against *quote*."""
    return not violations(claim_text, quote)


__all__ = [
    "QUALIFIER_WORDS",
    "extract_literals",
    "is_faithful",
    "violations",
]
