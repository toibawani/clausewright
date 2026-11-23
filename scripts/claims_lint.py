"""Keep claims out of the docs that the measurements do not support.

Two mechanical rules, both run by `tests/test_claims_lint.py`:

1. `README.md` and `docs/` may not contain the vocabulary of a pitch
   ("excellent", "best", "production-ready", and so on). Every one of those
   words is a claim about quality, and quality here is whatever the tables say.
2. `README.md` may not contain a metric number outside a generated block. A
   number in prose is a copy that no test regenerates, so it drifts from the
   table it came from.

Generated blocks (the ones between `<!-- BEGIN GENERATED: name -->` and its end
marker) are exempt from rule 2: they are produced from the result store.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Words that assert quality rather than report it.
BANNED_WORDS = (
    "excellent",
    "production-ready",
    "elite",
    "best",
    "state-of-the-art",
    "recruiter-ready",
)

#: A metric number: three or more decimals, e.g. 0.6302 or 3.141.
METRIC_NUMBER = re.compile(r"(?<![\w.])\d*\.\d{3,}(?![\w])")

#: A percentage with a decimal, e.g. 87.5%.
PERCENTAGE = re.compile(r"(?<![\w.])\d{1,3}\.\d+\s?%")

BLOCK = re.compile(
    r"<!-- BEGIN GENERATED: (?P<name>[a-z0-9_]+) -->.*?<!-- END GENERATED: (?P=name) -->",
    re.DOTALL,
)


def generated_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by generated blocks."""
    return [(match.start(), match.end()) for match in BLOCK.finditer(text)]


def _in_spans(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def banned_word_hits(text: str) -> list[tuple[int, str]]:
    """Positions and words that break the vocabulary rule."""
    hits: list[tuple[int, str]] = []
    lowered = text.lower()
    for word in BANNED_WORDS:
        for match in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lowered):
            hits.append((match.start(), word))
    return sorted(hits)


def number_hits(text: str) -> list[tuple[int, str]]:
    """Metric numbers and decimal percentages outside generated blocks."""
    spans = generated_spans(text)
    hits: list[tuple[int, str]] = []
    for pattern in (METRIC_NUMBER, PERCENTAGE):
        for match in pattern.finditer(text):
            if not _in_spans(match.start(), spans):
                hits.append((match.start(), match.group(0)))
    return sorted(hits)


def _line_of(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def lint_docs(root: Path) -> list[str]:
    """Check every document; return one message per violation."""
    problems: list[str] = []
    readme = root / "README.md"
    documents = [readme, *sorted((root / "docs").rglob("*.md"))]

    for path in documents:
        if not path.is_file():
            problems.append(f"{path} is missing")
            continue
        text = path.read_text()
        name = path.relative_to(root)
        for position, word in banned_word_hits(text):
            problems.append(f"{name}:{_line_of(text, position)}: banned word {word!r}")
        if path == readme:
            for position, number in number_hits(text):
                problems.append(
                    f"{name}:{_line_of(text, position)}: metric number {number!r} "
                    "outside a generated block"
                )
    return problems
