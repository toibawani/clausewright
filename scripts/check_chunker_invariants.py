#!/usr/bin/env python3
"""Check chunker invariants against the real CUAD contracts.

Reads PDFs from ``data/raw/cuad`` (gitignored) and reports aggregate counts
only; no contract text is written anywhere. Exits non-zero when any invariant
is violated.

Invariants checked per contract:

1. With non-overlapping windows (``stride == max_words``) the concatenation of
   chunk text equals the document text exactly: no words lost or duplicated.
2. Every non-overlapping chunk is a contiguous slice of the document, and its
   ``section_number`` is the section that owns its first word.
3. With the default overlapping windows, each chunk's words appear in document
   order.
4. With the default overlapping windows, no chunk is below ``MIN_CHUNK_WORDS``
   unless the whole document is smaller than the floor.
5. Chunking is deterministic.

Usage:
    python scripts/check_chunker_invariants.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clausewright.ingestion.chunker import MIN_CHUNK_WORDS, chunk_sections
from clausewright.ingestion.pdf_loader import load_pdf
from clausewright.ingestion.section_parser import parse_sections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUAD_PDF_DIRS = (
    PROJECT_ROOT / "data" / "raw" / "cuad" / "CUAD_v1" / "full_contract_pdf",
    PROJECT_ROOT / "data" / "raw" / "CUAD_v1" / "full_contract_pdf",
)


def check_contract(pdf: Path, doc_index: int) -> tuple[list[str], int, int]:
    """Return (violations, section_count, chunk_count) for one contract."""
    doc = load_pdf(pdf, doc_id=f"cuad-{doc_index}")
    sections = parse_sections(doc)
    if not sections:
        return [], 0, 0

    # Keep sections as a list: section numbers are not unique in real contracts,
    # so keying by number would silently drop blocks.
    per_section = [
        [word for block in section.blocks for word in block.text.split()] for section in sections
    ]
    document_words = [word for words in per_section for word in words]
    violations: list[str] = []

    # 1 + 2: non-overlapping reconstruction and provenance.
    disjoint = chunk_sections(doc.doc_id, sections, max_words=300, stride=300)
    if " ".join(chunk.text for chunk in disjoint).split() != document_words:
        violations.append("words lost or duplicated")

    section_of_index: list[str] = []
    for section, words in zip(sections, per_section, strict=True):
        section_of_index.extend([section.section_number] * len(words))

    cursor = 0
    for chunk in disjoint:
        size = len(chunk.text.split())
        start = cursor
        if document_words[start : start + size] != chunk.text.split():
            violations.append("chunk is not a contiguous document slice")
            break
        if start >= len(section_of_index) or section_of_index[start] != chunk.section_number:
            violations.append("section provenance does not match first word")
            break
        cursor += size

    # 3 + 4: default overlapping windows preserve order and respect the floor.
    default = chunk_sections(doc.doc_id, sections)
    for chunk in default:
        pos = 0
        for word in chunk.text.split():
            while pos < len(document_words) and document_words[pos] != word:
                pos += 1
            if pos >= len(document_words):
                violations.append("word order not preserved")
                break
            pos += 1

    if len(document_words) >= MIN_CHUNK_WORDS:
        for chunk in default:
            if chunk.word_count < MIN_CHUNK_WORDS:
                violations.append("chunk below MIN_CHUNK_WORDS")
                break

    # 5: determinism.
    again = chunk_sections(doc.doc_id, sections)
    if [(c.chunk_id, c.text) for c in default] != [(c.chunk_id, c.text) for c in again]:
        violations.append("non-deterministic output")

    return violations, len(sections), len(default)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="check only the first N PDFs")
    args = parser.parse_args()

    cuad_pdf_dir = next((path for path in CUAD_PDF_DIRS if path.is_dir()), None)
    if cuad_pdf_dir is None:
        print(f"missing CUAD PDFs under {CUAD_PDF_DIRS[0]}", file=sys.stderr)
        return 1

    pdfs = sorted(cuad_pdf_dir.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    contracts = sections_total = chunks_total = skipped = 0
    failures: list[str] = []

    for index, pdf in enumerate(pdfs, start=1):
        try:
            violations, n_sections, n_chunks = check_contract(pdf, index)
        except ValueError:
            skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            failures.append(f"{pdf.name}: could not check ({exc})")
            continue

        contracts += 1
        sections_total += n_sections
        chunks_total += n_chunks
        failures.extend(f"{pdf.name}: {violation}" for violation in violations)

    print(f"contracts checked: {contracts}")
    print(f"contracts skipped (no extractable text): {skipped}")
    print(f"sections: {sections_total}")
    print(f"chunks (default 300/150): {chunks_total}")
    print(f"violations: {len(failures)}")
    for failure in failures[:20]:
        print(f"  {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
