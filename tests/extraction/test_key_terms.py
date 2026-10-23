"""Unit tests for KeyTermsExtractor (issue #30)."""

from __future__ import annotations

from clausewright.extraction import KeyTermsExtractor, KeyTermsReport

# ---------------------------------------------------------------------------
# Stub engine
# ---------------------------------------------------------------------------


class _StubAnswer:
    """Minimal duck-type for clausewright.qa.schemas.Answer."""

    def __init__(self, text: str, chunk_id: str = "c1") -> None:
        self.text = text

        class _Citation:
            def __init__(self, cid: str) -> None:
                self.chunk_id = cid

        class _Claim:
            def __init__(self, t: str, cid: str) -> None:
                self.text = t
                self.citations = [_Citation(cid)]

        self.claims = (_Claim(text, chunk_id),)


class _StubAbstention:
    """Minimal duck-type for clausewright.qa.schemas.Abstention."""


class _AnswerEngine:
    """Configurable stub engine: maps question substring to result."""

    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping

    def answer(self, question: str) -> object:
        for key, result in self._mapping.items():
            if key.lower() in question.lower():
                return result
        return _StubAbstention()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_all_found():
    engine = _AnswerEngine(
        {
            "parties": _StubAnswer("Party A and Party B"),
            "duration": _StubAnswer("3 years"),  # won't match 'term' key
            "term": _StubAnswer("3 year term"),
            "renewal": _StubAnswer("auto-renews annually"),
            "termination": _StubAnswer("30 days written notice"),
            "liability": _StubAnswer("$1,000,000 aggregate cap"),
            "governing": _StubAnswer("New York law"),
            "payment": _StubAnswer("Net 30 days"),
        }
    )
    extractor = KeyTermsExtractor(engine)
    report = extractor.extract()

    assert not report.parties.abstained
    assert report.parties.value == "Party A and Party B"

    assert not report.term.abstained
    assert "3 year" in (report.term.value or "")

    assert not report.renewal.abstained
    assert not report.termination_notice.abstained
    assert not report.liability_cap.abstained
    assert not report.governing_law.abstained
    assert not report.payment_terms.abstained


def test_extract_partial_abstention():
    """Some fields found, some abstained."""
    engine = _AnswerEngine(
        {
            "parties": _StubAnswer("Acme Corp and Beta Inc"),
            # All other fields will be Abstention.
        }
    )
    extractor = KeyTermsExtractor(engine)
    report = extractor.extract()

    assert not report.parties.abstained
    assert report.term.abstained
    assert report.renewal.abstained
    assert report.governing_law.abstained


def test_extract_all_abstained():
    engine = _AnswerEngine({})  # always abstains
    extractor = KeyTermsExtractor(engine)
    report = extractor.extract()

    for fname in (
        "parties", "term", "renewal", "termination_notice",
        "liability_cap", "governing_law", "payment_terms",
    ):
        assert getattr(report, fname).abstained, f"{fname} should be abstained"


def test_extract_citation_chunk_ids_populated():
    engine = _AnswerEngine(
        {"parties": _StubAnswer("Party A", chunk_id="chunk-42")}
    )
    extractor = KeyTermsExtractor(engine)
    report = extractor.extract()
    assert "chunk-42" in report.parties.citation_chunk_ids


def test_format_table_renders():
    report = KeyTermsReport()
    table = report.format_table()
    assert "parties" in table
    assert "governing_law" in table


def test_as_dict_keys():
    report = KeyTermsReport()
    d = report.as_dict()
    expected_keys = {
        "parties", "term", "renewal", "termination_notice",
        "liability_cap", "governing_law", "payment_terms",
    }
    assert set(d.keys()) == expected_keys
    for v in d.values():
        assert "value" in v
        assert "abstained" in v
        assert "citation_chunk_ids" in v


def test_engine_exception_causes_field_abstention(monkeypatch):
    """An engine that throws should not abort the whole extraction."""

    class _RaisingEngine:
        def answer(self, question: str) -> object:
            raise RuntimeError("model crash")

    extractor = KeyTermsExtractor(_RaisingEngine())
    report = extractor.extract()
    # All fields should be abstained, no exception propagated.
    for fname in ("parties", "term", "renewal"):
        assert getattr(report, fname).abstained
