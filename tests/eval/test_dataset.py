"""Tests for clausewright.eval.dataset.

Uses a tiny synthetic CUAD-shaped fixture; the real 40MB file is only touched
by the marked ``real_cuad`` test (skipped automatically when not fetched).
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from clausewright.eval.dataset import (
    CUAD_SMALL,
    EvalItem,
    SuiteConfig,
    build_eval_set,
    dataset_hash,
    save_manifests,
)


def _write_fixture(
    root: Path, n_contracts: int = 4, categories: tuple[str, ...] = ("Cat A", "Cat B")
) -> Path:
    """Write a CUAD-shaped json; every contract missing a pdf named 'no_pdf_*'."""
    data = []
    for idx in range(n_contracts):
        title = f"no_pdf_contract_{idx}" if idx == 0 else f"contract_{idx}"
        qas = []
        for cat in categories:
            absent = cat == "Cat B" and idx % 2 == 0
            qas.append(
                {
                    "id": f"{title}__{cat}",
                    "question": f"question for {cat}",
                    "is_impossible": absent,
                    "answers": []
                    if absent
                    else [{"text": f"gold-{cat}-{idx}", "answer_start": 10}],
                }
            )
        data.append({"title": title, "paragraphs": [{"context": "x", "qas": qas}]})
    path = root / "CUAD_v1.json"
    path.write_text(json.dumps({"version": "test", "data": data}), encoding="utf-8")
    return path


def _pdf_root(root: Path) -> Path:
    pdfs = root / "full_contract_pdf"
    pdfs.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        (pdfs / f"contract_{idx}.pdf").write_bytes(b"%PDF-1.4 fake")
    return pdfs


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    _pdf_root(tmp_path)
    return tmp_path


def test_requires_annotations_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CUAD annotations not found"):
        build_eval_set(
            CUAD_SMALL,
            json_path=tmp_path / "missing.json",
            pdf_root=tmp_path,
        )


def test_excludes_contracts_without_pdf(fixture_root: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("Cat A", "Cat B"), seed=1)
    es = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    titles = {i.contract_title for i in es.items}
    assert "no_pdf_contract_0" not in titles
    assert es.n_contracts == 3


def test_absent_clause_maps_to_abstention_item(fixture_root: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("Cat A", "Cat B"), seed=1)
    es = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    absent = [i for i in es.items if i.clause_absent]
    present = [i for i in es.items if not i.clause_absent]
    assert all(i.category == "Cat B" for i in absent)
    assert all(i.gold_texts == () for i in absent)
    assert present and all(i.gold_texts for i in present)


def test_split_by_contract_no_leakage(fixture_root: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("Cat A",), seed=7, dev_fraction=0.5)
    es = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    dev_c = {i.contract_title for i in es.by_split("dev")}
    test_c = {i.contract_title for i in es.by_split("test")}
    assert dev_c and test_c
    assert not dev_c & test_c


def test_deterministic_selection(fixture_root: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("Cat A",), seed=7)
    a = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    b = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    assert [i.item_id for i in a.items] == [i.item_id for i in b.items]


def test_seed_changes_selection(fixture_root: Path) -> None:
    args = {
        "json_path": fixture_root / "CUAD_v1.json",
        "pdf_root": fixture_root / "full_contract_pdf",
    }
    a = build_eval_set(SuiteConfig(name="t", n_contracts=2, categories=("Cat A",), seed=1), **args)
    b = build_eval_set(SuiteConfig(name="t", n_contracts=2, categories=("Cat A",), seed=7), **args)
    assert [i.item_id for i in a.items] != [i.item_id for i in b.items]


def test_dataset_hash_is_stable(fixture_root: Path) -> None:
    p = fixture_root / "CUAD_v1.json"
    assert dataset_hash(p) == dataset_hash(p)
    assert len(dataset_hash(p)) == 32


def test_unknown_category_rejected(fixture_root: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("No Such Category",), seed=1)
    with pytest.raises(ValueError, match="absent from CUAD"):
        build_eval_set(
            suite,
            json_path=fixture_root / "CUAD_v1.json",
            pdf_root=fixture_root / "full_contract_pdf",
        )


def test_eval_item_is_frozen() -> None:
    item = EvalItem(
        item_id="x",
        split="dev",
        contract_title="t",
        pdf_path=Path("/x.pdf"),
        category="c",
        question="q",
        gold_texts=("g",),
        clause_absent=False,
    )
    with pytest.raises(FrozenInstanceError):
        item.question = "changed"  # type: ignore[misc]


def test_save_manifests_creates_valid_json(fixture_root: Path, tmp_path: Path) -> None:
    suite = SuiteConfig(name="t", n_contracts=3, categories=("Cat A", "Cat B"), seed=1)
    es = build_eval_set(
        suite,
        json_path=fixture_root / "CUAD_v1.json",
        pdf_root=fixture_root / "full_contract_pdf",
    )
    dev_path, test_path = save_manifests(es, output_dir=tmp_path / "manifests")
    assert dev_path.is_file()
    assert test_path.is_file()

    dev_data = json.loads(dev_path.read_text(encoding="utf-8"))
    assert dev_data["split"] == "dev"
    assert dev_data["n_contracts"] > 0
    assert len(dev_data["items"]) == dev_data["n_items"]
    assert "sha256" in dev_data["contracts"][0]


@pytest.mark.real_cuad
def test_real_cuad_builds() -> None:
    es = build_eval_set(CUAD_SMALL)
    assert len(es.items) == 450
    assert es.n_contracts == 30
    assert {i.split for i in es.items} == {"dev", "test"}
