"""Tests for the frozen held-out split and its final-run guard.

Synthetic corpus for the mechanics; one ``real_cuad`` test checks that the
committed manifest is actually disjoint from the dev and test contracts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clausewright.eval import heldout
from clausewright.eval.dataset import CUAD_SMALL, build_eval_set, load_contract_index

EXCLUDED = {"contract_00", "contract_01", "contract_02"}


def _write_fixture(root: Path, n: int = 12) -> tuple[Path, Path]:
    data = []
    for i in range(n):
        title = f"contract_{i:02d}"
        data.append(
            {
                "title": title,
                "paragraphs": [
                    {
                        "context": "x",
                        "qas": [
                            {
                                "id": f"{title}__Cat A",
                                "question": "q",
                                "is_impossible": False,
                                "answers": [{"text": f"gold-{i}", "answer_start": 0}],
                            }
                        ],
                    }
                ],
            }
        )
    json_path = root / "CUAD_v1.json"
    json_path.write_text(json.dumps({"version": "t", "data": data}), encoding="utf-8")

    pdf_root = root / "full_contract_pdf"
    pdf_root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (pdf_root / f"contract_{i:02d}.pdf").write_bytes(f"%PDF-1.4 fake {i}".encode())
    return json_path, pdf_root


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    return _write_fixture(tmp_path)


def _draw(
    corpus: tuple[Path, Path],
    *,
    seed: int = heldout.HELDOUT_SEED,
    n: int = 5,
) -> list[str]:
    json_path, pdf_root = corpus
    index = load_contract_index(json_path, pdf_root)
    return heldout.draw_titles(index, excluded=EXCLUDED, seed=seed, n_contracts=n)


def _build(corpus: tuple[Path, Path], tmp_path: Path, **overrides: object) -> dict[str, object]:
    json_path, pdf_root = corpus
    kwargs: dict[str, object] = {
        "json_path": json_path,
        "pdf_root": pdf_root,
        "root": tmp_path,
        "excluded": EXCLUDED,
        "categories": ("Cat A",),
        "n_contracts": 5,
    }
    kwargs.update(overrides)
    return heldout.build_manifest(**kwargs)  # type: ignore[arg-type]


def test_draw_excludes_dev_and_test_contracts(corpus: tuple[Path, Path]) -> None:
    titles = _draw(corpus)

    assert len(titles) == 5
    assert not set(titles) & EXCLUDED
    assert titles == sorted(titles)


def test_draw_is_deterministic_per_seed(corpus: tuple[Path, Path]) -> None:
    assert _draw(corpus) == _draw(corpus)
    assert _draw(corpus, seed=1) != _draw(corpus)


def test_draw_refuses_when_too_few_contracts(corpus: tuple[Path, Path]) -> None:
    with pytest.raises(ValueError, match="cannot draw"):
        _draw(corpus, n=50)


def test_manifest_pins_hashes_and_a_file_list_digest(
    corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    manifest = _build(corpus, tmp_path)
    contracts = manifest["contracts"]

    assert isinstance(contracts, list)
    assert manifest["n_contracts"] == len(contracts) == 5
    assert manifest["seed"] == heldout.HELDOUT_SEED
    assert manifest["file_list_sha256"] == heldout.file_list_sha256(contracts)
    assert all(c["sha256"] != "missing" for c in contracts)
    assert all(not Path(c["pdf_relative_path"]).is_absolute() for c in contracts)


def test_file_list_digest_changes_with_the_documents(tmp_path: Path) -> None:
    a = [{"sha256": "aa", "pdf_relative_path": "data/a.pdf"}]
    b = [{"sha256": "bb", "pdf_relative_path": "data/a.pdf"}]
    c = [{"sha256": "aa", "pdf_relative_path": "data/b.pdf"}]

    assert heldout.file_list_sha256(a) != heldout.file_list_sha256(b)
    assert heldout.file_list_sha256(a) != heldout.file_list_sha256(c)


def test_freeze_writes_once_and_refuses_a_second_draw(
    corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    json_path, pdf_root = corpus
    target = tmp_path / "benchmarks" / "heldout_manifest.json"
    kwargs: dict[str, object] = {
        "manifest_path": target,
        "json_path": json_path,
        "pdf_root": pdf_root,
        "root": tmp_path,
        "excluded": EXCLUDED,
        "categories": ("Cat A",),
        "n_contracts": 5,
    }

    path, digest = heldout.freeze(**kwargs)  # type: ignore[arg-type]
    assert path.is_file()
    assert digest == json.loads(path.read_text(encoding="utf-8"))["file_list_sha256"]
    first = path.read_text(encoding="utf-8")

    heldout.freeze(**kwargs)  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == first

    with pytest.raises(heldout.HeldoutRefused, match="frozen"):
        heldout.freeze(**{**kwargs, "seed": 7})  # type: ignore[arg-type]


def test_verify_manifest_detects_a_changed_document(
    corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    manifest = _build(corpus, tmp_path)
    heldout.verify_manifest(manifest, root=tmp_path)

    victim = tmp_path / str(manifest["contracts"][0]["pdf_relative_path"])  # type: ignore[index]
    victim.write_bytes(b"%PDF-1.4 tampered")

    with pytest.raises(heldout.HeldoutRefused, match="changed on disk"):
        heldout.verify_manifest(manifest, root=tmp_path)


def test_final_run_refused_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(heldout.FINAL_RUN_ENV, raising=False)

    with pytest.raises(heldout.HeldoutRefused, match="held-out split is closed"):
        heldout.require_final_run()


def test_final_run_refused_without_an_explicit_finalist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(heldout.FINAL_RUN_ENV, "1")

    with pytest.raises(heldout.HeldoutRefused, match="name the frozen finalist"):
        heldout.require_final_run()


def test_final_run_refused_with_a_different_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(heldout.FINAL_RUN_ENV, "1")
    frozen = tmp_path / "finalist.json"
    frozen.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(heldout, "FINALIST_PATH", frozen)

    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")

    with pytest.raises(heldout.HeldoutRefused, match="frozen finalist file"):
        heldout.require_final_run(finalist_path=other)


def test_final_run_unlocks_with_flag_and_frozen_finalist(
    corpus: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path, pdf_root = corpus
    target = tmp_path / "heldout_manifest.json"
    frozen = tmp_path / "finalist.json"
    frozen.write_text("{}", encoding="utf-8")

    heldout.freeze(
        manifest_path=target,
        json_path=json_path,
        pdf_root=pdf_root,
        root=tmp_path,
        excluded=EXCLUDED,
        categories=("Cat A",),
        n_contracts=5,
    )
    monkeypatch.setenv(heldout.FINAL_RUN_ENV, "1")
    monkeypatch.setattr(heldout, "FINALIST_PATH", frozen)
    monkeypatch.setattr(heldout, "_find_default_cuad_paths", lambda: (json_path, pdf_root))

    manifest = heldout.require_final_run(
        finalist_path=frozen, manifest_path=target, root=tmp_path
    )
    assert manifest["split"] == "heldout"

    eval_set = heldout.build_heldout_eval_set(target, root=tmp_path)
    assert {item.split for item in eval_set.items} == {"heldout"}
    assert len({item.contract_title for item in eval_set.items}) == 5


def test_record_look_appends_rows(tmp_path: Path) -> None:
    log = tmp_path / "heldout-log.md"
    log.write_text("| header |\n", encoding="utf-8")

    heldout.record_look(
        configuration="struct+hybrid_rerank",
        result_path="data/eval/x.json",
        note="pytest",
        log_path=log,
    )
    heldout.record_look(configuration="second", log_path=log)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "struct+hybrid_rerank" in lines[1]
    assert "second" in lines[2]


_json_path, _pdf_dir = heldout._find_default_cuad_paths()
_cuad_present = _json_path.is_file() and _pdf_dir.is_dir()


@pytest.mark.real_cuad
@pytest.mark.skipif(not _cuad_present, reason="CUAD dataset not fetched under data/raw")
def test_committed_manifest_is_disjoint_from_dev_and_test() -> None:
    manifest = heldout.load_manifest()
    occupied = {item.contract_title for item in build_eval_set(CUAD_SMALL).items}
    held = {str(c["contract_title"]) for c in manifest["contracts"]}

    assert len(held) == 30
    assert not held & occupied
    heldout.verify_manifest(manifest)
