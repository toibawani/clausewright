"""Tests for run provenance and cache keying."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clausewright.eval import provenance as prov
from clausewright.eval.cache import _digest, _SqliteCache


def test_require_clean_tree_raises_when_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prov.ALLOW_DIRTY_ENV, raising=False)
    monkeypatch.setattr(prov, "git_status_lines", lambda: [" M src/x.py"])

    with pytest.raises(prov.DirtyTreeError):
        prov.require_clean_tree()


def test_require_clean_tree_passes_on_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prov.ALLOW_DIRTY_ENV, raising=False)
    monkeypatch.setattr(prov, "git_status_lines", list)

    prov.require_clean_tree()


def test_dirty_override_allows_the_run_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(prov.ALLOW_DIRTY_ENV, "1")
    monkeypatch.setattr(prov, "git_status_lines", lambda: [" M src/x.py"])

    prov.require_clean_tree()
    block = prov.provenance()

    assert block["dirty_override"] is True
    assert block["clean_tree"] is False


def test_provenance_records_commit_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(prov, "git_status_lines", list)
    monkeypatch.setattr(prov, "git_commit", lambda full=True: "abc123" if full else "abc123"[:6])

    block = prov.provenance(embedding_model="m", reranker_model="r", reranker_batch_size=64)

    assert block["commit"] == "abc123"
    assert block["threads"]["OMP_NUM_THREADS"] == "1"
    assert block["reranker_batch_size"] == 64


def test_cache_keys_separate_models_and_texts() -> None:
    assert _digest("m1", "text") != _digest("m2", "text")
    assert _digest("m1", "text") != _digest("m1", "other")
    assert _digest("m1", "text") == _digest("m1", "text")


def test_vector_roundtrip_through_cache(tmp_path: Path) -> None:
    cache = _SqliteCache(tmp_path / "c.sqlite3")
    vector = np.asarray([0.5, -0.25], dtype=np.float32)

    cache.put_many({"k": vector.tobytes()})

    assert np.frombuffer(cache.get("k"), dtype=np.float32).tolist() == pytest.approx([0.5, -0.25])
