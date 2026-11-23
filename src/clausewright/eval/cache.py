"""On-disk caches for embeddings and reranker scores.

Why
---
This machine has 8 GB of RAM and one usable core per job. Recomputing the same
chunk embedding or the same (query, chunk) cross-encoder score in every cell of
a sweep is the difference between a run that finishes and one that does not.

Keys
----
* Embedding: ``sha256(model name | normalise flag | text)``.
* Reranker score: ``sha256(model name | query | text)``.

The model name is part of the key, so swapping the embedder cannot silently
reuse stale vectors. Text is hashed, never stored, so the cache does not keep
contract text on disk beyond the vectors themselves.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from clausewright.config import PATHS
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


class _SqliteCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> bytes | None:
        row = self._conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put_many(self, items: dict[str, bytes]) -> None:
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)",
            list(items.items()),
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return int(row[0]) if row else 0


class CachedEmbedder(Embedder):
    """``Embedder`` that reuses vectors by (model, text) across runs."""

    def __init__(self, *, cache_dir: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        directory = cache_dir or (PATHS.root / "data" / "cache")
        self._cache = _SqliteCache(directory / f"embeddings-{self.model_name}.sqlite3")
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return str(self._model_name)

    def embed(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        if not texts:
            raise ValueError("texts must be non-empty")

        keys = [_digest(self.model_name, str(normalize), text) for text in texts]
        cached: dict[str, np.ndarray] = {}
        missing: list[int] = []
        for index, key in enumerate(keys):
            raw = self._cache.get(key)
            if raw is None:
                missing.append(index)
                continue
            cached[key] = np.frombuffer(raw, dtype=np.float32)

        self.hits += len(texts) - len(missing)
        self.misses += len(missing)

        if missing:
            vectors = super().embed([texts[i] for i in missing], normalize=normalize)
            payload: dict[str, bytes] = {}
            for position, index in enumerate(missing):
                vector = np.ascontiguousarray(vectors[position], dtype=np.float32)
                payload[keys[index]] = vector.tobytes()
                cached[keys[index]] = vector
            self._cache.put_many(payload)

        return np.vstack([cached[key] for key in keys])


class CachedReranker(CrossEncoderReranker):
    """``CrossEncoderReranker`` that reuses scores by (model, query, text)."""

    def __init__(self, *, cache_dir: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        directory = cache_dir or (PATHS.root / "data" / "cache")
        self._cache = _SqliteCache(directory / f"rerank-{self.model_name}.sqlite3")
        self.hits = 0
        self.misses = 0

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        *,
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        keys = [_digest(self.model_name, query, chunk.text) for chunk, _ in candidates]
        scores: dict[str, float] = {}
        missing: list[int] = []
        for index, key in enumerate(keys):
            raw = self._cache.get(key)
            if raw is None:
                missing.append(index)
                continue
            scores[key] = float(np.frombuffer(raw, dtype=np.float32)[0])

        self.hits += len(candidates) - len(missing)
        self.misses += len(missing)

        if missing:
            pairs = [(query, candidates[i][0].text) for i in missing]
            raw_scores: Any = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=False,
            )
            payload: dict[str, bytes] = {}
            for position, index in enumerate(missing):
                value = float(raw_scores[position])
                scores[keys[index]] = value
                payload[keys[index]] = np.asarray([value], dtype=np.float32).tobytes()
            self._cache.put_many(payload)

        scored = [(chunk, scores[key]) for (chunk, _), key in zip(candidates, keys)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n] if top_n is not None else scored
