"""Run provenance: refuse to measure from a dirty tree, and record the source.

Why this exists
---------------
An eval that runs while the working tree is being edited or switched measures
code that matches no commit. The number is then unreproducible: nobody can say
which chunker, which tie-break, or which prompt produced it. That happened
once on this project, so the rule is now enforced by the runner rather than by
memory.

Every result JSON carries a ``provenance`` block: branch, full commit, clean
flag, interpreter, model names, and the threading settings that materially
change throughput and can change floating-point tie-breaks.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

#: Set to any non-empty value to run from a dirty tree deliberately (debugging
#: only). Result files record the flag, so such runs stay visibly provisional.
ALLOW_DIRTY_ENV = "CLAUSEWRIGHT_ALLOW_DIRTY"


class DirtyTreeError(RuntimeError):
    """Raised when a measurement is attempted from an uncommitted tree."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"


def git_commit(full: bool = True) -> str:
    args = ("rev-parse", "HEAD") if full else ("rev-parse", "--short", "HEAD")
    return _git(*args) or "unknown"


def git_status_lines() -> list[str]:
    output = _git("status", "--porcelain")
    return [line for line in output.splitlines() if line.strip()]


def is_clean_tree() -> bool:
    """True when nothing is modified, staged, or untracked in the working tree."""
    return not git_status_lines()


def require_clean_tree() -> None:
    """Abort the run unless the tree is clean, or the caller opted out.

    Raises
    ------
    DirtyTreeError
        When ``git status --porcelain`` reports any entry and
        ``CLAUSEWRIGHT_ALLOW_DIRTY`` is unset.
    """
    if os.environ.get(ALLOW_DIRTY_ENV):
        return
    dirty = git_status_lines()
    if dirty:
        preview = "; ".join(dirty[:5])
        raise DirtyTreeError(
            "refusing to run an eval from a dirty tree "
            f"({len(dirty)} changed paths, first: {preview}). "
            "Commit the code or run from a worktree at a known commit. "
            f"Set {ALLOW_DIRTY_ENV}=1 only for a debug run."
        )


def thread_settings() -> dict[str, str]:
    """Threading knobs that change speed and can change ties."""
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "KMP_DUPLICATE_LIB_OK",
    )
    return {key: os.environ.get(key, "unset") for key in keys}


def provenance(
    *,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    reranker_batch_size: int | None = None,
) -> dict[str, Any]:
    """Return the block written into every result file."""
    import clausewright

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "branch": git_branch(),
        "commit": git_commit(full=True),
        "commit_short": git_commit(full=False),
        "clean_tree": is_clean_tree(),
        "dirty_override": bool(os.environ.get(ALLOW_DIRTY_ENV)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "clausewright_path": str(clausewright.__file__),
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "reranker_batch_size": reranker_batch_size,
        "threads": thread_settings(),
    }
