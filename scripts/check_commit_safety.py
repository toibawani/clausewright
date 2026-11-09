#!/usr/bin/env python3
"""Block commits that carry secrets, corpus data, or oversized files.

Run by pre-commit with the staged filenames as arguments, or manually with
``--all`` to walk the working tree.  Exits 1 and prints every finding when a
file should not be committed.

The checks are deliberately narrow: a bare ``secret`` or ``key`` appears in
ordinary code and prose, so only credential-shaped strings are matched.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

MAX_FILE_BYTES: Final[int] = 512 * 1024

# Corpus, index, cache, and result paths are reproducible or contract-derived;
# they never belong in git.
FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    "data/raw/",
    "data/index/",
    "data/store/",
    "data/eval/",
    "data/llm_cache/",
    "results/",
)
FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset({".env"})

SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("AWS access key id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)

# A matched line containing one of these words is a documented placeholder.
PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "example",
    "placeholder",
    "change-me",
    "change_me",
    "insecure",
    "redacted",
    "xxxx",
    "your-key",
)

SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
)


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, so findings read like ``data/x.pdf``."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix().removeprefix("./")


def check_path(path: Path) -> list[str]:
    """Return human-readable problems with *path*; empty means safe to commit."""
    problems: list[str] = []
    rel = _display_path(path)
    name = path.name

    if name in FORBIDDEN_NAMES:
        problems.append(f"{rel}: environment file must not be committed")
    if any(rel.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        problems.append(f"{rel}: corpus, index, or result path must not be committed")
    if not path.is_file():
        return problems

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        problems.append(
            f"{rel}: {size} bytes exceeds the {MAX_FILE_BYTES} byte commit limit"
        )

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return problems

    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                problems.append(f"{rel}:{line_number}: possible {label}")
    return problems


def _walk(root: Path) -> Iterable[Path]:
    """Yield files that could plausibly be committed.

    Ignored corpus/index trees are skipped here: they are expected to exist
    locally, and pre-commit still blocks them if someone forces one in.
    """
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if any(rel.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            continue
        if path.is_file():
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to scan (pre-commit passes these)")
    parser.add_argument("--all", action="store_true", help="scan the whole working tree")
    args = parser.parse_args(argv)

    if args.all:
        targets: list[Path] = list(_walk(Path.cwd()))
    else:
        targets = [Path(p) for p in args.paths]

    findings: list[str] = []
    for target in targets:
        findings.extend(check_path(target))

    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        print(
            f"\nrefusing to commit: {len(findings)} problem(s) found",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
