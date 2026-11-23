"""Freeze a contract-disjoint held-out CUAD split and gate every look at it.

Why this exists
---------------
``dev`` and ``test`` are drawn once from 30 contracts (``CUAD_SMALL``), and
every retrieval decision in Part B was made by looking at ``dev``.  A choice
made against dev can no longer be confirmed on dev: the interval that matters
is the one on contracts the choice never saw.  This module freezes that set
exactly once, pins the identity of every document in it by sha256, and refuses
to run against it except in a declared final run that is written down.

Rules encoded here
------------------
* **Disjoint by construction.**  The draw excludes every title in the
  cuad-small dev *and* test split before sampling, so no held-out contract can
  also be in dev or test.
* **Deterministic.**  Sorted eligible titles, a seeded shuffle with its own
  seed (recorded in the manifest), then the first ``n_contracts``.
* **Frozen once.**  ``freeze`` refuses to overwrite a manifest that differs
  from the one on disk; the committed manifest is the split.
* **Closed unless declared.**  A held-out run needs ``CLAUSEWRIGHT_FINAL_RUN=1``
  and the frozen finalist file.  Anything else is refused before a model loads.
* **Auditable.**  Every permitted run appends a dated row to
  ``docs/heldout-log.md`` with the commit and the configuration, so anyone can
  count how many times the held-out set has been looked at.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.config import PATHS
from clausewright.eval.dataset import (
    CUAD_SMALL,
    DEFAULT_CATEGORIES,
    ContractIndex,
    EvalSet,
    SuiteConfig,
    _find_default_cuad_paths,
    build_eval_set,
    build_items,
    dataset_hash,
    file_sha256,
    load_contract_index,
)

#: A fresh seed: it must not be the seed that produced dev/test, or the draws
#: would be the same draw.
HELDOUT_SEED: int = 20261123

#: 30 unseen contracts costs about the same as one dev run and keeps the
#: per-contract interval width in the same range as dev.
HELDOUT_CONTRACTS: int = 30

#: The only thing that unlocks the split: an explicit, declared final run.
FINAL_RUN_ENV: str = "CLAUSEWRIGHT_FINAL_RUN"

MANIFEST_PATH: Path = PATHS.root / "benchmarks" / "heldout_manifest.json"
FINALIST_PATH: Path = PATHS.root / "benchmarks" / "heldout_finalist.json"
LOG_PATH: Path = PATHS.root / "docs" / "heldout-log.md"


class HeldoutRefused(RuntimeError):
    """Raised when the held-out split is read outside a declared final run."""


def _relpath(path: Path, root: Path) -> str:
    """Path relative to the project root, or the absolute path if outside it."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def file_list_sha256(contracts: list[dict[str, Any]]) -> str:
    """sha256 over the sorted ``"<sha256>  <path>"`` lines of the file list.

    This is the identity of the split: if one document is swapped, renamed or
    re-fetched with different bytes, the digest changes and ``verify_manifest``
    says so instead of quietly measuring a different set.
    """
    lines = [
        f"{c['sha256']}  {c['pdf_relative_path']}"
        for c in sorted(contracts, key=lambda c: str(c["pdf_relative_path"]))
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def draw_titles(
    index: ContractIndex,
    *,
    excluded: set[str],
    seed: int = HELDOUT_SEED,
    n_contracts: int = HELDOUT_CONTRACTS,
) -> list[str]:
    """Deterministically draw *n_contracts* titles that are not in *excluded*."""
    candidates = sorted(t for t in index.per_contract if t not in excluded)
    if len(candidates) < n_contracts:
        raise ValueError(
            f"only {len(candidates)} eligible contracts remain after excluding "
            f"{len(excluded)}; cannot draw {n_contracts}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:n_contracts])


def build_manifest(
    *,
    seed: int = HELDOUT_SEED,
    n_contracts: int = HELDOUT_CONTRACTS,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    json_path: Path | None = None,
    pdf_root: Path | None = None,
    root: Path = PATHS.root,
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict for the held-out split.

    ``excluded`` defaults to every title in the cuad-small dev and test split.
    It is a parameter only so tests can use a tiny synthetic corpus.
    """
    default_json, default_pdf = _find_default_cuad_paths()
    json_path = json_path or default_json
    pdf_root = pdf_root or default_pdf
    index = load_contract_index(json_path, pdf_root)

    if excluded is None:
        occupied = build_eval_set(CUAD_SMALL, json_path=json_path, pdf_root=pdf_root)
        excluded = {item.contract_title for item in occupied.items}

    titles = draw_titles(index, excluded=excluded, seed=seed, n_contracts=n_contracts)

    contracts: list[dict[str, Any]] = []
    for title in titles:
        pdf = index.pdf_for(title)
        contracts.append(
            {
                "contract_title": title,
                "pdf_relative_path": _relpath(pdf, root),
                "sha256": file_sha256(pdf) if pdf.is_file() else "missing",
                "categories_annotated": sum(
                    1 for category in categories if category in index.per_contract[title]
                ),
            }
        )

    return {
        "split": "heldout",
        "schema": 1,
        "suite": f"cuad-heldout-{n_contracts}c",
        "seed": seed,
        "n_contracts": len(titles),
        "categories": list(categories),
        "dataset_hash": dataset_hash(json_path),
        "excluded": {
            "suite": CUAD_SMALL.name,
            "seed": CUAD_SMALL.seed,
            "n_contracts": len(excluded),
            "rule": "every contract in the cuad-small dev or test split",
        },
        "contracts": contracts,
        "file_list_sha256": file_list_sha256(contracts),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the frozen manifest, refusing when it was never written."""
    if not manifest_path.is_file():
        raise HeldoutRefused(
            f"no held-out manifest at {manifest_path}; run "
            "`python -m clausewright.eval heldout --freeze` once"
        )
    loaded: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return loaded


def _stable_part(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest minus its timestamp: what "the same split" means."""
    return {k: v for k, v in manifest.items() if k != "frozen_at"}


def freeze(
    *,
    manifest_path: Path = MANIFEST_PATH,
    seed: int = HELDOUT_SEED,
    n_contracts: int = HELDOUT_CONTRACTS,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    json_path: Path | None = None,
    pdf_root: Path | None = None,
    root: Path = PATHS.root,
    excluded: set[str] | None = None,
) -> tuple[Path, str]:
    """Write the manifest once; refuse to change an already frozen split.

    Returns the manifest path and its file-list digest.
    """
    manifest = build_manifest(
        seed=seed,
        n_contracts=n_contracts,
        categories=categories,
        json_path=json_path,
        pdf_root=pdf_root,
        root=root,
        excluded=excluded,
    )
    if manifest_path.is_file():
        existing = load_manifest(manifest_path)
        if _stable_part(existing) != _stable_part(manifest):
            raise HeldoutRefused(
                f"{manifest_path} is frozen and differs from a fresh draw; "
                "a held-out split is not re-drawn once it exists. Delete is not "
                "allowed here either: raise an issue and change the seed in a "
                "new manifest path instead."
            )
        return manifest_path, str(existing["file_list_sha256"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, str(manifest["file_list_sha256"])


def verify_manifest(
    manifest: dict[str, Any],
    *,
    root: Path = PATHS.root,
) -> None:
    """Check the file list digest and every document hash on disk."""
    expected = file_list_sha256(manifest["contracts"])
    if manifest.get("file_list_sha256") != expected:
        raise HeldoutRefused(
            "held-out manifest file list does not match its own digest; the "
            "manifest was edited after freezing"
        )
    for contract in manifest["contracts"]:
        path = root / str(contract["pdf_relative_path"])
        if not path.is_file():
            raise HeldoutRefused(f"held-out document missing: {path}")
        if file_sha256(path) != contract["sha256"]:
            raise HeldoutRefused(f"held-out document changed on disk: {path}")


def require_final_run(
    *,
    finalist_path: Path | str | None = None,
    manifest_path: Path = MANIFEST_PATH,
    root: Path = PATHS.root,
) -> dict[str, Any]:
    """Unlock the held-out split, or refuse with the reason.

    Both conditions are required: the environment flag marks the run as
    declared, and the finalist path must be the frozen configuration file, so a
    stray command cannot measure the split with an exploratory setting.
    """
    if os.environ.get(FINAL_RUN_ENV) != "1":
        raise HeldoutRefused(
            "the held-out split is closed. It may only be read in a declared "
            f"final run: set {FINAL_RUN_ENV}=1 and pass "
            f"--finalist {FINALIST_PATH.name}, and the run will be logged in "
            f"{_relpath(LOG_PATH, PATHS.root)}"
        )

    if finalist_path is None:
        raise HeldoutRefused(
            "a held-out run must name the frozen finalist file explicitly: "
            f"--finalist {FINALIST_PATH}. The path is checked against the frozen "
            "configuration; it is never inferred, so no run reaches the split by "
            "setting an environment variable alone."
        )

    finalist = Path(finalist_path)
    if finalist.resolve() != FINALIST_PATH.resolve():
        raise HeldoutRefused(
            f"a held-out run must use the frozen finalist file {FINALIST_PATH}, "
            f"not {finalist}. Choose the configuration on dev, write it there once, "
            "and do not edit it afterwards."
        )
    if not finalist.is_file():
        raise HeldoutRefused(f"frozen finalist file not found: {finalist}")

    manifest = load_manifest(manifest_path)
    verify_manifest(manifest, root=root)
    return manifest


def build_heldout_eval_set(
    manifest_path: Path = MANIFEST_PATH,
    *,
    root: Path = PATHS.root,
) -> EvalSet:
    """Materialise the frozen split as an :class:`EvalSet` with split ``heldout``.

    Callers must have passed :func:`require_final_run` first; this function
    only builds what the manifest already nailed down.
    """
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest, root=root)

    default_json, default_pdf = _find_default_cuad_paths()
    index = load_contract_index(default_json, default_pdf)
    categories = tuple(str(c) for c in manifest["categories"])
    titles = [str(c["contract_title"]) for c in manifest["contracts"]]

    missing = [t for t in titles if t not in index.per_contract]
    if missing:
        raise HeldoutRefused(
            f"{len(missing)} held-out contracts have no annotations on disk: {missing[:3]}"
        )

    items = build_items(index, titles, categories, {t: "heldout" for t in titles})
    return EvalSet(
        suite=SuiteConfig(
            name=str(manifest["suite"]),
            n_contracts=len(titles),
            categories=categories,
            seed=int(manifest["seed"]),
        ),
        dataset_hash=str(manifest["dataset_hash"]),
        items=items,
    )


def finalist_settings() -> dict[str, Any]:
    """The frozen configuration a declared final run is allowed to use."""
    if not FINALIST_PATH.is_file():
        raise HeldoutRefused(f"frozen finalist file not found: {FINALIST_PATH}")
    loaded: dict[str, Any] = json.loads(FINALIST_PATH.read_text(encoding="utf-8"))
    return loaded


def record_look(
    *,
    configuration: str,
    result_path: Path | str | None = None,
    note: str = "",
    log_path: Path = LOG_PATH,
) -> None:
    """Append one row to the held-out log. Append-only by construction."""
    from clausewright.eval.provenance import git_commit

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    row = (
        f"| {stamp} | `{git_commit(full=False)}` | {configuration} "
        f"| {result_path or '-'} | {note or '-'} |\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def describe_command(argv: list[str]) -> str:
    """A shell-quoted form of the command, for the log row."""
    return " ".join(shlex.quote(a) for a in argv)
