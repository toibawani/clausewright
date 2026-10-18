"""Build the eval set from CUAD clause annotations.

Sources
-------
``data/raw/CUAD_v1/CUAD_v1.json`` (SQuAD-style, from the checksum-verified
Zenodo zip) and the 199 PDFs shipped alongside it.  Annotations cover 510
contracts but only 197 titles have a PDF, so eligibility = "PDF present".

Design decisions
----------------
* **One question per category.**  CUAD ships exactly one question string per
  category across all 510 contracts (verified), so the category question is
  the eval question.  Gold = all ``answers[].text`` spans for that qa entry.
* **Absent clause = abstention test.**  ``is_impossible: true`` means the
  annotators found no such clause; we treat that question as a case the system
  should abstain on.
* **Deterministic split by contract.**  Seeded shuffle over sorted eligible
  titles, then a prefix goes to dev and the remainder to test.  Nothing is
  ever tuned on test.
* **No model-generated gold.**  Every gold span comes from CUAD's human
  annotators; adversarial cases are hand-written YAML, never model output.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clausewright.config import PATHS

# Default 15 categories: mix of present-heavy and absent-heavy so both
# answerable and abstention metrics have support on 30 contracts.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Governing Law",
    "Non-Compete",
    "Exclusivity",
    "Termination For Convenience",
    "Anti-Assignment",
    "Change Of Control",
    "Ip Ownership Assignment",
    "Cap On Liability",
    "Insurance",
    "Audit Rights",
    "License Grant",
)

Split = Literal["dev", "test"]


@dataclass(frozen=True)
class SuiteConfig:
    """A named, reproducible eval set definition."""

    name: str
    n_contracts: int
    categories: tuple[str, ...]
    seed: int
    dev_fraction: float = 2.0 / 3.0


#: The default small suite: 30 contracts x 15 categories = 450 questions.
CUAD_SMALL = SuiteConfig(
    name="cuad-small",
    n_contracts=30,
    categories=DEFAULT_CATEGORIES,
    seed=20261018,
)

SUITES: dict[str, SuiteConfig] = {CUAD_SMALL.name: CUAD_SMALL}


@dataclass(frozen=True)
class EvalItem:
    """One question about one contract, with human gold spans."""

    item_id: str            # CUAD qa id: "<title>__<category>"
    split: Split
    contract_title: str
    pdf_path: Path
    category: str
    question: str
    gold_texts: tuple[str, ...]   # normalized-on-use answer spans
    clause_absent: bool           # is_impossible -> abstention test

    @property
    def is_answerable(self) -> bool:
        return not self.clause_absent


@dataclass(frozen=True)
class EvalSet:
    suite: SuiteConfig
    dataset_hash: str        # md5 of the CUAD json actually used
    items: tuple[EvalItem, ...]

    def by_split(self, split: Split) -> tuple[EvalItem, ...]:
        return tuple(i for i in self.items if i.split == split)

    @property
    def n_contracts(self) -> int:
        return len({i.contract_title for i in self.items})


def _normalize_golds(answers: list[dict[str, object]]) -> tuple[str, ...]:
    """Unique, order-preserving gold span texts."""
    seen: list[str] = []
    for a in answers:
        text = str(a.get("text", "")).strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def build_eval_set(
    suite: SuiteConfig,
    *,
    json_path: Path | None = None,
    pdf_root: Path | None = None,
) -> EvalSet:
    """Materialise the eval set for *suite* from the CUAD files on disk.

    Selection is fully deterministic for a given suite config: eligible
    contracts (annotations + PDF present) are sorted by title, shuffled with
    the suite seed, and the first ``n_contracts`` are taken.  The dev/test
    split is over contracts — no contract appears in both.

    Raises
    ------
    FileNotFoundError
        If the CUAD annotation file or PDF root is missing (run fetch_cuad.py).
    ValueError
        If fewer than ``n_contracts`` eligible contracts exist.
    """
    json_path = json_path or (PATHS.raw_dir / "CUAD_v1" / "CUAD_v1.json")
    pdf_root = pdf_root or (PATHS.raw_dir / "CUAD_v1" / "full_contract_pdf")
    if not json_path.is_file():
        raise FileNotFoundError(f"CUAD annotations not found: {json_path} (run scripts/fetch_cuad.py)")
    if not pdf_root.is_dir():
        raise FileNotFoundError(f"CUAD pdf dir not found: {pdf_root} (run scripts/fetch_cuad.py)")

    with json_path.open(encoding="utf-8") as f:
        raw: dict[str, object] = json.load(f)
    data = raw["data"]  # type: ignore[index]
    assert isinstance(data, list)

    pdf_by_stem = {p.stem: p for p in pdf_root.rglob("*.pdf")}

    # title -> {category -> qa entry}; keep only contracts with a PDF.
    per_contract: dict[str, dict[str, dict[str, object]]] = {}
    for item in data:
        assert isinstance(item, dict)
        title = str(item["title"])
        pdf = pdf_by_stem.get(title)
        if pdf is None:
            continue
        cat_map: dict[str, dict[str, object]] = {}
        for para in item["paragraphs"]:  # type: ignore[index]
            for qa in para["qas"]:
                category = str(qa["id"]).rsplit("__", 1)[-1]
                cat_map[category] = qa
        per_contract[title] = cat_map

    missing = [c for c in suite.categories if not any(c in m for m in per_contract.values())]
    if missing:
        raise ValueError(f"categories absent from CUAD entirely: {missing}")

    titles = sorted(per_contract)
    rng = random.Random(suite.seed)
    rng.shuffle(titles)
    if len(titles) < suite.n_contracts:
        raise ValueError(
            f"only {len(titles)} contracts have both annotations and a PDF; "
            f"suite {suite.name} wants {suite.n_contracts}"
        )
    selected = sorted(titles[: suite.n_contracts])
    n_dev = max(1, round(len(selected) * suite.dev_fraction))
    split_of: dict[str, Split] = {t: ("dev" if idx < n_dev else "test") for idx, t in enumerate(selected)}

    items: list[EvalItem] = []
    for title in selected:
        cat_map = per_contract[title]
        pdf = pdf_by_stem[title]
        for category in suite.categories:
            qa = cat_map.get(category)
            if qa is None:
                continue  # category not annotated for this contract
            answers = qa.get("answers", [])
            golds = _normalize_golds(answers) if isinstance(answers, list) else ()
            absent = bool(qa.get("is_impossible", False))
            if absent and golds:
                # CUAD guarantees impossible => empty answers; defend anyway.
                absent = False
            items.append(
                EvalItem(
                    item_id=str(qa["id"]),
                    split=split_of[title],
                    contract_title=title,
                    pdf_path=pdf,
                    category=category,
                    question=str(qa["question"]),
                    gold_texts=golds,
                    clause_absent=absent,
                )
            )

    return EvalSet(
        suite=suite,
        dataset_hash=dataset_hash(json_path),
        items=tuple(items),
    )


def dataset_hash(json_path: Path) -> str:
    """md5 of the annotation file; recorded in every result file."""
    h = hashlib.md5()
    with json_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
