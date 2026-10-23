"""Build the eval set from CUAD clause annotations.

Sources
-------
``data/raw/cuad/CUAD_v1.json`` or ``data/raw/CUAD_v1/CUAD_v1.json`` (SQuAD-style, from
the checksum-verified Zenodo zip) and the 199 PDFs shipped alongside it.
Annotations cover 510 contracts but only 197 titles have a PDF, so eligibility = "PDF present".

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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

    item_id: str  # CUAD qa id: "<title>__<category>"
    split: Split
    contract_title: str
    pdf_path: Path
    category: str
    question: str
    gold_texts: tuple[str, ...]  # normalized-on-use answer spans
    clause_absent: bool  # is_impossible -> abstention test

    @property
    def is_answerable(self) -> bool:
        return not self.clause_absent

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "split": self.split,
            "contract_title": self.contract_title,
            "pdf_path": str(self.pdf_path),
            "category": self.category,
            "question": self.question,
            "gold_texts": list(self.gold_texts),
            "clause_absent": self.clause_absent,
            "is_answerable": self.is_answerable,
        }


@dataclass(frozen=True)
class EvalSet:
    suite: SuiteConfig
    dataset_hash: str  # md5 of the CUAD json actually used
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


def _find_default_cuad_paths() -> tuple[Path, Path]:
    """Locate annotations and pdf dir under data/raw/."""
    candidates_json = [
        PATHS.raw_dir / "cuad" / "CUAD_v1.json",
        PATHS.raw_dir / "CUAD_v1" / "CUAD_v1.json",
    ]
    candidates_pdf = [
        PATHS.raw_dir / "cuad" / "full_contract_pdf",
        PATHS.raw_dir / "CUAD_v1" / "full_contract_pdf",
    ]
    json_path = next((p for p in candidates_json if p.is_file()), candidates_json[0])
    pdf_root = next((p for p in candidates_pdf if p.is_dir()), candidates_pdf[0])
    return json_path, pdf_root


def file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a document file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    """
    default_json, default_pdf = _find_default_cuad_paths()
    json_path = json_path or default_json
    pdf_root = pdf_root or default_pdf

    if not json_path.is_file():
        raise FileNotFoundError(
            f"CUAD annotations not found: {json_path} (run scripts/download_cuad.py)"
        )
    if not pdf_root.is_dir():
        raise FileNotFoundError(
            f"CUAD pdf dir not found: {pdf_root} (run scripts/download_cuad.py)"
        )

    with json_path.open(encoding="utf-8") as f:
        raw: dict[str, object] = json.load(f)
    data = raw["data"]  # type: ignore[index]
    assert isinstance(data, list)

    pdf_by_stem = {p.stem: p for p in pdf_root.rglob("*.pdf")}

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
    split_of: dict[str, Split] = {
        t: ("dev" if idx < n_dev else "test") for idx, t in enumerate(selected)
    }

    items: list[EvalItem] = []
    for title in selected:
        cat_map = per_contract[title]
        pdf = pdf_by_stem[title]
        for category in suite.categories:
            qa = cat_map.get(category)
            if qa is None:
                continue
            answers = qa.get("answers", [])
            golds = _normalize_golds(answers) if isinstance(answers, list) else ()
            absent = bool(qa.get("is_impossible", False))
            if absent and golds:
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


def save_manifests(eval_set: EvalSet, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Serialize dev and test manifests into JSON files with document hashes."""
    out = output_dir or (PATHS.root / "data" / "eval")
    out.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for split in ("dev", "test"):
        split_items = eval_set.by_split(split)  # type: ignore[arg-type]
        contracts: dict[str, dict[str, Any]] = {}
        for item in split_items:
            if item.contract_title not in contracts:
                doc_hash = file_sha256(item.pdf_path) if item.pdf_path.is_file() else "missing"
                contracts[item.contract_title] = {
                    "contract_title": item.contract_title,
                    "pdf_path": str(item.pdf_path),
                    "sha256": doc_hash,
                    "categories_annotated": 0,
                }
            contracts[item.contract_title]["categories_annotated"] += 1

        manifest = {
            "split": split,
            "suite": asdict(eval_set.suite),
            "dataset_hash": eval_set.dataset_hash,
            "n_contracts": len(contracts),
            "n_items": len(split_items),
            "n_answerable": sum(1 for i in split_items if i.is_answerable),
            "n_abstention": sum(1 for i in split_items if i.clause_absent),
            "contracts": list(contracts.values()),
            "items": [i.to_dict() for i in split_items],
        }
        target = out / f"{split}_manifest.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        results.append(target)

    return results[0], results[1]


def dataset_hash(json_path: Path) -> str:
    """md5 of the annotation file; recorded in every result file."""
    h = hashlib.md5()
    with json_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
