"""Evaluation runner: unified retrieval + answer evaluation with reproducibility.

Coordinates full evaluation runs, enforces reproducible envelopes (git SHA,
dataset hash, timestamps, version, config), and logs runs to `data/eval/runs.jsonl`.
Provides a ReplayLLMClient to record/replay model completions for zero API spend.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausewright.config import PATHS
from clausewright.eval.answer_metrics import answer_quality_report
from clausewright.eval.dataset import (
    CUAD_SMALL,
    EvalItem,
    EvalSet,
    _find_default_cuad_paths,
    build_eval_set,
    dataset_hash,
)
from clausewright.eval.retrieval import CONFIGS, evaluate_retrieval
from clausewright.ingestion import chunk_sections, load_pdf, parse_sections
from clausewright.qa.engine import AnswerEngine
from clausewright.qa.llm import LLMClient
from clausewright.qa.schemas import AnswerResult
from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.reranker import CrossEncoderReranker


def get_git_sha() -> str:
    """Current HEAD SHA for reproducibility; 'unknown' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.strip().encode("utf-8"))
    h.update(b"\n---\n")
    h.update(user.strip().encode("utf-8"))
    return h.hexdigest()


class ReplayLLMClient:
    """LLM client that checks a disk cache first, with budget enforcement.

    Parameters
    ----------
    cache_path:
        Path to JSON or JSONL file of cached exchanges.
    max_calls:
        Max real LLM calls permitted before falling back to offline abstention.
    underlying_client:
        Real LLMClient (e.g. AnthropicClient). If None, cache misses abstain.
    model_name:
        Reported model name identifier.
    """

    def __init__(
        self,
        cache_path: Path | str | None = None,
        max_calls: int = 0,
        underlying_client: LLMClient | None = None,
        model_name: str = "claude-sonnet-4-5",
    ) -> None:
        self._cache_path = Path(cache_path) if cache_path else None
        self._max_calls = max_calls
        self._underlying = underlying_client
        self._model_name = model_name
        self._calls_made = 0
        self._cache: dict[str, str] = {}
        self._load_cache()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def calls_made(self) -> int:
        return self._calls_made

    def _load_cache(self) -> None:
        if not self._cache_path or not self._cache_path.is_file():
            return
        text = self._cache_path.read_text(encoding="utf-8")
        if self._cache_path.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    k = entry.get("key") or entry.get("prompt_hash")
                    v = entry.get("response") or entry.get("completion")
                    if k and v:
                        self._cache[k] = v
                except json.JSONDecodeError:
                    continue
        else:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    self._cache.update(data)
            except json.JSONDecodeError:
                pass

    def _save_entry(self, key: str, response: str) -> None:
        self._cache[key] = response
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self._cache_path.suffix == ".jsonl":
            entry = json.dumps({"key": key, "response": response}) + "\n"
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(entry)
        else:
            self._cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def complete(self, *, system: str, user: str) -> str:
        key = _prompt_hash(system, user)
        if key in self._cache:
            return self._cache[key]

        if self._calls_made >= self._max_calls or self._underlying is None:
            # Fallback offline abstention response
            return json.dumps({
                "answerable": False,
                "reason_if_unanswerable": (
                    "No LLM calls allowed (--max-llm-calls budget reached or cache miss)."
                ),
            })

        response = self._underlying.complete(system=system, user=user)
        self._calls_made += 1
        self._save_entry(key, response)
        return response


def evaluate_answers(
    eval_set: EvalSet,
    *,
    split: str = "dev",
    llm_client: LLMClient,
    sample_contracts: int | None = None,
    min_rerank_score: float | None = None,
) -> dict[str, object]:
    """Run AnswerEngine across all items in *split* and evaluate answer quality."""
    items = eval_set.by_split(split)  # type: ignore[arg-type]
    by_contract: dict[str, list[EvalItem]] = defaultdict(list)
    for it in items:
        by_contract[it.contract_title].append(it)

    titles = sorted(by_contract.keys())
    if sample_contracts:
        titles = titles[:sample_contracts]

    embedder = Embedder()
    reranker = CrossEncoderReranker()

    results: list[AnswerResult] = []
    true_absents: list[bool] = []

    for title in titles:
        c_items = by_contract[title]
        doc = load_pdf(c_items[0].pdf_path, doc_id=title)
        sections = parse_sections(doc)
        chunks = chunk_sections(title, sections)
        retriever = HybridRetriever.from_chunks(chunks, embedder=embedder)
        engine = AnswerEngine(
            retriever=retriever,
            reranker=reranker,
            llm=llm_client,
            min_rerank_score=min_rerank_score,
        )

        for item in c_items:
            res = engine.answer(item.question)
            results.append(res)
            true_absents.append(not item.is_answerable)

    if not results:
        return {}

    return answer_quality_report(results, true_absent=true_absents)


def run_full_eval(
    *,
    split: str = "dev",
    configs: tuple[str, ...] = CONFIGS,
    sample_contracts: int | None = None,
    max_llm_calls: int = 0,
    replay_cache: Path | str | None = None,
    output_path: Path | str | None = None,
    run_log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute full evaluation run, print results table, and record reproducibility envelope."""
    try:
        from importlib.metadata import version as pkg_version

        cw_version = pkg_version("clausewright")
    except Exception:  # noqa: BLE001
        cw_version = "dev"

    eval_set = build_eval_set(CUAD_SMALL)
    cuad_json, _ = _find_default_cuad_paths()
    ds_hash = dataset_hash(cuad_json) if cuad_json.is_file() else "unknown"
    git_sha = get_git_sha()

    # 1. Retrieval evaluation
    retrieval_res = evaluate_retrieval(
        eval_set, split=split, configs=configs, sample_contracts=sample_contracts
    )

    # 2. Answer evaluation with replay client
    llm = ReplayLLMClient(cache_path=replay_cache, max_calls=max_llm_calls)
    answer_res = evaluate_answers(
        eval_set, split=split, llm_client=llm, sample_contracts=sample_contracts
    )

    full_record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "clausewright_version": cw_version,
        "model_id": llm.model_name,
        "split": split,
        "dataset_hash": ds_hash,
        "config": {
            "retrieval_configs": configs,
            "max_llm_calls": max_llm_calls,
            "replay_cache": str(replay_cache) if replay_cache else None,
            "sample_contracts": sample_contracts,
        },
        "retrieval_metrics": retrieval_res.get("configs", {}),
        "answer_metrics": answer_res,
    }

    # Append to runs log (data/eval/runs.jsonl)
    runs_log = Path(run_log_path) if run_log_path else PATHS.root / "data" / "eval" / "runs.jsonl"
    runs_log.parent.mkdir(parents=True, exist_ok=True)
    with runs_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(full_record) + "\n")

    # Save to single output JSON if requested
    out = Path(output_path) if output_path else PATHS.root / "data" / "eval" / "retrieval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(full_record, indent=2), encoding="utf-8")

    return full_record
