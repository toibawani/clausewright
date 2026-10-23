"""Tests for clausewright.eval.runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from clausewright.eval.runner import (
    ReplayLLMClient,
    _prompt_hash,
    get_git_sha,
)


class TestGetGitSha:
    def test_git_sha_returns_string(self) -> None:
        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0


class TestPromptHash:
    def test_deterministic(self) -> None:
        h1 = _prompt_hash("sys prompt", "user prompt")
        h2 = _prompt_hash("sys prompt", "user prompt")
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex


class TestReplayLLMClient:
    def test_empty_cache_with_zero_calls_returns_abstention(self) -> None:
        client = ReplayLLMClient(max_calls=0)
        res = client.complete(system="sys", user="user")
        data = json.loads(res)
        assert data["answerable"] is False
        assert client.calls_made == 0

    def test_cache_hit_returns_cached_text_without_calls(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.jsonl"
        key = _prompt_hash("sys", "user")
        cache_file.write_text(
            json.dumps({"key": key, "response": '{"answerable": true, "claims": []}'}) + "\n",
            encoding="utf-8",
        )
        mock_underlying = MagicMock()
        client = ReplayLLMClient(cache_path=cache_file, max_calls=0, underlying_client=mock_underlying)
        res = client.complete(system="sys", user="user")
        assert '{"answerable": true' in res
        mock_underlying.complete.assert_not_called()
        assert client.calls_made == 0

    def test_calls_underlying_when_budget_permits(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.jsonl"
        mock_underlying = MagicMock()
        mock_underlying.complete.return_value = '{"answerable": false}'
        client = ReplayLLMClient(cache_path=cache_file, max_calls=1, underlying_client=mock_underlying)

        res = client.complete(system="sys", user="user")
        assert res == '{"answerable": false}'
        assert client.calls_made == 1
        mock_underlying.complete.assert_called_once()

        # Second call exceeds budget of 1
        res2 = client.complete(system="sys2", user="user2")
        data2 = json.loads(res2)
        assert data2["answerable"] is False
        assert client.calls_made == 1
