"""Tests for clausewright.qa.llm.

The SDK is imported lazily inside ``AnthropicClient.__init__``; these tests
mock it so no network or real key is ever involved.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from clausewright.qa.llm import AnthropicClient, LLMClient, LLMError


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``anthropic`` module that records the constructor key."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[])
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    module = ModuleType("anthropic")
    module.Anthropic = mock_anthropic.Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return mock_anthropic


class TestAnthropicClientConstruction:
    def test_missing_key_raises_before_sdk_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            AnthropicClient()

    def test_env_key_reaches_sdk(self, fake_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        client = AnthropicClient()
        fake_sdk.Anthropic.assert_called_once_with(api_key="env-key")
        assert client.model_name == "claude-sonnet-4-5"

    def test_explicit_key_wins_over_env(
        self, fake_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        AnthropicClient(api_key="explicit-key")
        fake_sdk.Anthropic.assert_called_once_with(api_key="explicit-key")


class TestAnthropicClientComplete:
    def _client(self, fake_sdk: MagicMock, text_blocks: list[object]) -> AnthropicClient:
        block = MagicMock()
        block.type = "text"
        block.text = text_blocks[0] if text_blocks else ""
        fake_sdk.Anthropic.return_value.messages.create.return_value = MagicMock(
            content=[block] if text_blocks else []
        )
        return AnthropicClient(api_key="k")

    def test_returns_joined_text(
        self, fake_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        client = self._client(fake_sdk, ["hello world"])
        assert client.complete(system="s", user="u") == "hello world"

    def test_empty_completion_raises(self, fake_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        client = self._client(fake_sdk, [])
        with pytest.raises(LLMError, match="empty"):
            client.complete(system="s", user="u")

    def test_api_error_wrapped(
        self, fake_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        fake_sdk.Anthropic.return_value.messages.create.side_effect = RuntimeError("boom")
        client = AnthropicClient(api_key="k")
        with pytest.raises(LLMError, match="failed"):
            client.complete(system="s", user="u")


class TestProtocol:
    def test_fake_client_satisfies_protocol(self) -> None:
        class Fake:
            @property
            def model_name(self) -> str:
                return "fake"

            def complete(self, *, system: str, user: str) -> str:
                return "ok"

        client: LLMClient = Fake()
        assert client.complete(system="s", user="u") == "ok"
        assert isinstance(client, LLMClient)

