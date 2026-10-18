"""LLM client abstraction: protocol + Anthropic implementation.

The rest of the QA layer depends only on :class:`LLMClient`, so tests inject a
fake and never touch the network.  The Anthropic client:

* reads the API key from ``ANTHROPIC_API_KEY`` (never from code or a repo file),
* never logs prompts or completions (customer contract text flows through both),
* sends the system prompt separately from the user message so document content
  cannot be mistaken for instructions at the API level.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion surface the QA engine needs."""

    @property
    def model_name(self) -> str:
        """Identifier recorded on answers for audit."""
        ...

    def complete(self, *, system: str, user: str) -> str:
        """Return the model's raw completion for one system/user exchange."""
        ...


class LLMError(RuntimeError):
    """Raised when the model call fails or returns nothing usable."""


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API.

    Parameters
    ----------
    model_name:
        Model to call.  Defaults to ``claude-sonnet-4-5``.
    api_key:
        Explicit key; falls back to the ``ANTHROPIC_API_KEY`` environment
        variable.  Never hardcode a key in source.
    max_tokens:
        Cap on completion length.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Export it in your environment or "
                "pass api_key explicitly."
            )
        # Imported lazily so the QA layer is importable without the SDK installed.
        import anthropic

        self._client = anthropic.Anthropic(api_key=key)
        self._model_name = model_name
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, *, system: str, user: str) -> str:
        """Call the Messages API.  Logs timing only — never prompt content."""
        try:
            response = self._client.messages.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            # Log the exception class and SDK message only; no payloads.
            logger.error("Anthropic call failed: %s: %s", type(exc).__name__, exc)
            raise LLMError("The language model call failed.") from exc

        parts = [block.text for block in response.content if block.type == "text"]
        text = "".join(parts).strip()
        if not text:
            raise LLMError("The language model returned an empty completion.")
        return text
