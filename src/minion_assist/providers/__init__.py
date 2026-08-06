"""Public API and factory for the providers subsystem.

This module is the single import point for anything related to LLM providers.
It re-exports all provider classes and types, and provides the
:func:`create_provider` factory function that picks the right implementation
based on a string key from the config.

Factory pattern
---------------
:func:`create_provider` implements the **Factory pattern**: instead of callers
writing ``if api == "anthropic": return AnthropicProvider(...) else: ...``
everywhere, all that branching logic lives here. Callers just call
``create_provider(api="anthropic", ...)`` and get the right object back.

Adding a new provider
----------------------
1. Create ``src/minion_assist/providers/myprovider.py`` with a class that has a
   ``chat()`` method matching :class:`LLMProvider`.
2. Import it here and add a ``case "my-api":`` branch in the ``match`` below.
3. Add the provider config to ``config.json``.

Talks to
--------
- ``minion.py`` calls :func:`create_provider` once per agent at startup.
- ``runner.py`` uses the returned provider to call the LLM.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .anthropic import AnthropicProvider
from .base import LLMProvider, LLMResponse, TokenUsage, ToolCall
from .codex import CodexProvider
from .lmstudio import LMStudioProvider
from .openai_compatible import OpenAICompatibleProvider
from ..config import codex_cfg

if TYPE_CHECKING:
    from ..tools import ToolRegistry


def create_provider(
    api: str,
    base_url: str,
    api_key: str,
    model: str,
    log_dir: Path | None = None,
    registry: "ToolRegistry | None" = None,
    approve_command: Callable[[str, dict], str] | None = None,
) -> LLMProvider:
    """Instantiate the correct provider class for the given API type.

    This is the only place in the codebase that knows which ``api`` string
    maps to which implementation class. All other code uses the returned
    object via the :class:`LLMProvider` protocol.

    Args:
        api (str): The adapter type string from ``config.json``, e.g.
            ``"openai-completions"``, ``"codex"``, ``"lmstudio"``,
            or ``"anthropic"``.
        base_url (str): HTTP endpoint for the provider. Unused for Codex
            (the binary manages auth) and Anthropic.
        api_key (str): Authentication token loaded from ``.env``.
        model (str): Model identifier string for API requests.
        log_dir (Path | None): When set, every request and response is appended
            to ``log_dir/YYYY-MM-DD.log`` in LM Studio's server log format.
            ``None`` disables logging (default).
        registry (ToolRegistry | None): Tool registry whose tools are exposed
            to Codex as dynamic tools.  Ignored by non-Codex providers (they
            receive tool schemas via ``chat()`` and return tool_calls for the
            runner to dispatch).

    Returns:
        LLMProvider: A fully constructed provider ready to call ``.chat()``.

    Notes:
        Unknown ``api`` values fall through to :class:`OpenAICompatibleProvider`
        as a convenient default, so any OpenAI-compatible endpoint works even
        without an explicit ``case`` branch.
    """
    match api:
        case "anthropic":
            # Anthropic uses a different SDK and message format; see anthropic.py.
            return AnthropicProvider(api_key=api_key, model=model, log_dir=log_dir)
        case "lmstudio":
            # LMStudioProvider is an alias for OpenAICompatibleProvider; same behavior.
            return LMStudioProvider(base_url=base_url, api_key=api_key, model=model, log_dir=log_dir)
        case "codex":
            # Codex app-server — auth injected via stored OAuth token (run codex-login).
            # base_url and api_key are ignored.
            codex_bin = os.environ.get("CODEX_BIN", "").strip() or "codex"
            return CodexProvider(
                codex_bin=codex_bin,
                model=model,
                log_dir=log_dir,
                registry=registry,
                approve_command=approve_command,
                auth_refresh_interval=codex_cfg.auth_refresh_interval_seconds,
            )
        case _:
            # Default: treat any unknown api value as an OpenAI-compatible endpoint.
            # Covers "openai-completions", "openai-responses", and any Chat Completions API.
            return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model, log_dir=log_dir)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "LMStudioProvider",
    "CodexProvider",
    "create_provider",
]
