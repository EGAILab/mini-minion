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
1. Create ``src/mini_minion/providers/myprovider.py`` with a class that has a
   ``chat()`` method matching :class:`LLMProvider`.
2. Import it here and add a ``case "my-api":`` branch in the ``match`` below.
3. Add the provider config to ``config.json``.

Talks to
--------
- ``minion.py`` calls :func:`create_provider` once per agent at startup.
- ``runner.py`` uses the returned provider to call the LLM.
"""

from .anthropic import AnthropicProvider
from .base import LLMProvider, LLMResponse, TokenUsage, ToolCall
from .lmstudio import LMStudioProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(api: str, base_url: str, api_key: str, model: str) -> LLMProvider:
    """Instantiate the correct provider class for the given API type.

    This is the only place in the codebase that knows which ``api`` string
    maps to which implementation class. All other code uses the returned
    object via the :class:`LLMProvider` protocol.

    Args:
        api (str): The adapter type string from ``config.json``, e.g.
            ``"openai-completions"``, ``"lmstudio"``, or ``"anthropic"``.
        base_url (str): HTTP endpoint for the provider. Unused for Anthropic
            (the SDK uses its own default endpoint).
        api_key (str): Authentication token loaded from ``.env``.
        model (str): Model identifier string for API requests.

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
            return AnthropicProvider(api_key=api_key, model=model)
        case "lmstudio":
            # LMStudioProvider is an alias for OpenAICompatibleProvider; same behavior.
            return LMStudioProvider(base_url=base_url, api_key=api_key, model=model)
        case _:
            # Default: treat any unknown api value as an OpenAI-compatible endpoint.
            # Covers "openai-completions", "openai-responses", "openai", and others.
            return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "LMStudioProvider",
    "create_provider",
]
