"""LM Studio provider — an alias for OpenAICompatibleProvider.

LM Studio is a desktop app that hosts local LLM models and exposes them via
an OpenAI-compatible Chat Completions API at ``http://127.0.0.1:1234/v1``.

Because LM Studio's API is identical to OpenAI's, no new code is needed —
:class:`LMStudioProvider` is just a different name for the same class.

Why have a separate file?
--------------------------
The config system uses the ``"api"`` field (e.g. ``"lmstudio"``) to select a
provider class. Having a named alias here lets the factory (``__init__.py``)
match it cleanly and provides a hook for future LM Studio–specific behavior
(e.g. health-check endpoints, model listing) without breaking the interface.

Talks to
--------
- ``openai_compatible.py`` — where the actual implementation lives.
- ``providers/__init__.py`` — imports this alias for the factory.
"""

from .openai_compatible import OpenAICompatibleProvider

# LMStudioProvider is intentionally identical to OpenAICompatibleProvider.
# "alias" pattern: same class, different name for config-driven selection.
LMStudioProvider = OpenAICompatibleProvider
