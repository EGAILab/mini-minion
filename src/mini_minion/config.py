"""Load and validate ``config.json`` + ``.env`` at import time.

This module is the foundation the entire codebase builds on. Every other module
that needs configuration (which provider to use, which model, where to store
data) imports from here.

Why does this run at import time?
----------------------------------
Python executes module-level code when a module is first imported. All the code
below the function definitions — the ``open()``, ``_validate()``, and resolver
calls — runs the moment any other module writes ``from mini_minion.config import
agents``. This means:

- Config is validated **once**, before the REPL starts.
- If ``config.json`` is missing or broken, ``ConfigError`` is raised at import
  time and the program exits with a clear error message — *before* ``main()``
  even begins.
- The resulting dataclasses (``agents``, ``streaming``, ``compaction``,
  ``workspace``) are module-level globals that any module can import freely,
  knowing they are already validated and immutable.

Why are the dataclasses frozen?
---------------------------------
``@dataclass(frozen=True)`` makes the instances immutable — you cannot write
``config.agents["main"].model = something`` after construction. This is
intentional: config should be set once at startup and never mutated, so freezing
provides a compile-time guarantee rather than relying on discipline.

Module layout
-------------
1. **Validation types** — :class:`ConfigIssue` and :class:`ConfigError` are
   defined first so they can be used in error handling at load time.
2. **``_validate()``** — scans the raw dict for every problem, returns them all
   at once so the user can fix everything in one edit.
3. **Frozen dataclasses** — :class:`ProviderConfig`, :class:`ModelConfig`,
   :class:`AgentModelConfig`, :class:`CompactionConfig`, :class:`StreamingConfig`
   are the strongly-typed, immutable output of parsing.
4. **Private resolvers** — ``_resolve_provider()``, ``_resolve_all()``, etc.
   convert the raw dict into the dataclasses above.
5. **Module-level execution block** — loads the file, validates it, raises if
   invalid, then calls the resolvers and assigns the public exports.

Public exports
--------------
- ``agents``     — ``dict[str, AgentModelConfig]``: provider + model per agent.
- ``workspace``  — ``Path``: root folder for sessions, memory, and metadata.
- ``streaming``  — :class:`StreamingConfig`: whether to stream tokens.
- ``compaction`` — :class:`CompactionConfig`: token reservation for compaction.

API keys and ``.env``
----------------------
API keys are **never** read from ``config.json`` (the validator reports an error
if it finds one there). Instead, ``load_dotenv()`` reads ``.env`` and injects
keys into ``os.environ`` at module load time. ``_resolve_provider()`` then reads
them with ``os.environ.get("{PROVIDER_NAME_UPPER}_API_KEY")``. This keeps secrets
out of version control.

Raises
------
:class:`ConfigError`: If ``config.json`` is missing, contains invalid JSON, or
    fails any validation check. All issues in a single run are collected and
    raised together.
"""

import difflib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root: this file lives at src/mini_minion/config.py,
# so three .parent calls walk up to the repo root directory.
_ROOT = Path(__file__).resolve().parent.parent.parent

# Load the .env file so API keys land in os.environ before we read them below.
load_dotenv(_ROOT / ".env")


# ---------------------------------------------------------------------------
# Validation types — defined before the file-open block so they can be used
# in error handling at load time.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigIssue:
    """A single validation problem found in config.json.

    Attributes:
        path (str): Dot-separated JSON path to the bad value,
            e.g. ``"agents.main.model"`` or ``"streaming.chat_mode"``.
        message (str): Human-readable description of what is wrong.
    """
    path: str
    message: str


class ConfigError(Exception):
    """Raised when config.json is missing, malformed, or fails validation.

    All issues found during a single validation pass are collected and reported
    together, so the user can fix everything in one edit rather than discovering
    problems one at a time.

    Attributes:
        issues (list[ConfigIssue]): Every problem found during validation.
    """

    def __init__(self, issues: list[ConfigIssue]) -> None:
        self.issues = issues
        lines = "\n".join(f"  {i.path}: {i.message}" for i in issues)
        super().__init__(f"Invalid config.json:\n{lines}")


# ---------------------------------------------------------------------------
# Validation — runs before resolvers build dataclasses.
# ---------------------------------------------------------------------------

def _validate(raw: dict) -> list[ConfigIssue]:
    """Validate the raw config dict and return all issues found.

    Continues past individual failures where possible so all issues are
    reported together. Dependent checks (e.g. model exists under a provider)
    are skipped when their parent check already failed.

    Returns:
        list[ConfigIssue]: Empty list if the config is valid.
    """
    issues: list[ConfigIssue] = []

    # Guard: root must be a JSON object, not a list or primitive.
    if not isinstance(raw, dict):
        return [ConfigIssue("config.json", "Expected a JSON object at root.")]

    # --- models.providers ---
    # Guard models section before chaining .get("providers") — if "models"
    # is a string or list, the chained call would raise AttributeError.
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        issues.append(ConfigIssue("models", "Expected an object."))
        models_raw = {}

    providers_raw: dict = models_raw.get("providers", {})
    if not isinstance(providers_raw, dict):
        issues.append(ConfigIssue("models.providers", "Expected an object."))
        providers_raw = {}

    # API types that work without an API key (no env var warning for these).
    _LOCAL_APIS = frozenset({"lmstudio"})

    for pname, praw in providers_raw.items():
        ppath = f"models.providers.{pname}"
        if not isinstance(praw, dict):
            issues.append(ConfigIssue(ppath, "Expected an object."))
            continue
        if not isinstance(praw.get("api"), str) or not praw.get("api"):
            issues.append(ConfigIssue(f"{ppath}.api", "Required non-empty string."))
        if isinstance(praw.get("apiKey"), str) and praw.get("apiKey"):
            issues.append(ConfigIssue(
                f"{ppath}.apiKey",
                f"Inline API keys risk being committed to version control. "
                f"Use the `{pname.upper()}_API_KEY` environment variable in `.env` instead.",
            ))
        api_key_in_env = os.environ.get(f"{pname.upper()}_API_KEY", "")
        if not api_key_in_env and praw.get("api") not in _LOCAL_APIS:
            issues.append(ConfigIssue(
                ppath,
                f"Environment variable {pname.upper()}_API_KEY is not set. "
                f"API calls will fail at runtime with an authentication error.",
            ))
        models_list = praw.get("models", [])
        if not isinstance(models_list, list):
            issues.append(ConfigIssue(f"{ppath}.models", "Expected a list."))
        else:
            for i, mraw in enumerate(models_list):
                mpath = f"{ppath}.models[{i}]"
                if not isinstance(mraw, dict):
                    issues.append(ConfigIssue(mpath, "Expected an object."))
                    continue
                if not isinstance(mraw.get("id"), str) or not mraw.get("id"):
                    issues.append(ConfigIssue(f"{mpath}.id", "Required non-empty string."))
                for field in ("contextWindow", "maxOutputTokens"):
                    val = mraw.get(field)
                    if val is None:
                        issues.append(ConfigIssue(f"{mpath}.{field}", "Required field missing."))
                    elif not isinstance(val, int) or val <= 0:
                        issues.append(ConfigIssue(f"{mpath}.{field}", f"Expected positive integer, got {val!r}."))

    # --- agents ---
    agents_raw: dict = raw.get("agents", {})
    if not isinstance(agents_raw, dict) or not agents_raw:
        issues.append(ConfigIssue("agents", "Expected a non-empty object."))
        agents_raw = {}

    seen_prefixes: dict[str, str] = {}
    for agent_id, agent_raw in agents_raw.items():
        apath = f"agents.{agent_id}"
        if not isinstance(agent_raw, dict):
            issues.append(ConfigIssue(apath, "Expected an object."))
            continue

        # Validate the "provider/model-id" reference.
        model_str = agent_raw.get("model")
        if not model_str:
            issues.append(ConfigIssue(f"{apath}.model", "Required field missing."))
        elif not isinstance(model_str, str):
            issues.append(ConfigIssue(f"{apath}.model", f"Expected string, got {type(model_str).__name__}."))
        elif "/" not in model_str:
            issues.append(ConfigIssue(f"{apath}.model", f'Expected "provider/model-id" format, got {model_str!r}.'))
        else:
            provider_name, model_id = model_str.split("/", 1)
            if not provider_name or not model_id:
                issues.append(ConfigIssue(f"{apath}.model", f"Provider and model id must both be non-empty, got {model_str!r}."))
            elif provider_name not in providers_raw:
                close = difflib.get_close_matches(provider_name, providers_raw.keys(), n=1)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                issues.append(ConfigIssue(f"{apath}.model", f"Unknown provider {provider_name!r}.{hint}"))
            else:
                # Only validate model id when the provider itself is known.
                available = [
                    m["id"] for m in providers_raw[provider_name].get("models", [])
                    if isinstance(m, dict) and "id" in m
                ]
                if model_id not in available:
                    close = difflib.get_close_matches(model_id, available, n=1)
                    hint = f" Did you mean {close[0]!r}?" if close else (f" Available: {available}." if available else "")
                    issues.append(ConfigIssue(f"{apath}.model", f"Unknown model {model_id!r} under provider {provider_name!r}.{hint}"))

        # Validate route_prefix format and uniqueness.
        prefix = agent_raw.get("route_prefix") or None
        if prefix is not None:
            if not isinstance(prefix, str):
                issues.append(ConfigIssue(f"{apath}.route_prefix", f"Expected string or null, got {type(prefix).__name__}."))
            elif not prefix.startswith("/"):
                issues.append(ConfigIssue(f"{apath}.route_prefix", f"Must start with '/', got {prefix!r}."))
            elif prefix in seen_prefixes:
                issues.append(ConfigIssue(f"{apath}.route_prefix", f"Duplicate prefix {prefix!r} already used by '{seen_prefixes[prefix]}'."))
            else:
                seen_prefixes[prefix] = agent_id

    # Require at least one agent with no route_prefix to serve as the router fallback.
    # If every agent has a prefix, router.py would silently fall back to the hardcoded
    # string "main", which may not exist.
    if agents_raw and not any(
        isinstance(a, dict) and not a.get("route_prefix")
        for a in agents_raw.values()
    ):
        issues.append(ConfigIssue(
            "agents",
            "At least one agent must have no 'route_prefix' to serve as the default fallback.",
        ))

    # --- streaming ---
    streaming_raw = raw.get("streaming", {})
    if isinstance(streaming_raw, dict):
        for key in ("chat_mode", "task_mode"):
            val = streaming_raw.get(key)
            if val is not None and not isinstance(val, bool):
                issues.append(ConfigIssue(
                    f"streaming.{key}",
                    f"Expected boolean (true/false), got {val!r}. JSON strings are not booleans.",
                ))

    # --- compaction ---
    compaction_raw = raw.get("compaction", {})
    if isinstance(compaction_raw, dict):
        pt = compaction_raw.get("preserve_tokens")
        if pt is not None and (not isinstance(pt, int) or pt <= 0):
            issues.append(ConfigIssue("compaction.preserve_tokens", f"Expected positive integer, got {pt!r}."))

    return issues


# ---------------------------------------------------------------------------
# Frozen dataclasses — "frozen=True" means the objects are immutable once
# created (like named constants). This prevents accidental mutation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    """Settings for a single LLM provider backend (e.g. LM Studio, Anthropic).

    Attributes:
        name (str): Provider identifier, e.g. ``"lmstudio"`` or ``"anthropic"``.
        base_url (str): The HTTP endpoint the provider SDK talks to, e.g.
            ``"http://127.0.0.1:1234/v1"``. Empty string for Anthropic (uses
            the default SDK endpoint).
        api_key (str): Authentication token. Resolved from .env; never from
            config.json directly.
        api (str): Which adapter class to use, e.g. ``"openai-completions"``,
            ``"lmstudio"``, or ``"anthropic"``.
    """
    name: str
    base_url: str
    api_key: str
    api: str


@dataclass(frozen=True)
class ModelConfig:
    """Settings for one specific model offered by a provider.

    Attributes:
        id (str): The model identifier string sent in API requests, e.g.
            ``"qwen-qwen3.5-9b"`` or ``"claude-3-5-sonnet-20241022"``.
        context_window (int): The model's total token capacity. Used to
            determine when conversation history needs compaction. Set this
            to the actual context window advertised by the model (e.g. 8192,
            128000). Comes from ``contextWindow`` in config.json.
        max_output_tokens (int): The maximum tokens the model may generate
            in a single response. Sent directly to the API as ``max_tokens``.
            Comes from ``maxOutputTokens`` in config.json.
    """
    id: str
    context_window: int
    max_output_tokens: int


@dataclass(frozen=True)
class AgentModelConfig:
    """Complete configuration for one agent: which provider, model, and routing prefix.

    This is what ``agents["main"]`` returns — everything ``minion.py`` needs
    to spin up a provider and start chatting, plus the optional routing prefix
    used by ``router.py`` to direct user messages to this agent.

    Attributes:
        provider (ProviderConfig): Connection details for the LLM backend.
        model (ModelConfig): Which model to use and its token limit.
        route_prefix (str | None): The command prefix that routes messages to this
            agent (e.g. ``"/research"``). ``None`` means this agent is the default
            fallback — it handles messages that match no other prefix.
    """
    provider: ProviderConfig
    model: ModelConfig
    route_prefix: str | None = None  # None → default/fallback agent; field with default must come last


@dataclass(frozen=True)
class CompactionConfig:
    """Controls when and how conversation history is compacted.

    The context window budget is derived per-agent from ``ModelConfig.context_window``
    rather than stored here.

    Attributes:
        preserve_tokens (int | None): Tokens to reserve for the model's response
            and protocol overhead.  ``None`` (the default when ``config.json``
            omits the field) means "auto-compute from the model's
            ``maxOutputTokens``" — ``minion.py`` substitutes
            ``model.max_output_tokens`` at Compactor construction time.
            When set explicitly it is used as-is (clamped to
            ``[_MIN_PRESERVE, context_window // 2]`` inside :class:`Compactor`).
    """
    preserve_tokens: int | None = None


@dataclass(frozen=True)
class StreamingConfig:
    """Controls whether token-by-token streaming is active per execution mode.

    Streaming prints each word as the model generates it rather than waiting
    for the full response. This makes the experience feel more responsive in
    interactive use, but adds complexity so it can be disabled for programmatic
    or task-based usage where the full response is needed at once.

    These settings map to how the program is invoked:
    - ``chat_mode``  — the interactive REPL (``uv run mini-minion``).
    - ``task_mode``  — future non-interactive / programmatic invocation.

    Configured in ``config.json`` under the ``"streaming"`` key:

    .. code-block:: json

        "streaming": {
            "chat_mode": true,
            "task_mode": false
        }

    Attributes:
        chat_mode (bool): If ``True``, stream tokens to the terminal during the
            interactive chat loop. Defaults to ``False`` if omitted from config.
        task_mode (bool): If ``True``, stream tokens during programmatic / task
            execution. Defaults to ``False`` if omitted from config.
    """
    chat_mode: bool
    task_mode: bool


# ---------------------------------------------------------------------------
# Private helper functions — only used at module startup, not exported.
# ---------------------------------------------------------------------------

def _resolve_provider(provider_name: str, model_id: str) -> AgentModelConfig:
    """Look up one provider + model combination and build an AgentModelConfig.

    Validation has already confirmed both exist, so lookups here cannot fail.

    Args:
        provider_name (str): Key in ``config.json`` → ``models.providers``.
        model_id (str): The ``id`` field of one entry in that provider's models list.

    Returns:
        AgentModelConfig: A fully populated, immutable config object.
    """
    provider_raw = _raw["models"]["providers"][provider_name]
    model_raw = next(m for m in provider_raw["models"] if m["id"] == model_id)

    # Read the API key from the environment. Set {PROVIDER_NAME_UPPERCASE}_API_KEY in .env.
    # Falls back to empty string, which will cause auth errors at runtime — intentional,
    # so the failure is loud rather than silently passing an empty credential.
    api_key = os.environ.get(f"{provider_name.upper()}_API_KEY", "")

    return AgentModelConfig(
        provider=ProviderConfig(
            name=provider_name,
            base_url=provider_raw.get("baseUrl", ""),
            api_key=api_key,
            api=provider_raw["api"],
        ),
        model=ModelConfig(
            id=model_id,
            context_window=model_raw.get("contextWindow", 32_768),
            max_output_tokens=model_raw.get("maxOutputTokens", 4_096),
        ),
    )


def _resolve_all() -> dict[str, AgentModelConfig]:
    """Build the full agent config dict by iterating over ``config.json`` agents.

    Each agent entry looks like::

        {"model": "lmstudio/qwen-qwen3.5-9b", "route_prefix": "/research"}

    The model string is split on the first "/" to get provider name and model id.
    The optional ``route_prefix`` is stored so ``router.py`` can build its routing
    table from config rather than hard-coding prefixes in Python source.

    Returns:
        dict[str, AgentModelConfig]: Maps agent IDs (``"main"``, ``"researcher"``)
            to their resolved provider + model + routing configs.
    """
    result: dict[str, AgentModelConfig] = {}
    for agent_id, agent_raw in _raw.get("agents", {}).items():
        # Split "lmstudio/qwen-qwen3.5-9b" → provider="lmstudio", model="qwen-qwen3.5-9b"
        provider_name, model_id = agent_raw["model"].split("/", 1)
        base = _resolve_provider(provider_name, model_id)
        # route_prefix is optional; absent or empty string both mean "default agent".
        route_prefix = agent_raw.get("route_prefix") or None
        result[agent_id] = AgentModelConfig(
            provider=base.provider,
            model=base.model,
            route_prefix=route_prefix,
        )
    return result


def _resolve_compaction() -> CompactionConfig:
    """Read the ``"compaction"`` section from config.json and build a CompactionConfig.

    The context window is no longer stored here — it comes from each agent's
    ``ModelConfig.context_window``. Only ``preserve_tokens`` is global.

    Returns:
        CompactionConfig: Immutable compaction settings.
    """
    raw = _raw.get("compaction", {})
    pt = raw.get("preserve_tokens")
    return CompactionConfig(
        # None → caller (minion.py) substitutes model.max_output_tokens.
        preserve_tokens=int(pt) if pt is not None else None,
    )


def _resolve_streaming() -> StreamingConfig:
    """Read the ``"streaming"`` section from config.json and build a StreamingConfig.

    Both keys default to ``False`` if absent. Validation has already confirmed
    that any present values are booleans, so no coercion is needed here.

    Returns:
        StreamingConfig: Immutable streaming settings for all execution modes.
    """
    raw = _raw.get("streaming", {})
    return StreamingConfig(
        chat_mode=raw.get("chat_mode", False),
        task_mode=raw.get("task_mode", False),
    )


# ---------------------------------------------------------------------------
# Load, validate, then resolve — in that order.
# ---------------------------------------------------------------------------

_config_path = _ROOT / "config.json"
try:
    with open(_config_path, encoding="utf-8") as _f:
        _raw = json.load(_f)
except FileNotFoundError:
    raise ConfigError([ConfigIssue("config.json", f"File not found at {_config_path}.")]) from None
except json.JSONDecodeError as exc:
    raise ConfigError([ConfigIssue("config.json", f"Invalid JSON: {exc}")]) from exc

_issues = _validate(_raw)
if _issues:
    raise ConfigError(_issues)


# ---------------------------------------------------------------------------
# Module-level exports — these are the things other modules import.
# ---------------------------------------------------------------------------

# agents: dict mapping agent ID → provider+model config.
# Built once at import time; safe to read from anywhere without re-parsing.
agents: dict[str, AgentModelConfig] = _resolve_all()

# workspace: the root folder for all persisted data (history, memory notes).
# expanduser() converts "~/.mini-minion" to the actual home-directory path.
workspace: Path = Path(_raw.get("workspace", {}).get("path", "~/.mini-minion")).expanduser()

# streaming: controls whether token-by-token streaming is active per mode.
# Read from the "streaming" section of config.json; defaults to all-False if absent.
streaming: StreamingConfig = _resolve_streaming()

# compaction: controls when conversation history is summarised to free context window.
# Read from the "compaction" section of config.json; defaults to 4 000 if absent.
compaction: CompactionConfig = _resolve_compaction()
