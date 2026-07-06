"""Load and validate ``config.json`` + ``.env`` at import time.

This module is the foundation the entire codebase builds on. Every other module
that needs configuration (which provider to use, which model, where to store
data) imports from here.

Why does this run at import time?
----------------------------------
Python executes module-level code when a module is first imported. All the code
below the function definitions — the ``open()``, ``_validate()``, and resolver
calls — runs the moment any other module writes ``from minion_assist.config import
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
import re as _re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# The user's minion-assist home directory.  Kept as a module-level name so
# other modules can import it instead of re-computing the path themselves.
# MINION_ASSIST_HOME lets the user redirect everything to a custom location
# (e.g. different profiles, CI environments, or network drives).
_HOME = Path(os.environ.get("MINION_ASSIST_HOME", "~/.minion-assist")).expanduser()

# Load API keys from .env files before reading the config (providers need keys).
# User-home .env is loaded first, then the project-local one — project .env
# values override the user-level defaults so per-project keys work transparently.
load_dotenv(_HOME / ".env", override=False)
load_dotenv(Path.cwd() / ".env", override=False)


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
        warning (bool): When ``True``, this issue is printed to stderr but
            does not prevent startup. Use for conditions that will fail at
            runtime (e.g. missing API key) but shouldn't block the REPL from
            opening.  Defaults to ``False`` (fatal).
    """
    path: str
    message: str
    warning: bool = False


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
    # codex: uses OAuth tokens from codex-auth.json, not PROVIDER_API_KEY.
    _LOCAL_APIS = frozenset({"lmstudio", "codex"})

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
            # Warning rather than error: a missing key doesn't prevent startup.
            # The first real API call will fail with a clear authentication error.
            issues.append(ConfigIssue(
                ppath,
                f"Environment variable {pname.upper()}_API_KEY is not set. "
                f"API calls will fail at runtime with an authentication error.",
                warning=True,
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
                # Validate inputModalities when present — omitting it is fine (defaults to ["text"]).
                _VALID_MODALITIES = frozenset({"text", "image", "audio", "video"})
                modalities = mraw.get("inputModalities")
                if modalities is not None:
                    if not isinstance(modalities, list):
                        issues.append(ConfigIssue(f"{mpath}.inputModalities", "Expected a list of strings."))
                    else:
                        for mod in modalities:
                            if mod not in _VALID_MODALITIES:
                                issues.append(ConfigIssue(
                                    f"{mpath}.inputModalities",
                                    f"Unknown modality {mod!r}. Valid: {sorted(_VALID_MODALITIES)}.",
                                ))

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

    # --- memory ---
    memory_raw = raw.get("memory", {})
    if memory_raw and not isinstance(memory_raw, dict):
        issues.append(ConfigIssue("memory", "Expected an object."))
    elif isinstance(memory_raw, dict):
        ee = memory_raw.get("enable_extraction")
        if ee is not None and not isinstance(ee, bool):
            issues.append(ConfigIssue(
                "memory.enable_extraction",
                f"Expected boolean (true/false), got {ee!r}.",
            ))

    # --- multi_agent ---
    ma_raw = raw.get("multi_agent", {})
    if ma_raw and not isinstance(ma_raw, dict):
        issues.append(ConfigIssue("multi_agent", "Expected an object."))
    elif isinstance(ma_raw, dict):
        for _ma_key in ("max_spawn_depth", "max_children_per_agent", "default_subagent_timeout_seconds"):
            _ma_val = ma_raw.get(_ma_key)
            if _ma_val is not None and (not isinstance(_ma_val, int) or _ma_val <= 0):
                issues.append(ConfigIssue(
                    f"multi_agent.{_ma_key}",
                    f"Expected positive integer, got {_ma_val!r}.",
                ))

    # --- bootstrap ---
    bootstrap_raw = raw.get("bootstrap", {})
    if bootstrap_raw and not isinstance(bootstrap_raw, dict):
        issues.append(ConfigIssue("bootstrap", "Expected an object."))
    elif isinstance(bootstrap_raw, dict):
        _be = bootstrap_raw.get("enabled")
        if _be is not None and not isinstance(_be, bool):
            issues.append(ConfigIssue(
                "bootstrap.enabled",
                f"Expected boolean (true/false), got {_be!r}.",
            ))
        _bp = bootstrap_raw.get("path")
        if _bp is not None and not isinstance(_bp, str):
            issues.append(ConfigIssue(
                "bootstrap.path",
                f"Expected string or null, got {type(_bp).__name__}.",
            ))
        for _bfield in ("max_chars", "total_max_chars"):
            _bval = bootstrap_raw.get(_bfield)
            if _bval is not None and (not isinstance(_bval, int) or _bval <= 0):
                issues.append(ConfigIssue(
                    f"bootstrap.{_bfield}",
                    f"Expected positive integer, got {_bval!r}.",
                ))
        _btw = bootstrap_raw.get("truncation_warning")
        if _btw is not None and _btw not in ("off", "once", "always"):
            issues.append(ConfigIssue(
                "bootstrap.truncation_warning",
                f"Expected 'off', 'once', or 'always', got {_btw!r}.",
            ))

    # --- extra_plugin_manifests ---
    # Optional list of additional plugins.json paths to load, on top of the
    # two fixed locations (~/.minion-assist/plugins.json + .minion-assist/plugins.json).
    epm = raw.get("extra_plugin_manifests")
    if epm is not None:
        if not isinstance(epm, list):
            issues.append(ConfigIssue("extra_plugin_manifests", "Expected a list of file paths."))
        else:
            for i, p in enumerate(epm):
                if not isinstance(p, str):
                    issues.append(ConfigIssue(f"extra_plugin_manifests[{i}]", f"Expected string path, got {type(p).__name__}."))


    # --- logging ---
    logging_raw = raw.get("logging", {})
    if logging_raw and not isinstance(logging_raw, dict):
        issues.append(ConfigIssue("logging", "Expected an object."))
    elif isinstance(logging_raw, dict):
        _llmr = logging_raw.get("llm_requests")
        if _llmr is not None and not isinstance(_llmr, bool):
            issues.append(ConfigIssue(
                "logging.llm_requests",
                f"Expected boolean (true/false), got {_llmr!r}.",
            ))

    # --- channels ---
    channels_raw = raw.get("channels", {})
    if channels_raw:
        if not isinstance(channels_raw, dict):
            issues.append(ConfigIssue("channels", "Expected an object."))
        elif "matrix" in channels_raw:
            m = channels_raw["matrix"]
            if not isinstance(m, dict):
                issues.append(ConfigIssue("channels.matrix", "Expected an object."))
            else:
                if not m.get("homeserver"):
                    issues.append(ConfigIssue("channels.matrix.homeserver", "Required non-empty string."))
                if not m.get("userId"):
                    issues.append(ConfigIssue("channels.matrix.userId", "Required non-empty string."))
                if not m.get("accessToken") and not m.get("password"):
                    issues.append(ConfigIssue(
                        "channels.matrix",
                        "At least one of 'accessToken' or 'password' is required.",
                    ))

    # --- mcp ---
    mcp_raw = raw.get("mcp", {})
    if mcp_raw and not isinstance(mcp_raw, dict):
        issues.append(ConfigIssue("mcp", "Expected an object."))
    elif isinstance(mcp_raw, dict):
        servers_raw = mcp_raw.get("servers", {})
        if not isinstance(servers_raw, dict):
            issues.append(ConfigIssue("mcp.servers", "Expected an object mapping server names to configs."))
        else:
            for sname, sraw in servers_raw.items():
                spath = f"mcp.servers.{sname}"
                if not _SERVER_NAME_RE.match(sname):
                    issues.append(ConfigIssue(spath, f"Server name must match [A-Za-z0-9][A-Za-z0-9_-]{{0,63}}, got {sname!r}."))
                if not isinstance(sraw, dict):
                    issues.append(ConfigIssue(spath, "Expected an object."))
                    continue
                # Normalize and validate transport
                transport = sraw.get("transport", "")
                transport = _TRANSPORT_ALIASES.get(transport, transport)
                if transport not in _VALID_TRANSPORTS:
                    issues.append(ConfigIssue(f"{spath}.transport", f"Expected one of {sorted(_VALID_TRANSPORTS)}, got {transport!r}."))
                elif transport == "stdio":
                    if not sraw.get("command"):
                        issues.append(ConfigIssue(f"{spath}.command", "Required non-empty string for stdio transport."))
                elif transport in ("sse", "streamableHttp"):
                    if not sraw.get("url"):
                        issues.append(ConfigIssue(f"{spath}.url", f"Required non-empty string for {transport} transport."))
                # Validate env is dict[str, str]
                env_raw = sraw.get("env", {})
                if not isinstance(env_raw, dict):
                    issues.append(ConfigIssue(f"{spath}.env", "Expected an object mapping strings to strings."))
                else:
                    for k, v in env_raw.items():
                        if not isinstance(v, str):
                            issues.append(ConfigIssue(f"{spath}.env.{k}", f"Expected string value, got {type(v).__name__}."))
                        if k.upper() in _DANGEROUS_ENV_KEYS:
                            issues.append(ConfigIssue(f"{spath}.env.{k}", f"Env key {k!r} is dangerous and not allowed."))
                # Validate headers is dict[str, str]
                headers_raw = sraw.get("headers", {})
                if not isinstance(headers_raw, dict):
                    issues.append(ConfigIssue(f"{spath}.headers", "Expected an object mapping strings to strings."))
                else:
                    for k, v in headers_raw.items():
                        if not isinstance(v, str):
                            issues.append(ConfigIssue(f"{spath}.headers.{k}", f"Expected string value, got {type(v).__name__}."))

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
        input_modalities (tuple[str, ...]): Media types this model can accept
            as input.  ``"text"`` is always included.  Vision-capable models
            also list ``"image"``.  Comes from ``inputModalities`` in
            config.json; defaults to ``("text",)`` when omitted.
            Valid values: "text", "image", "audio", "video".
    """
    id: str
    context_window: int
    max_output_tokens: int
    # Fields with defaults must come AFTER fields without defaults (Python rule).
    # input_modalities defaults to text-only so existing configs work unchanged.
    input_modalities: tuple[str, ...] = ("text",)


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
class McpServerConfig:
    """One MCP server entry from the config.json 'mcp.servers' section.

    Fields map 1:1 to the JSON config shape validated in _validate_mcp().
    All fields are immutable (frozen=True) because config is read-only at runtime.
    """
    name: str                              # server key e.g. "context7"
    transport: str                         # "stdio", "sse", or "streamableHttp"
    # stdio fields
    command: str = ""                      # e.g. "npx"
    args: tuple[str, ...] = ()            # e.g. ("-y", "@upstash/context7-mcp@latest")
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    # network fields
    url: str = ""                          # for sse/streamableHttp
    headers: dict[str, str] = field(default_factory=dict)
    # shared
    enabled_tools: tuple[str, ...] = ("*",)  # allowlist; "*" = all
    tool_timeout: int = 30                 # seconds, clamped to [5, 600]


@dataclass(frozen=True)
class McpConfig:
    """All MCP servers configured for this minion-assist instance."""
    servers: tuple[McpServerConfig, ...] = ()


@dataclass(frozen=True)
class ChannelsConfig:
    """Channel integrations configured under ``channels`` in config.json.

    Attributes:
        matrix: Parsed :class:`~minion_assist.matrix.config.MatrixConfig` when
            ``channels.matrix`` is present in config.json; ``None`` otherwise.
    """
    matrix: object = None  # MatrixConfig | None; typed as object to avoid circular import


@dataclass(frozen=True)
class MemoryConfig:
    """Controls optional memory subsystem behaviour.

    Attributes:
        enable_extraction (bool): When ``True`` (default), a background daemon
            thread runs after each successful turn to extract 0–3 short facts
            from the last exchange and append them to ``_auto_extracted.md``.
            Each extraction triggers one extra provider ``chat()`` call, which
            adds API cost.  Set to ``false`` in ``config.json`` under
            ``"memory": {"enable_extraction": false}`` to disable.
    """
    enable_extraction: bool = True


@dataclass(frozen=True)
class MultiAgentConfig:
    """Controls multi-agent subagent spawning limits and defaults.

    Read from the optional ``"multi_agent"`` section in ``config.json``.
    All fields have sensible defaults so the section can be omitted entirely.

    Attributes:
        max_spawn_depth (int): Maximum parent→child nesting depth.  A root
            agent has depth 0; each spawned subagent adds 1.  Default 4.
        max_children_per_agent (int): Maximum number of child sessions a single
            parent session may spawn in its lifetime.  Default 5.
        default_subagent_timeout_seconds (int): Seconds the parent waits for a
            subagent to finish before returning a timeout error.  Default 120.
    """
    max_spawn_depth: int = 4
    max_children_per_agent: int = 5
    default_subagent_timeout_seconds: int = 120


@dataclass(frozen=True)
class BootstrapConfig:
    """Controls the workspace bootstrap prompt injection layer.

    Bootstrap files (``AGENTS.md``, ``SOUL.md``, ``TOOLS.md``, etc.) are
    discovered under ``path`` on every agent turn, budget-clamped, and
    rendered into a ``# Project Context`` block injected into the system
    prompt after the static agent soul.  Editing any bootstrap file takes
    effect on the next turn without restarting the process.

    Configured in ``config.json`` under the ``"bootstrap"`` key::

        "bootstrap": {
            "enabled": true,
            "path": null,
            "max_chars": 20000,
            "total_max_chars": 60000,
            "truncation_warning": "always"
        }

    Attributes:
        enabled (bool): When ``True`` (default), bootstrap injection is active.
            Set to ``false`` to disable entirely without removing config.
        path (str | None): Filesystem path to search for bootstrap files.
            ``None`` (the default) resolves to ``Path.cwd()`` at call time,
            picking up the working directory of the process.
        max_chars (int): Maximum characters to inject from any single bootstrap
            file.  Content exceeding this limit is truncated with a head+tail
            split.  Default 20 000 (≈ 5 000 tokens).
        total_max_chars (int): Maximum total characters across all bootstrap
            files combined.  Processing stops when this limit is reached.
            Default 60 000 (≈ 15 000 tokens).
        truncation_warning (str): Controls when a truncation warning is injected
            into the prompt.  ``"always"`` (default) warns on every turn that
            has truncated content; ``"once"`` warns at most once per process
            lifetime; ``"off"`` suppresses the warning entirely.
    """
    enabled: bool = True
    path: str | None = None
    max_chars: int = 20_000
    total_max_chars: int = 60_000
    truncation_warning: str = "always"


@dataclass(frozen=True)
class HeartbeatConfig:
    """Controls the periodic background heartbeat scheduler.

    The heartbeat fires an agent turn on a timer so the agent can check emails,
    calendars, or other proactive tasks without a human prompting it.

    Configured in ``config.json`` under the ``"heartbeat"`` key::

        "heartbeat": {
            "enabled": true,
            "interval_seconds": 1800,
            "prompt": "Read HEARTBEAT.md if it exists. If nothing needs attention, reply HEARTBEAT_OK.",
            "agent_id": "main",
            "notification_room_id": "!abc:example.org"
        }

    Attributes:
        enabled (bool): When ``True`` (default ``False``), the scheduler starts
            automatically at process startup.
        interval_seconds (int): Seconds between heartbeat turns.  Default 1800
            (30 minutes).  Clamped to [60, 86400] at runtime.
        prompt (str): The message sent to the agent each heartbeat turn.
        agent_id (str): Which agent receives the heartbeat.  Default ``"main"``.
        notification_room_id (str | None): Matrix room ID to post proactive
            notifications when the agent calls ``heartbeat_respond``.  ``None``
            means print to the terminal instead.
    """
    enabled: bool = False
    interval_seconds: int = 1800
    prompt: str = (
        "Read HEARTBEAT.md if it exists (workspace context). "
        "Follow it strictly. Do not infer or repeat old tasks from prior chats. "
        "If nothing needs attention, reply HEARTBEAT_OK."
    )
    agent_id: str = "main"
    notification_room_id: str | None = None


@dataclass(frozen=True)
class DreamingConfig:
    """Controls the nightly dream diary scheduler.

    The dreaming system fires an isolated agent turn each night that reads
    recent daily memory files and writes a poetic diary entry to DREAMS.md
    in the agent's workspace directory.

    Configured in ``config.json`` under the ``"dreaming"`` key::

        "dreaming": {
            "enabled": true,
            "hour": 3,
            "minute": 0,
            "timezone": "Australia/Sydney",
            "lookback_days": 3,
            "agent_id": "main"
        }

    Attributes:
        enabled (bool): When ``True`` (default ``False``), the dreaming
            scheduler starts at process startup.
        hour (int): Wall-clock hour of day to fire (0-23). Default 3 (3am).
        minute (int): Wall-clock minute to fire (0-59). Default 0.
        timezone (str): IANA timezone name for scheduling. Default
            ``"Australia/Sydney"``.  Requires the ``tzdata`` package on Windows.
        lookback_days (int): How many days of daily memory files to read as
            source material.  Default 3.
        agent_id (str): Which agent produces the dream entry. Default ``"main"``.
    """

    enabled: bool = False
    hour: int = 3
    minute: int = 0
    timezone: str = "Australia/Sydney"
    lookback_days: int = 3
    agent_id: str = "main"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls verbose LLM request/response logging.

    When enabled, every request body sent to the LLM and every response received
    is appended to a daily log file at ``~/.minion-assist/logs/YYYY-MM-DD.log``
    in LM Studio's server log format so both files can be correlated by timestamp.

    Configured in ``config.json`` under ``"logging"``:

        "logging": {
            "llm_requests": false
        }

    Attributes:
        llm_requests (bool): When ``True`` (the default), log all LLM traffic.
            Set to ``false`` in ``config.json`` to suppress logging.
    """
    llm_requests: bool = True


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
    - ``chat_mode``  — the interactive REPL (``uv run minion-assist``).
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
# MCP validation constants
# ---------------------------------------------------------------------------

# Valid server name format: starts with alphanumeric, allows alphanumeric/underscore/hyphen.
_SERVER_NAME_RE = _re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')

# Transport aliases allow friendlier spellings in config.json.
# They are normalized to the canonical form before building McpServerConfig.
_TRANSPORT_ALIASES: dict[str, str] = {
    "http": "streamableHttp",
    "streamable-http": "streamableHttp",
}
_VALID_TRANSPORTS = frozenset({"stdio", "sse", "streamableHttp"})

# Env var keys known to affect interpreter or shell startup in dangerous ways.
# Rejecting them prevents MCP server config from being used to inject code
# into Python, Node.js, Ruby, Perl, or the shell itself.
_DANGEROUS_ENV_KEYS = frozenset({
    "NODE_OPTIONS", "PYTHONPATH", "PYTHONSTARTUP", "RUBYOPT",
    "PERL5OPT", "SHELLOPTS", "PS4", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES",
})


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

    # Read inputModalities from config; default to ["text"] when absent so
    # existing configs work unchanged and text-only providers need no changes.
    input_modalities = tuple(model_raw.get("inputModalities", ["text"]))

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
            input_modalities=input_modalities,
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


def _resolve_memory() -> MemoryConfig:
    """Read the ``"memory"`` section from config.json and build a MemoryConfig.

    Returns:
        MemoryConfig: Immutable memory settings; defaults apply when section absent.
    """
    raw = _raw.get("memory", {})
    return MemoryConfig(
        # Default True so existing configs work unchanged.
        enable_extraction=raw.get("enable_extraction", True),
    )


def _resolve_bootstrap() -> BootstrapConfig:
    """Read the ``"bootstrap"`` section from config.json and build a BootstrapConfig.

    All fields have sensible defaults so the section can be omitted entirely
    from ``config.json`` without breaking the bootstrap layer — it will simply
    discover files from ``Path.cwd()`` with the standard budget limits.

    Returns:
        BootstrapConfig: Immutable bootstrap settings; defaults apply when absent.
    """
    raw = _raw.get("bootstrap", {})
    if not isinstance(raw, dict):
        return BootstrapConfig()
    return BootstrapConfig(
        enabled=raw.get("enabled", True),
        path=raw.get("path"),
        max_chars=raw.get("max_chars", 20_000),
        total_max_chars=raw.get("total_max_chars", 60_000),
        truncation_warning=raw.get("truncation_warning", "always"),
    )


def _resolve_multi_agent() -> MultiAgentConfig:
    """Read the ``"multi_agent"`` section from config.json and build a MultiAgentConfig.

    All fields have sensible defaults so the section can be omitted entirely.

    Returns:
        MultiAgentConfig: Immutable spawn-limit settings; defaults apply when absent.
    """
    raw = _raw.get("multi_agent", {})
    if not isinstance(raw, dict):
        return MultiAgentConfig()
    return MultiAgentConfig(
        max_spawn_depth=raw.get("max_spawn_depth", 4),
        max_children_per_agent=raw.get("max_children_per_agent", 5),
        default_subagent_timeout_seconds=raw.get("default_subagent_timeout_seconds", 120),
    )


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


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} references in a string using os.environ.

    Only expands ${VAR} syntax (not $VAR without braces) to avoid accidentally
    expanding single-dollar signs that appear in URLs or commands.
    """
    return _re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _resolve_mcp() -> McpConfig:
    """Read the 'mcp' section from config.json and build an McpConfig.

    Normalizes transport aliases, clamps tool_timeout to [5, 600], and
    expands ${VAR} references in env/headers/url values.

    Returns:
        McpConfig: Immutable MCP configuration; empty servers tuple if section absent.
    """
    mcp_raw = _raw.get("mcp", {})
    if not isinstance(mcp_raw, dict):
        return McpConfig()

    servers_raw = mcp_raw.get("servers", {})
    if not isinstance(servers_raw, dict):
        return McpConfig()

    server_list = []
    for sname, sraw in servers_raw.items():
        if not isinstance(sraw, dict):
            continue

        transport = sraw.get("transport", "stdio")
        # Normalize transport aliases (e.g. "http" → "streamableHttp")
        transport = _TRANSPORT_ALIASES.get(transport, transport)

        # Expand ${VAR} in url and string values
        url = _expand_env_vars(sraw.get("url", ""))

        # Expand env values
        env_raw = sraw.get("env", {})
        env = {k: _expand_env_vars(v) for k, v in env_raw.items() if isinstance(v, str)}

        # Expand header values
        headers_raw = sraw.get("headers", {})
        headers = {k: _expand_env_vars(v) for k, v in headers_raw.items() if isinstance(v, str)}

        # Clamp tool_timeout to [5, 600] seconds
        raw_timeout = sraw.get("tool_timeout", 30)
        tool_timeout = max(5, min(600, int(raw_timeout) if isinstance(raw_timeout, (int, float)) else 30))

        # enabled_tools: default to ["*"] (all tools)
        enabled_raw = sraw.get("enabled_tools", ["*"])
        enabled_tools: tuple[str, ...] = tuple(enabled_raw) if isinstance(enabled_raw, list) else ("*",)

        # Expand ${VAR} in args so config.json can reference environment variables.
        #
        # Why only ${VAR} syntax (not $VAR)?
        #   The _expand_env_vars function only matches ${...} braces, not bare $VAR.
        #   This prevents accidentally expanding things like "$5" in shell scripts
        #   or "$?" in command output. See _expand_env_vars() in this file for details.
        #
        # Practical use cases:
        #   --output-dir ${USERPROFILE}\.minion-assist\playwright-output  (Windows)
        #   --output-dir ${HOME}/.minion-assist/playwright-output         (Linux/Mac)
        #   --storage-state ${HOME}/.minion-assist/playwright-session.json
        #
        # If a variable is NOT set in the environment, ${VAR} is left as-is
        # (not replaced with empty string) so the user gets a visible clue that
        # the expansion failed rather than a cryptic "path not found" error.
        raw_args = sraw.get("args", [])
        args = tuple(
            _expand_env_vars(a) if isinstance(a, str) else str(a)
            for a in raw_args
        )

        server_list.append(McpServerConfig(
            name=sname,
            transport=transport,
            command=sraw.get("command", ""),
            args=args,
            env=env,
            cwd=sraw.get("cwd", ""),
            url=url,
            headers=headers,
            enabled_tools=enabled_tools,
            tool_timeout=tool_timeout,
        ))

    return McpConfig(servers=tuple(server_list))


def _resolve_channels() -> ChannelsConfig:
    """Read the ``"channels"`` section from config.json and build a ChannelsConfig.

    Lazily imports :class:`~minion_assist.matrix.config.MatrixConfig` so that
    ``matrix-nio`` is only required when the Matrix channel is actually configured.

    Returns:
        ChannelsConfig: Config with ``matrix`` set to a ``MatrixConfig`` instance
            when ``channels.matrix`` is present, otherwise ``matrix=None``.
    """
    channels_raw = _raw.get("channels", {})
    if not isinstance(channels_raw, dict):
        return ChannelsConfig()
    matrix_raw = channels_raw.get("matrix")
    if not matrix_raw or not isinstance(matrix_raw, dict):
        return ChannelsConfig()
    try:
        from minion_assist.matrix.config import MatrixConfig  # noqa: PLC0415
        matrix_cfg = MatrixConfig.from_dict(matrix_raw)
    except Exception as exc:
        raise ConfigError([ConfigIssue("channels.matrix", str(exc))]) from exc
    return ChannelsConfig(matrix=matrix_cfg)


def _resolve_heartbeat() -> HeartbeatConfig:
    """Read the ``"heartbeat"`` section from config.json and build a HeartbeatConfig.

    All fields have sensible defaults so the section can be omitted entirely.

    Returns:
        HeartbeatConfig: Immutable heartbeat settings; defaults apply when absent.
    """
    raw = _raw.get("heartbeat", {})
    if not isinstance(raw, dict):
        return HeartbeatConfig()
    return HeartbeatConfig(
        enabled=raw.get("enabled", False),
        interval_seconds=int(raw.get("interval_seconds", 1800)),
        prompt=raw.get("prompt") or HeartbeatConfig().prompt,
        agent_id=raw.get("agent_id", "main"),
        notification_room_id=raw.get("notification_room_id") or None,
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

# Config lookup order (first file found wins):
#   1. ~/.minion-assist/config.json  — user home; keeps credentials out of projects
#   2. ./config.json                 — current working directory; useful during development
#
# Why not the repo root?  config.json may contain secrets (e.g. Matrix access
# tokens) that must never be committed to version control.  Storing it in the
# user home directory ensures it stays off git even if the developer forgets to
# add it to .gitignore in a new clone.
_CONFIG_CANDIDATES = [
    _HOME / "config.json",
    Path.cwd() / "config.json",
]

_config_path: Path | None = next((p for p in _CONFIG_CANDIDATES if p.exists()), None)

if _config_path is None:
    _searched = "\n".join(f"  {p}" for p in _CONFIG_CANDIDATES)
    raise ConfigError([ConfigIssue(
        "config.json",
        f"File not found. Searched:\n{_searched}\n"
        "Create it at ~/.minion-assist/config.json — "
        "see config.example.json in the repo for the template.",
    )]) from None

try:
    with open(_config_path, encoding="utf-8") as _f:
        _raw = json.load(_f)
except json.JSONDecodeError as exc:
    raise ConfigError([ConfigIssue("config.json", f"Invalid JSON in {_config_path}: {exc}")]) from exc

_issues = _validate(_raw)
_errors = [i for i in _issues if not i.warning]
_warnings = [i for i in _issues if i.warning]
# Print non-fatal warnings to stderr so they are visible without blocking startup.
for _w in _warnings:
    print(f"[minion-assist] Warning: {_w.path}: {_w.message}", file=sys.stderr)
if _errors:
    raise ConfigError(_errors)


# ---------------------------------------------------------------------------
# Module-level exports — these are the things other modules import.
# ---------------------------------------------------------------------------

# agents: dict mapping agent ID → provider+model config.
# Built once at import time; safe to read from anywhere without re-parsing.
agents: dict[str, AgentModelConfig] = _resolve_all()

# workspace: the root folder for all persisted data (history, memory notes).
# expanduser() converts "~/.minion-assist" to the actual home-directory path.
workspace: Path = Path(_raw.get("workspace", {}).get("path", "~/.minion-assist")).expanduser()

# streaming: controls whether token-by-token streaming is active per mode.
# Read from the "streaming" section of config.json; defaults to all-False if absent.
streaming: StreamingConfig = _resolve_streaming()

# compaction: controls when conversation history is summarised to free context window.
# Read from the "compaction" section of config.json; defaults to 4 000 if absent.
compaction: CompactionConfig = _resolve_compaction()

# mcp: all configured MCP servers. Empty servers tuple when section absent from config.
# Built once at import time from the "mcp.servers" section of config.json.
mcp: McpConfig = _resolve_mcp()

# memory: controls the optional background memory extraction subsystem.
# Read from "memory" section; defaults to enable_extraction=True if absent.
memory: MemoryConfig = _resolve_memory()

# bootstrap: controls workspace bootstrap file injection into agent system prompts.
# Read from "bootstrap" section; all fields default so the section can be omitted.
bootstrap: BootstrapConfig = _resolve_bootstrap()

# channels: integrations like Matrix that run alongside the REPL.
# matrix is None when channels.matrix is absent from config.json.
channels: ChannelsConfig = _resolve_channels()

# multi_agent: subagent spawning limits and timeout defaults.
# Read from "multi_agent" section; all fields default when section absent.
multi_agent: MultiAgentConfig = _resolve_multi_agent()

# extra_plugin_manifests: additional plugins.json paths to load beyond the two
# fixed locations (~/.minion-assist/plugins.json and .minion-assist/plugins.json).
# Declared in config.json as: "extra_plugin_manifests": ["/path/to/plugins.json"]
# Paths are resolved at startup; ~ is expanded.
extra_plugin_manifests: tuple[str, ...] = tuple(
    _raw.get("extra_plugin_manifests", [])
)

# heartbeat: controls the periodic background heartbeat scheduler.
# Disabled by default; enabled via "heartbeat": {"enabled": true} in config.json.
heartbeat: HeartbeatConfig = _resolve_heartbeat()


def _resolve_dreaming() -> DreamingConfig:
    """Read the ``"dreaming"`` section from config.json and build a DreamingConfig.

    All fields have sensible defaults so the section can be omitted entirely.

    Returns:
        DreamingConfig: Immutable dreaming settings; defaults apply when absent.
    """
    raw = _raw.get("dreaming", {})
    if not isinstance(raw, dict):
        return DreamingConfig()
    hour = int(raw.get("hour", 3))
    minute = int(raw.get("minute", 0))
    return DreamingConfig(
        enabled=bool(raw.get("enabled", False)),
        hour=max(0, min(23, hour)),
        minute=max(0, min(59, minute)),
        timezone=str(raw.get("timezone", "Australia/Sydney")),
        lookback_days=max(1, int(raw.get("lookback_days", 3))),
        agent_id=str(raw.get("agent_id", "main")),
    )


# dreaming: controls the nightly dream diary scheduler.
# Disabled by default; enabled via "dreaming": {"enabled": true} in config.json.
dreaming: DreamingConfig = _resolve_dreaming()


def _resolve_logging() -> LoggingConfig:
    """Read the ``"logging"`` section from config.json and build a LoggingConfig.

    Defaults to ``llm_requests=True`` so logging is on out of the box.

    Returns:
        LoggingConfig: Immutable logging settings; defaults apply when absent.
    """
    raw = _raw.get("logging", {})
    if not isinstance(raw, dict):
        return LoggingConfig()
    return LoggingConfig(
        llm_requests=raw.get("llm_requests", True),
    )


# logging: controls verbose LLM request/response logging.
# Enabled by default; set "logging": {"llm_requests": false} to suppress.
logging_cfg: LoggingConfig = _resolve_logging()
