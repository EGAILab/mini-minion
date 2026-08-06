"""``minion-assist config`` CLI subcommand — the effective, secret-redacted configuration.

MEM-GAP-019 ("Configuration and administrative surface lag implemented
behavior" — see ``minion-assist-docs/improve/openclaw-memory-gap-analysis.md``):
``config.py`` validates and resolves ~20 top-level ``config.json`` sections
into typed dataclasses at import time, but before this module there was no
way to see the *result* of that resolution without reading source or
instrumenting the running process. An operator setting up a new deployment
could only confirm what they *intended* to write in ``config.json`` — not
what minion-assist actually understood from it (e.g. whether an agent's
``model`` string resolved to a real provider, or whether ``embeddings`` is
actually active).

Why redact rather than just not print secrets at all?
--------------------------------------------------------
Printing "``api_key: ***``" (present, masked) instead of omitting the field
lets an operator confirm a key *is* configured without ever exposing its
value — omitting the field entirely would look identical to "not
configured," the exact ambiguity this report exists to remove.

Why walk dataclasses generically instead of a per-section formatter?
-------------------------------------------------------------------------
There are ~20 heterogeneous config dataclasses (some nested two or three
levels deep, e.g. ``channels.matrix.dm``). A hand-written formatter per
class would need updating every time a field is added anywhere in
``config.py`` — an easy place to silently forget a redaction. Walking
``dataclasses.fields()`` generically means a newly added field is rendered
(and redacted, if its name matches the known-sensitive set) automatically,
with no separate formatter to remember to update.

Talks to
--------
- ``config.py`` — every module-level resolved config export
  :func:`format_config_report` walks and renders.
- ``minion.py`` — dispatches ``minion-assist config`` to :func:`main` here,
  the same way it already dispatches ``minion-assist memory ...`` to
  ``memory/cli.py``, before any REPL/session/provider setup — this report
  only needs already-resolved config, never a live provider connection.
"""

from __future__ import annotations

import dataclasses
import re
import sys

# Field names whose entire value is always secret, regardless of content —
# masked outright rather than rendered. Exhaustive as of this writing
# (ProviderConfig.api_key, MatrixConfig.access_token/password); a new
# secret-bearing field elsewhere would need adding here explicitly, since
# there's no way to detect "this string is a credential" from the value
# alone.
_REDACT_WHOLE_FIELD = frozenset({"api_key", "access_token", "password"})

# Field names that are themselves dicts of arbitrary, caller-defined
# entries (MCP server env vars / HTTP headers) — any entry could be a
# credential, and there's no reliable way to tell which ones from the
# field name alone, so every value in these two dicts is masked rather
# than guessing per-key.
_REDACT_DICT_VALUES = frozenset({"env", "headers"})

_MASK = "***"


def _redact_url(url: str) -> str:
    """Strip inline ``user:pass@`` credentials from a connection string, keeping host/path visible.

    ``postgresql://minion:minion@localhost:5433/minion_assist`` becomes
    ``postgresql://***@localhost:5433/minion_assist`` — enough to confirm
    *which* database is configured without exposing its password.
    """
    return re.sub(r"://[^@/\s]+@", f"://{_MASK}@", url)


def _render(value: object, *, field_name: str | None, indent: int) -> list[str]:
    """Recursively render one resolved config value as indented text lines, redacting secrets.

    Args:
        value: The value to render — a dataclass instance, dict, list/tuple,
            or a plain scalar (str/int/float/bool/None).
        field_name: This value's field name in its parent, or ``None`` for
            a top-level (unnamed) value — used both as the printed label
            and to check against the redaction sets above.
        indent: Nesting depth, in two-space increments.

    Returns:
        list[str]: One or more lines, deepest-first ordering matching the
            dataclass field order (or dict insertion order).
    """
    prefix = "  " * indent
    label = f"{field_name}: " if field_name is not None else ""

    if field_name in _REDACT_WHOLE_FIELD:
        return [f"{prefix}{label}{_MASK if value else '(not set)'}"]

    if value is None:
        return [f"{prefix}{label}(not configured)"]

    if dataclasses.is_dataclass(value):
        lines = [f"{prefix}{field_name}:"] if field_name is not None else []
        child_indent = indent + 1 if field_name is not None else indent
        for f in dataclasses.fields(value):
            lines.extend(_render(getattr(value, f.name), field_name=f.name, indent=child_indent))
        return lines

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{label}(none)"]
        if field_name in _REDACT_DICT_VALUES:
            lines = [f"{prefix}{field_name}:"]
            for k, v in value.items():
                lines.append(f"{prefix}  {k}: {_MASK if v else '(empty)'}")
            return lines
        lines = [f"{prefix}{field_name}:"] if field_name is not None else []
        child_indent = indent + 1 if field_name is not None else indent
        for k, v in value.items():
            if dataclasses.is_dataclass(v):
                lines.extend(_render(v, field_name=str(k), indent=child_indent))
            else:
                lines.append(f"{'  ' * child_indent}{k}: {v!r}")
        return lines

    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{prefix}{label}(none)"]
        if all(dataclasses.is_dataclass(v) for v in value):
            lines = [f"{prefix}{field_name}:"]
            for i, v in enumerate(value):
                lines.extend(_render(v, field_name=f"[{i}]", indent=indent + 1))
            return lines
        return [f"{prefix}{label}{list(value)!r}"]

    if field_name == "url" and isinstance(value, str) and "@" in value:
        return [f"{prefix}{label}{_redact_url(value)!r}"]

    return [f"{prefix}{label}{value!r}"]


def format_config_report() -> str:
    """Render every resolved top-level ``config.json`` section as redacted text.

    Imports ``config`` lazily (not at module load) so this module itself
    stays importable — and therefore its pure rendering logic testable —
    even in a process where ``config.json`` hasn't been set up yet.
    Importing ``minion_assist.config`` is what actually loads/validates/
    resolves the file; by the time this function runs that has already
    either succeeded or raised ``ConfigError`` (caught by ``main()``).

    Returns:
        str: Multi-line report, one top-level section per block, secrets
            masked per :data:`_REDACT_WHOLE_FIELD`/:data:`_REDACT_DICT_VALUES`.
    """
    lines: list[str] = []
    for name, value in _config_sections():
        lines.append(f"{name}:")
        lines.extend(_render(value, field_name=None, indent=1))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _config_sections() -> list[tuple[str, object]]:
    """Every top-level resolved ``config.json`` section, name-labeled.

    Shared by :func:`format_config_report` and :func:`collect_secret_values`
    (MEM-GAP-017) so both walk the exact same section list — a new config
    section added to one automatically appears in the other, with no
    separate list to remember to update.
    """
    from . import config as _cfg  # noqa: PLC0415

    return [
        ("agents", _cfg.agents),
        ("workspace", _cfg.workspace),
        ("streaming", _cfg.streaming),
        ("compaction", _cfg.compaction),
        ("mcp", _cfg.mcp),
        ("memory", _cfg.memory),
        ("bootstrap", _cfg.bootstrap),
        ("channels", _cfg.channels),
        ("multi_agent", _cfg.multi_agent),
        ("heartbeat", _cfg.heartbeat),
        ("dreaming", _cfg.dreaming),
        ("memory_consolidation", _cfg.memory_consolidation),
        ("memory_reconciliation", _cfg.memory_reconciliation),
        ("commitments", _cfg.commitments),
        ("knowledge_digest", _cfg.knowledge_digest),
        ("memory_retention", _cfg.memory_retention),
        ("logging", _cfg.logging_cfg),
        ("codex", _cfg.codex_cfg),
        ("database", _cfg.database),
        ("embeddings", _cfg.embeddings),
        ("voice", _cfg.voice),
    ]


def _collect_secrets(value: object, *, field_name: str | None, out: set[str]) -> None:
    """Recursively gather every known-sensitive value out of a resolved config value into ``out``.

    Same traversal shape as :func:`_render`, but collects raw values
    instead of building display text — see :func:`collect_secret_values`.
    """
    if field_name in _REDACT_WHOLE_FIELD:
        if isinstance(value, str) and value:
            out.add(value)
        return
    if value is None:
        return
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _collect_secrets(getattr(value, f.name), field_name=f.name, out=out)
        return
    if isinstance(value, dict):
        if field_name in _REDACT_DICT_VALUES:
            for v in value.values():
                if isinstance(v, str) and v:
                    out.add(v)
            return
        for v in value.values():
            _collect_secrets(v, field_name=None, out=out)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _collect_secrets(v, field_name=None, out=out)
        return
    if field_name == "url" and isinstance(value, str):
        # A connection string's inline user:pass@ credential, not the whole
        # URL (host/port/db name aren't secret and are useful in a log).
        match = re.search(r"://([^@/\s]+)@", value)
        if match:
            out.add(match.group(1))


def collect_secret_values() -> frozenset[str]:
    """Every currently-configured secret value, across every resolved config section (MEM-GAP-017).

    Used to redact known credentials from LLM request/response logs
    (``llm_logger.py``) — masks exact occurrences of a real, currently-
    configured secret wherever it appears in logged text (a provider
    ``api_key``, Matrix's ``access_token``/``password``, a database URL's
    inline credential, an MCP server's ``env``/``headers`` values),
    complementing rather than replacing careful defaults. This is
    deliberately *known-value* matching, not pattern-guessing: a secret
    that was never configured through ``config.json``/``.env`` (e.g. a key
    a user pasted directly into a memory note) can't be recognized as a
    secret without already knowing it's one — see the module docstring's
    "Why redact rather than just not print secrets at all?" note for the
    same reasoning applied to :func:`format_config_report`.

    Returns:
        frozenset[str]: Every non-empty secret value found. Cached by the
            caller (``llm_logger.py`` computes this once, not per log
            call) since config is immutable after process startup.
    """
    out: set[str] = set()
    for _name, value in _config_sections():
        _collect_secrets(value, field_name=None, out=out)
    return frozenset(out)


def main(argv: list[str]) -> int:
    """Entry point for the ``config`` CLI subcommand.

    Args:
        argv: Arguments after the leading ``config`` token. Currently
            ignored — there is only one report to show; a subcommand tree
            (mirroring ``memory/cli.py``'s) can grow here if a second
            config-related action is ever needed.

    Returns:
        int: Process exit code — 2 if ``config.json`` fails to load/
            validate (the ``ConfigError`` itself already prints its own
            actionable message via config.py's module-load error path),
            0 otherwise.
    """
    del argv  # unused for now — see docstring
    try:
        report = format_config_report()
    except Exception as exc:  # config.py's own ConfigError, or an import-time failure
        print(f"Failed to load configuration: {exc}")
        return 2
    try:
        print(report, end="")
    except UnicodeEncodeError:
        # A config value (e.g. Matrix's default emoji ack_reaction) can
        # contain a character the terminal's active codepage can't render
        # (common on a default Windows cmd.exe, cp1252) — fall back to a
        # lossy-but-never-crashing render rather than losing the whole
        # report to a stdout encoding error.
        encoding = sys.stdout.encoding or "ascii"
        print(report.encode(encoding, errors="replace").decode(encoding), end="")
    return 0
