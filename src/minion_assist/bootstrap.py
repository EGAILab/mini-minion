"""Bootstrap prompt layer — load workspace files and inject them into agent system prompts.

This module implements the Project Context bootstrap layer for Minion Assist.
It mirrors the OpenClaw workspace bootstrap behaviour: recognized workspace files
are discovered, budget-clamped, and rendered into a ``# Project Context`` block
that is injected into each agent's system prompt on every turn.

The key design goals are:

- **Live updates** — files are re-read on every turn so edits to ``AGENTS.md``,
  ``SOUL.md``, etc. take effect without restarting the process.
- **Bounded cost** — a per-file char cap (default 20 000) and a total cap
  (default 60 000) prevent runaway prompt growth from large files.
- **Deterministic ordering** — files are always injected in the canonical order
  listed in ``_BOOTSTRAP_FILES``, regardless of filesystem ordering.
- **Safe reads** — every candidate path is resolved and checked against the
  configured root before any bytes are read; symlink escapes are rejected.
- **Graceful degradation** — missing files are silently skipped; decode errors
  fall back to UTF-8 replacement characters.

Recognized files (in injection order):

    AGENTS.md   SOUL.md   TOOLS.md   IDENTITY.md   USER.md   BOOTSTRAP.md
    MEMORY.md   KNOWLEDGE_DIGEST.md

``HEARTBEAT.md`` is intentionally omitted from ordinary turns (no heartbeat
runs exist in Minion Assist yet). ``KNOWLEDGE_DIGEST.md`` (Stage One Phase
7, slice D) is a fully machine-compiled file — see
``memory/knowledge.py``'s ``compile_digest`` and
``memory/digest_scheduler.py`` — read here exactly like any other
bootstrap file, no different handling needed.

Typical call path::

    build_bootstrap_prompt_block(root, bootstrap_cfg)   # called per turn by AgentSession

Which internally runs::

    load_bootstrap_files(root)
    build_bootstrap_context_files(files, max_chars, total_max_chars)
    render_bootstrap_pending_context(ctx_files)  # when BOOTSTRAP.md is present
    render_project_context(ctx_files)
    render_truncation_warning(ctx_files, mode)

Public API
----------
- :func:`load_bootstrap_files`
- :func:`build_bootstrap_context_files`
- :func:`render_project_context`
- :func:`render_bootstrap_pending_context`
- :func:`render_truncation_warning`
- :func:`build_bootstrap_prompt_block`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognized bootstrap file names in canonical injection order.
# HEARTBEAT.md is injected after BOOTSTRAP.md so the agent sees its checklist
# on every turn without needing an explicit read call.
_BOOTSTRAP_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "MEMORY.md",
    "KNOWLEDGE_DIGEST.md",
)

# Subset of bootstrap files injected into spawned subagents.
# Subagents do not receive SOUL/IDENTITY/USER — those define the root agent's
# character.  They only need AGENTS.md (available tools/agents) and TOOLS.md
# (tool usage instructions) so they can act effectively on their delegated task.
_SUBAGENT_BOOTSTRAP_FILES: tuple[str, ...] = ("AGENTS.md", "TOOLS.md")

# Maximum raw bytes to read from a single file before decoding.
# Prevents OOM from pathologically large workspace files.
# Mirrors OpenClaw's 2 MB raw file cap.
_RAW_FILE_CAP: int = 2 * 1024 * 1024  # 2 MB

# Process-local flag for "once" truncation warning mode.
# Set to True after the first warning is emitted; reset only on process restart.
_truncation_warned: bool = False

# ---------------------------------------------------------------------------
# Bootstrap block cache — avoids rebuilding the 60 K-char block every turn
# when workspace files have not changed on disk.
#
# Cache key: (resolved root path, config discriminator string, session_type).
# Cache value: (snapshot of {filepath: mtime}, rendered block string).
# On a hit, mtimes are re-checked; if any file changed the block is rebuilt.
# ---------------------------------------------------------------------------
_bootstrap_block_cache: dict[tuple[str, str, str], tuple[dict[str, float], str]] = {}


def _get_bootstrap_mtimes(root: Path, allowed_names: tuple[str, ...] | None) -> dict[str, float]:
    """Return {absolute_path_str: mtime} for each candidate bootstrap file that exists.

    Only files that actually exist on disk appear in the result.  Absent files
    are excluded rather than recorded as 0.0 so a file that appears or disappears
    correctly invalidates the cache.

    Args:
        root: Bootstrap root directory.
        allowed_names: Tuple of filenames to check, or None to check all in
            ``_BOOTSTRAP_FILES``.

    Returns:
        Dict mapping absolute path string → ``st_mtime`` float.
    """
    names = allowed_names if allowed_names is not None else _BOOTSTRAP_FILES
    mtimes: dict[str, float] = {}
    for name in names:
        p = root / name
        try:
            mtimes[str(p)] = p.stat().st_mtime
        except OSError:
            pass  # file absent — omit so its appearance triggers a cache miss
    return mtimes


def clear_bootstrap_block_cache() -> None:
    """Clear the process-level bootstrap block cache.

    Intended for use in tests that modify bootstrap files within a single
    process and need to guarantee a cache miss on the next call.
    """
    _bootstrap_block_cache.clear()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapFile:
    """Result of attempting to load one recognized bootstrap file from disk.

    Attributes:
        name: The filename (e.g. ``"AGENTS.md"``).
        path: Absolute path to the candidate file location, whether it exists or not.
        content: Decoded file text, or ``None`` when the file is missing or
            could not be read (permission denied, decode failure, etc.).
        missing: ``True`` when the file does not exist at ``path``.
    """

    name: str
    path: Path
    content: str | None
    missing: bool = False


@dataclass(frozen=True)
class ContextFile:
    """A bootstrap file after budget limits have been applied.

    Instances are produced by :func:`build_bootstrap_context_files`.  The
    ``content`` field holds the exact text that will be injected into the
    prompt, which may be a head+tail truncation of the raw file content.

    Attributes:
        path: Absolute path to the source file.
        content: Text that will be injected (may be truncated).
        truncated: ``True`` when the raw content was clipped to fit the budget.
        raw_chars: Length of the original (stripped) file text before truncation.
        injected_chars: Length of the text actually injected (``len(content)``).
    """

    path: Path
    content: str
    truncated: bool = False
    raw_chars: int = 0
    injected_chars: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str | None:
    """Read a file, capping at _RAW_FILE_CAP bytes. Returns None on any error.

    Uses UTF-8 with ``errors="replace"`` so a single bad byte does not abort
    the entire read.  Reading more than ``_RAW_FILE_CAP`` bytes is prevented
    at the binary level before decoding to avoid allocating huge strings.
    """
    try:
        raw = path.read_bytes()
        if len(raw) > _RAW_FILE_CAP:
            # Trim at byte level before decoding to keep memory bounded.
            raw = raw[:_RAW_FILE_CAP]
        return raw.decode("utf-8", errors="replace")
    except Exception:  # OSError, PermissionError, IsADirectoryError, …
        return None


def _truncate_content(raw_content: str, limit: int, name: str) -> str:
    """Produce a head+tail excerpt of raw_content that fits within limit chars.

    Injects a visible marker between the head and tail so the model knows
    the file was clipped and should read the full file when details matter.

    The marker itself adds a small number of extra chars beyond ``limit``.
    This is intentional — the marker is metadata, not content, and the
    slight overshoot is negligible relative to default limits.

    Deliberately uses the bare filename, not the absolute path: an absolute
    path's length is unbounded (deep nested directories), so embedding it
    here could itself blow past ``limit`` for a small budget, working against
    the very truncation this function exists to do. The absolute path is
    available anyway, from the ``## name (path)`` section header directly
    above this content (:func:`render_project_context`) and from
    :func:`render_truncation_warning` when this file was truncated.

    Example marker::

        [...truncated, read AGENTS.md for full content...]
    """
    head_size = limit // 2
    tail_size = limit - head_size
    marker = f"\n[...truncated, read {name} for full content...]\n"
    return raw_content[:head_size] + marker + raw_content[-tail_size:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_bootstrap_files(
    root: Path,
    allowed_names: tuple[str, ...] | None = None,
) -> list[BootstrapFile]:
    """Discover and load recognized bootstrap files from root.

    Files are returned in the fixed canonical order defined by
    ``_BOOTSTRAP_FILES`` regardless of filesystem ordering.  Missing files
    are represented with ``missing=True`` and ``content=None`` so callers
    can distinguish "absent" from "present but empty".

    Security constraints:

    - The ``root`` path is resolved to an absolute, symlink-free path.
    - Each candidate file path is also resolved.
    - A file is read only when its resolved path is contained within the
      resolved root (``relative_to`` check).  Symlink escapes and ``..``
      traversal are therefore rejected.
    - Raw read size is capped at :data:`_RAW_FILE_CAP` bytes.

    Args:
        root: Directory to search for bootstrap files.
        allowed_names: When provided, only files whose name appears in this
            tuple are loaded.  Used by :func:`build_bootstrap_prompt_block`
            with ``session_type="subagent"`` to restrict to
            :data:`_SUBAGENT_BOOTSTRAP_FILES`.  ``None`` loads all files in
            :data:`_BOOTSTRAP_FILES` (the default for root agents).

    Returns:
        A list of :class:`BootstrapFile` objects, one per recognized filename,
        in canonical injection order.
    """
    names = allowed_names if allowed_names is not None else _BOOTSTRAP_FILES
    # Resolve once; used for every containment check below.
    try:
        resolved_root = root.resolve()
    except OSError:
        # Unresolvable root — treat all files as missing.
        return [
            BootstrapFile(name=name, path=root / name, content=None, missing=True)
            for name in names
        ]

    result: list[BootstrapFile] = []
    for name in names:
        candidate = root / name

        # Security: reject any path that resolves outside the root.
        try:
            candidate.resolve().relative_to(resolved_root)
        except (ValueError, OSError):
            # Path escape or resolution failure — silently treat as missing.
            result.append(BootstrapFile(name=name, path=candidate, content=None, missing=True))
            continue

        if not candidate.exists():
            result.append(BootstrapFile(name=name, path=candidate, content=None, missing=True))
            continue

        content = _read_file(candidate)
        # content is None on read/decode failure; missing=False because the file exists.
        result.append(BootstrapFile(name=name, path=candidate, content=content, missing=False))

    return result


def build_bootstrap_context_files(
    files: list[BootstrapFile],
    max_chars: int,
    total_max_chars: int,
) -> list[ContextFile]:
    """Apply per-file and total budget limits to the loaded bootstrap files.

    Processing rules:

    1. Files that are missing or have no content are skipped entirely.
    2. Each file's content is stripped of leading/trailing whitespace before
       measuring.
    3. The effective char limit for any file is
       ``min(max_chars, remaining_total_budget)``.
    4. If the stripped content exceeds the effective limit it is truncated
       with a head+tail split (see :func:`_truncate_content`).
    5. Processing stops once the total budget is exhausted, even if more files
       remain.

    Args:
        files: Loaded bootstrap files in canonical order (from
            :func:`load_bootstrap_files`).
        max_chars: Per-file character limit.
        total_max_chars: Cumulative character limit across all files.

    Returns:
        Ordered list of :class:`ContextFile` objects ready for rendering.
        Only files with usable content are included.
    """
    result: list[ContextFile] = []
    total_used = 0

    for bf in files:
        # Skip absent or unreadable files.
        if bf.missing or bf.content is None:
            continue
        raw_content = bf.content.strip()
        if not raw_content:
            continue

        remaining_budget = total_max_chars - total_used
        if remaining_budget <= 0:
            break

        # Effective limit is the tighter of per-file cap and remaining budget.
        limit = min(max_chars, remaining_budget)
        raw_chars = len(raw_content)

        if raw_chars <= limit:
            content = raw_content
            truncated = False
        else:
            content = _truncate_content(raw_content, limit, bf.name)
            truncated = True

        injected_chars = len(content)
        result.append(ContextFile(
            path=bf.path,
            content=content,
            truncated=truncated,
            raw_chars=raw_chars,
            injected_chars=injected_chars,
        ))
        total_used += injected_chars

    return result


def render_project_context(files: list[ContextFile]) -> str:
    """Render the Project Context block from a list of budget-applied files.

    Wraps all injected files under a ``# Project Context`` top-level heading
    with each file as a ``## FILENAME`` section.  Returns an empty string
    when ``files`` is empty.

    Args:
        files: Budget-applied context files from :func:`build_bootstrap_context_files`.

    Returns:
        A multi-line string suitable for inclusion in a system prompt, or
        ``""`` when there are no files to inject.
    """
    if not files:
        return ""
    parts = ["# Project Context"]
    for ctx_file in files:
        # The absolute path (not just the bare filename) lets the model read
        # the file itself via a shell/read tool if it wants more than this
        # embedded excerpt — a bare name resolves relative to whatever cwd
        # that tool happens to use, which is not reliably this file's location.
        parts.append(f"## {ctx_file.path.name} ({ctx_file.path})\n{ctx_file.content}")
    return "\n\n".join(parts)


def render_bootstrap_pending_context(files: list[ContextFile]) -> str:
    """Return a bootstrap-pending guidance block when BOOTSTRAP.md is present.

    When ``BOOTSTRAP.md`` exists in the workspace and has content, the agent
    should handle the bootstrap workflow described in it before responding
    generically.  This function injects that guidance so the model sees it
    prominently before the Project Context block.

    Returns an empty string when BOOTSTRAP.md is not among the injected files.

    Args:
        files: Budget-applied context files from :func:`build_bootstrap_context_files`.

    Returns:
        A ``<bootstrap_pending>`` block string, or ``""`` if BOOTSTRAP.md is absent.
    """
    bootstrap_present = any(f.path.name == "BOOTSTRAP.md" for f in files)
    if not bootstrap_present:
        return ""
    return (
        "<bootstrap_pending>\n"
        "A BOOTSTRAP.md workflow file is present in the workspace. "
        "Handle the bootstrap workflow described in the Project Context below "
        "before responding generically. Complete the bootstrap steps once, "
        "then proceed normally.\n"
        "</bootstrap_pending>"
    )


def render_truncation_warning(files: list[ContextFile], mode: str) -> str:
    """Return a truncation warning when one or more files were clipped.

    Warning modes:

    - ``"off"``    — never emit the warning.
    - ``"once"``   — emit at most one warning per process lifetime (module-level flag).
    - ``"always"`` — emit the warning on every turn that has truncated content.

    Args:
        files: Budget-applied context files (truncated status is on each object).
        mode: One of ``"off"``, ``"once"``, or ``"always"``.

    Returns:
        Warning string, or ``""`` when suppressed by mode or no truncation occurred.
    """
    global _truncation_warned

    if mode == "off":
        return ""

    has_truncation = any(f.truncated for f in files)
    if not has_truncation:
        return ""

    if mode == "once":
        if _truncation_warned:
            return ""
        _truncation_warned = True

    # Absolute paths, not bare names — a bare name resolves relative to
    # whatever cwd the model's shell/read tool happens to use, which is not
    # reliably where these files actually live.
    truncated_paths = "\n".join(f"- {f.path}" for f in files if f.truncated)
    return (
        "[Bootstrap truncation warning]\n"
        "Some workspace bootstrap files were truncated before injection.\n"
        "Treat Project Context as partial and read the relevant files directly "
        "if details seem missing:\n"
        f"{truncated_paths}"
    )


def build_bootstrap_prompt_block(
    root: Path,
    config: object,
    session_type: str = "root",
) -> str:
    """Build the complete bootstrap prompt block for one agent turn.

    This is the main entry point called per turn from :class:`AgentSession`.
    It chains all the lower-level helpers:

    1. Load recognized files from ``root`` (filtered by ``session_type``).
    2. Apply per-file and total budget limits.
    3. If BOOTSTRAP.md is present, prepend a bootstrap-pending guidance block.
    4. Render the ``# Project Context`` section.
    5. Append a truncation warning when any file was clipped (respecting mode).

    The ``config`` argument is accessed via duck-typing; it must expose:

    - ``config.enabled`` (bool)
    - ``config.max_chars`` (int)
    - ``config.total_max_chars`` (int)
    - ``config.truncation_warning`` (str: "off", "once", or "always")

    Args:
        root: Bootstrap root directory (typically ``Path.cwd()``).
        config: A config object with the attributes listed above.
        session_type: ``"root"`` (default) injects all bootstrap files.
            ``"subagent"`` restricts to :data:`_SUBAGENT_BOOTSTRAP_FILES`
            (``AGENTS.md`` + ``TOOLS.md`` only) — subagents do not receive
            ``SOUL.md``, ``IDENTITY.md``, or ``USER.md`` which define the
            root agent's character.

    Returns:
        The complete block string to inject after the static agent soul, or
        ``""`` when bootstrap is disabled or no files have content.
    """
    if not config.enabled:
        return ""

    allowed = _SUBAGENT_BOOTSTRAP_FILES if session_type == "subagent" else None

    # Fast path: if no bootstrap file has changed on disk since the last build,
    # return the cached block.  This avoids reading 7–8 files and concatenating
    # ~60 K chars on every agent turn.
    #
    # Cache key discriminates on root, config limits, and session type so
    # different roots, budget settings, or session types never share a slot.
    config_key = (
        f"{config.max_chars}:{config.total_max_chars}:{config.truncation_warning}"
    )
    cache_key = (str(root.resolve()), config_key, session_type)
    current_mtimes = _get_bootstrap_mtimes(root, allowed)

    cached = _bootstrap_block_cache.get(cache_key)
    if cached is not None:
        cached_mtimes, cached_block = cached
        if cached_mtimes == current_mtimes:
            return cached_block

    # Cache miss — rebuild from disk.
    files = load_bootstrap_files(root, allowed_names=allowed)
    ctx_files = build_bootstrap_context_files(
        files,
        config.max_chars,
        config.total_max_chars,
    )

    if not ctx_files:
        # Cache the empty result too so we don't hit disk again next turn.
        _bootstrap_block_cache[cache_key] = (current_mtimes, "")
        return ""

    parts: list[str] = []

    # Bootstrap-pending guidance comes before the project context block so the
    # model sees the instruction prominently before the file contents.
    pending = render_bootstrap_pending_context(ctx_files)
    if pending:
        parts.append(pending)

    # Main project context block with all budget-applied files.
    parts.append(render_project_context(ctx_files))

    # Truncation warning at the end so the model sees it after the content.
    warning = render_truncation_warning(ctx_files, config.truncation_warning)
    if warning:
        parts.append(warning)

    result = "\n\n".join(parts)
    _bootstrap_block_cache[cache_key] = (current_mtimes, result)
    return result


# ---------------------------------------------------------------------------
# User name extraction
# ---------------------------------------------------------------------------

# Patterns tried in order when extracting the user's name from USER.md.
# 1. Key-value line, with optional markdown bold around the key and/or colon:
#      "Name: Alice" / "- Name: Alice" / "- **Name:** Alice" / "**Name:** Alice"
# 2. First H1 heading: "# Alice"
_NAME_KV_RE = re.compile(r"^[-*]?\s*\**[Nn]ame\**\s*:\**\s*(.+)", re.MULTILINE)
_NAME_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def read_user_name(root: Path) -> str | None:
    """Return the user's name from USER.md in *root*, or ``None`` if not found.

    Tries two patterns in order:
    - A ``Name: <value>`` line (also accepts ``- Name:`` or ``* Name:``).
    - The first top-level heading (``# <name>``).

    Args:
        root: Directory containing workspace bootstrap files.

    Returns:
        str | None: Stripped name string, or ``None`` when USER.md is absent
            or contains no recognizable name field.
    """
    user_md = root / "USER.md"
    try:
        text = user_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    m = _NAME_KV_RE.search(text) or _NAME_H1_RE.search(text)
    if m:
        return m.group(1).strip() or None
    return None
