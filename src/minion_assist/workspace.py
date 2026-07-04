"""Per-agent workspace management: directory creation, marker-file attestation, and path resolution.

Each agent can have its own workspace directory under ``~/.minion-assist/workspaces/{agent_id}/``.
The workspace holds bootstrap files (``AGENTS.md``, ``SOUL.md``, etc.) and a hidden marker
file (``.workspace-marker``) used to detect if the directory is accidentally deleted during
a session.

Directory resolution
--------------------
:func:`agent_workspace_root` looks for workspaces in this order:

1. ``~/.minion-assist/workspaces/{agent_id}/``  — per-agent workspace
2. ``~/.minion-assist/workspaces/main/``        — shared root workspace (fallback)
3. ``None``                                     — no per-agent workspace; caller uses global root

When ``None`` is returned, the caller (``minion.py``) falls back to the standard
bootstrap-root configured in ``config.json`` — backward-compatible behaviour.

Attestation (Phase 5)
---------------------
:func:`ensure_workspace` writes a ``.workspace-marker`` hash file on first call.
:func:`check_workspace` verifies the marker is still present at the start of
each agent turn.  If the directory or marker has disappeared,
:class:`WorkspaceVanishedError` is raised so the user gets a clear error
instead of a confusing provider failure.

Public API
----------
- :func:`ensure_workspace`        — create dir + marker if missing.
- :func:`check_workspace`         — raise :class:`WorkspaceVanishedError` if marker absent.
- :func:`agent_workspace_root`    — resolve per-agent or shared workspace path.
- :class:`WorkspaceVanishedError` — raised when workspace disappears mid-session.

Talks to
--------
- ``minion.py``           — calls ``ensure_workspace`` at startup, ``agent_workspace_root``
                            per agent, and passes ``workspace_root`` to ``AgentSession``.
- ``agents/session.py``   — calls ``check_workspace`` at the top of every ``send()``.
- ``tools/spawn_subagent.py`` — calls ``agent_workspace_root`` to resolve the
                            subagent's workspace before creating a child session.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Name of the hidden marker file written at workspace creation.
# Its presence (not its hash) is checked before each agent turn.
MARKER_FILENAME = ".workspace-marker"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class WorkspaceVanishedError(RuntimeError):
    """Raised when an agent workspace directory or its marker file has disappeared.

    Caught by ``AgentSession.send()`` — the error propagates up to the REPL
    which prints it and stops the turn rather than calling the provider with a
    broken state.

    Args:
        path (Path): The workspace root that is missing.
    """


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _compute_marker(root: Path) -> str:
    """SHA256 of sorted filenames present in root (excludes the marker itself).

    Used once at workspace creation to record the initial file set.  The hash
    is written to ``.workspace-marker`` so the directory's identity is captured
    without listing every file on every turn.

    Args:
        root: Workspace directory to hash.

    Returns:
        str: Hex digest of the sorted file list.
    """
    paths = sorted(str(p) for p in root.iterdir() if p.is_file() and p.name != MARKER_FILENAME)
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_workspace(root: Path) -> None:
    """Create the workspace directory and write a marker file if missing.

    Safe to call multiple times — ``mkdir(exist_ok=True)`` and the marker-exists
    check are both idempotent.  Called once per agent at process startup.

    The marker file records the initial file set as a SHA256 hash.  It is written
    only when first creating the directory (or when it somehow went missing and
    the directory was re-created externally).

    Args:
        root: Workspace directory to create.  Parent directories are created
            automatically.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER_FILENAME
    if not marker.exists():
        marker.write_text(_compute_marker(root), encoding="utf-8")


def check_workspace(root: Path) -> None:
    """Verify that the workspace directory and marker file still exist.

    Called at the top of ``AgentSession.send()`` when a ``workspace_root`` is
    set.  Only existence is checked (not the hash) — re-hashing on every turn
    would be too expensive and the goal is to catch accidental deletion, not
    detect modifications.

    Args:
        root: Workspace directory to verify.

    Raises:
        WorkspaceVanishedError: When the directory or ``.workspace-marker`` is
            missing.  The message includes the path so the user knows what to
            restore.
    """
    if not root.exists() or not (root / MARKER_FILENAME).exists():
        raise WorkspaceVanishedError(
            f"Workspace vanished: {root}. "
            "The workspace directory or its marker file is missing. "
            "Restore the directory or restart the session to reinitialise it."
        )


def agent_workspace_root(workspace: Path, agent_id: str) -> Path | None:
    """Resolve the workspace root for a given agent.

    Checks for a per-agent workspace directory first, then falls back to the
    shared ``main`` workspace.  Returns ``None`` when neither exists, signalling
    that the caller should use the global bootstrap root from ``config.json``
    (backward-compatible behaviour — users who have not set up per-agent
    workspaces are unaffected).

    Resolution order:
    1. ``{workspace}/workspaces/{agent_id}/``  — per-agent override
    2. ``{workspace}/workspaces/main/``        — shared workspace (any agent)
    3. ``None``                                — no per-agent workspace

    Args:
        workspace: The user's minion-assist home directory (e.g.
            ``~/.minion-assist``).
        agent_id: The agent whose workspace to resolve (e.g. ``"main"``,
            ``"researcher"``).

    Returns:
        Path | None: Resolved workspace root, or ``None`` if neither directory
            exists.
    """
    per_agent = workspace / "workspaces" / agent_id
    if per_agent.exists():
        return per_agent
    shared = workspace / "workspaces" / "main"
    if shared.exists():
        return shared
    return None
