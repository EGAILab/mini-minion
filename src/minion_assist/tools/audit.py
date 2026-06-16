"""Permission audit log and approval decision types.

Records every tool permission decision to an in-memory log so users can
review what the agent attempted during a session.  The log is session-scoped
(not persisted to disk).  Use ``/audit`` to print recent entries.

ApprovalDecision
----------------
Captures the user's response to a bash confirmation prompt:

- ``ALLOW_ONCE``    — approved for this single invocation only.
- ``ALLOW_SESSION`` — approved; no re-prompting for the same command this session.
- ``DENY``          — rejected for this invocation only.
- ``ALWAYS_DENY``   — rejected and auto-blocked for all future invocations.

Talks to
--------
- ``tools/policy.py`` — :class:`~minion_assist.tools.policy.PermissionPolicy` holds an
  :class:`AuditLog` instance and records denials from its check methods.
- ``tools/bash.py``   — :class:`~minion_assist.tools.bash.BashTool` records
  user approval/denial decisions and consults the session-level allow/deny sets.
- ``commands.py``     — the ``/audit`` command reads :attr:`AuditLog.entries` and
  formats them for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class ApprovalDecision(Enum):
    """Outcome of a per-command user confirmation prompt."""
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    ALWAYS_DENY = "always_deny"


@dataclass
class AuditEntry:
    """One recorded permission event.

    Attributes:
        timestamp:  ISO 8601 UTC time of the event.
        tool_name:  Tool that triggered the check, e.g. ``"bash"``, ``"write"``.
        args_repr:  Short string representation of the key argument (command or path),
                    truncated to 120 characters for display.
        decision:   ``"allowed"`` or ``"denied"`` (or an :class:`ApprovalDecision` value).
        reason:     Human-readable explanation of why (e.g. the policy rule name,
                    or ``"user deny"`` / ``"always deny"``).  Empty string when the
                    action was allowed with no special reason.
    """
    timestamp: str
    tool_name: str
    args_repr: str
    decision: str
    reason: str = ""


def _utcnow() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class AuditLog:
    """In-memory audit log for one session.

    Bounded at 1 000 entries — oldest are evicted when the log is full.
    Also tracks session-level approval state so :class:`BashTool` can
    skip re-prompting for ALLOW_SESSION commands and auto-block ALWAYS_DENY ones.

    Usage::

        log = AuditLog()
        log.record(AuditEntry(timestamp=_utcnow(), tool_name="bash",
                               args_repr="ls", decision="allowed"))
        for entry in log.entries:
            print(entry)
    """

    _MAX = 1_000  # evict oldest when full

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        # Commands approved for the whole session (ALLOW_SESSION).
        self._session_allowed: set[str] = set()
        # Commands blocked for the whole session (ALWAYS_DENY).
        self._session_denied: set[str] = set()

    def record(self, entry: AuditEntry) -> None:
        """Append an entry, evicting the oldest entry when at capacity."""
        if len(self._entries) >= self._MAX:
            self._entries.pop(0)
        self._entries.append(entry)

    @property
    def entries(self) -> list[AuditEntry]:
        """All recorded entries, oldest first (defensive copy)."""
        return list(self._entries)

    def is_session_allowed(self, command: str) -> bool:
        """Return True when a prior ALLOW_SESSION decision covers this exact command."""
        return command in self._session_allowed

    def is_session_denied(self, command: str) -> bool:
        """Return True when a prior ALWAYS_DENY decision covers this exact command."""
        return command in self._session_denied

    def set_session_allowed(self, command: str) -> None:
        """Store an ALLOW_SESSION approval so future prompts for this command are skipped."""
        self._session_allowed.add(command)

    def set_session_denied(self, command: str) -> None:
        """Store an ALWAYS_DENY block so this command is auto-rejected this session."""
        self._session_denied.add(command)
