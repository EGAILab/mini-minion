"""JSON-backed session metadata store.

This module tracks lightweight statistics about each agent's usage: when the
session was created, when it was last active, and how many conversation turns
have happened. It's separate from :class:`ShortTermMemory` (which stores the
actual messages) because session metadata has a different lifecycle and shape.

Why track session metadata?
----------------------------
- **Observability**: you can query ``sessions.json`` to see which agents have
  been used, how active each is, and when the last interaction was.
- **Future features**: turn counts and timestamps are useful for things like
  "summarize and compact history when turn_count > N" or "warn if the session
  has been idle for X days".
- **Decoupled**: keeping metadata separate from message history means you can
  clear the history without losing the metadata, or vice versa.

Storage format
--------------
A single ``sessions.json`` file in the workspace root. Example:
::

    {
      "main": {
        "agent_id": "main",
        "created_at": "2026-05-28T10:00:00+00:00",
        "last_active": "2026-05-28T12:30:00+00:00",
        "turn_count": 15
      },
      "researcher": { ... }
    }

Talks to
--------
- ``minion.py`` — creates a :class:`SessionStore` at startup, calls
  ``get_or_create()`` once per agent, and ``touch()`` after each turn.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class SessionInfo:
    """Metadata for a single agent's session.

    Attributes:
        agent_id (str): The agent this record belongs to, e.g. ``"main"``.
        created_at (str): ISO 8601 UTC timestamp of when the session was first
            created, e.g. ``"2026-05-28T10:00:00+00:00"``.
        last_active (str): ISO 8601 UTC timestamp of the most recent interaction.
            Updated by :meth:`SessionStore.touch` after every turn.
        turn_count (int): Total number of completed conversation turns (user
            message → agent response) since the session was created.
    """
    agent_id: str
    created_at: str
    last_active: str
    turn_count: int


class SessionStore:
    """Stores and retrieves session metadata from a single JSON file.

    Args:
        path (Path): Full path to the ``sessions.json`` file. The file and
            its parent directories are created automatically on first write.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        # None = not yet loaded from disk; {} = loaded and empty; populated = loaded with data.
        # This distinction prevents a get_or_create() call before any _load() from
        # mistakenly thinking no sessions exist and creating duplicates.
        self._cache: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        """Return cached data, loading from disk on the first call.

        Returns:
            dict[str, dict]: Raw dict mapping agent_id → session data dict.
                Returns an empty dict if the file doesn't exist yet.
        """
        if self._cache is not None:
            return self._cache
        # First access — load from disk and populate the cache.
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt file — preserve it for post-mortem and start fresh.
            os.replace(self._path, self._path.with_suffix(".corrupt"))
            self._cache = {}
        return self._cache

    def _save(self, data: dict[str, dict]) -> None:
        """Write the sessions dict to disk and update the in-memory cache.

        Args:
            data (dict[str, dict]): The complete sessions data to persist.
                indent=2 makes the file human-readable.
        """
        # Create parent directories if they don't exist (first run).
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then replace atomically (safe on POSIX and Windows).
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)
        self._cache = data  # keep cache in sync with what was just written

    def get_or_create(self, agent_id: str) -> SessionInfo:
        """Return the session for an agent, creating it if it doesn't exist.

        If this is the first time this agent has been seen, a new session record
        is created with the current timestamp and turn_count=0.

        Args:
            agent_id (str): The agent to look up, e.g. ``"main"``.

        Returns:
            SessionInfo: The existing or newly-created session record.
        """
        data = self._load()
        if agent_id not in data:
            now = _now()
            data[agent_id] = asdict(SessionInfo(
                agent_id=agent_id,
                created_at=now,
                last_active=now,
                turn_count=0,
            ))
            self._save(data)
        return SessionInfo(**data[agent_id])

    def touch(self, agent_id: str, increment_turns: bool = False) -> SessionInfo:
        """Update the session's last-active timestamp, optionally incrementing turns.

        Called after each conversation turn to keep the metadata current.

        Args:
            agent_id (str): The agent whose session to update.
            increment_turns (bool): If ``True``, add 1 to ``turn_count``.
                Pass ``True`` after a complete user→agent exchange.

        Returns:
            SessionInfo: The updated session record.

        Notes:
            If the agent_id is not found in the store (shouldn't happen in
            normal usage, but possible if the file was manually edited),
            this falls back to ``get_or_create()`` which creates a fresh record.
        """
        data = self._load()
        if agent_id not in data:
            # Defensive fallback: agent wasn't found, create a fresh record.
            return self.get_or_create(agent_id)
        data[agent_id]["last_active"] = _now()
        if increment_turns:
            data[agent_id]["turn_count"] += 1
        self._save(data)
        return SessionInfo(**data[agent_id])

    def list_sessions(self) -> list[SessionInfo]:
        """Return all stored session records.

        Returns:
            list[SessionInfo]: One :class:`SessionInfo` per agent that has ever
                run. Returns an empty list if the sessions file doesn't exist.
                Uses the in-memory cache when available to avoid a disk read.
        """
        return [SessionInfo(**v) for v in self._load().values()]


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        str: Current time in UTC, e.g. ``"2026-05-28T12:30:00.123456+00:00"``.
            Using ``timezone.utc`` ensures timestamps are timezone-aware and
            unambiguous regardless of the server's local timezone.
    """
    return datetime.now(UTC).isoformat()
