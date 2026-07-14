"""JSONL-backed short-term conversation history per agent session.

Short-term memory stores the full back-and-forth conversation between a user
and an agent. It's "short-term" in the sense that it holds the *current
session's* messages — the running dialogue the LLM needs as context.

Why JSONL?
----------
JSONL (JSON Lines) is a file format where each line is a separate JSON object.
It's ideal here because:
- **Append-friendly**: the :meth:`append` method adds one message = writes one
  line, without reading or rewriting the rest of the file.
- **Human-readable**: you can open the file in any text editor and read each
  conversation message on its own line.
- **Crash-safe**: :meth:`save` (called after every turn) writes to a ``.tmp``
  file and then renames it into place atomically (via ``os.replace``). A crash
  mid-write cannot corrupt the existing history file — the old file stays intact
  until the new one is fully written and flushed.
- **Recoverable**: :meth:`load` silently skips any corrupt lines rather than
  aborting the entire load, so a rare filesystem glitch cannot destroy a session.

Message format
--------------
Each line in the JSONL file is an OpenAI-format message dict, e.g.:
  ``{"role": "user", "content": "What is async?"}``
  ``{"role": "assistant", "content": "Async is..."}``

File layout
-----------
Files are stored at ``{base_dir}/{agent_id}/{session_id}.jsonl``.
Each agent has its own subdirectory; each session within that agent
gets its own JSONL file identified by its UUID session ID.  Old session
files remain on disk (for potential future resume) until pruned.

Talks to
--------
- ``agents/session.py`` — loads history at session start, saves after each turn.
- ``minion.py`` — creates this, resolves session ID at startup.
"""

import json
import os
from pathlib import Path


class ShortTermMemory:
    """Conversation history store backed by per-session JSONL files.

    Each agent has a subdirectory; each session within that agent has its
    own ``{session_id}.jsonl`` file.  Old sessions remain on disk until
    :meth:`prune_sessions` removes them.

    Args:
        base_dir (Path): Root directory where ``{agent_id}/`` subdirectories
            are stored.  Created automatically if it doesn't exist.
    """

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str, session_id: str) -> Path:
        """Return the filesystem path for a given agent + session history file.

        Args:
            agent_id (str): Agent ID, e.g. ``"main"`` or ``"researcher"``.
            session_id (str): Session UUID, e.g. ``"550e8400-e29b-41d4-a716-446655440000"``.

        Returns:
            Path: Full path to the JSONL file, e.g.
                ``~/.minion-assist/sessions/main/550e8400-....jsonl``.
        """
        return self._dir / agent_id / f"{session_id}.jsonl"

    def load(self, agent_id: str, session_id: str) -> list[dict]:
        """Load the conversation history for a specific session.

        Args:
            agent_id (str): Agent ID to load history for.
            session_id (str): Session UUID to load.

        Returns:
            list[dict]: List of message dicts in chronological order.
                Returns an empty list if the file doesn't exist yet.
        """
        p = self._path(agent_id, session_id)
        if not p.exists():
            return []
        messages = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return messages

    def save(self, agent_id: str, session_id: str, messages: list[dict]) -> None:
        """Overwrite the session history file with the given message list.

        Uses a temp-file swap (write to .tmp then os.replace) so a crash
        mid-write cannot corrupt or truncate an existing history file.

        Args:
            agent_id (str): Agent ID.
            session_id (str): Session UUID.
            messages (list[dict]): The complete message list to save.
        """
        p = self._path(agent_id, session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            "\n".join(json.dumps(m) for m in messages),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    def append(self, agent_id: str, session_id: str, message: dict) -> None:
        """Append a single message to the session history file.

        More efficient than ``save()`` for adding one message at a time.

        Args:
            agent_id (str): Agent ID.
            session_id (str): Session UUID.
            message (dict): The message dict to append.
        """
        p = self._path(agent_id, session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

    def clear(self, agent_id: str, session_id: str) -> None:
        """Delete the history file for a specific session.

        Args:
            agent_id (str): Agent ID.
            session_id (str): Session UUID whose history file should be removed.
                Does nothing if the file doesn't exist.
        """
        p = self._path(agent_id, session_id)
        if p.exists():
            p.unlink()

    def list_sessions(self, agent_id: str) -> list[Path]:
        """Return all session JSONL files for an agent, sorted oldest-first.

        Args:
            agent_id (str): Agent ID.

        Returns:
            list[Path]: Session file paths sorted by modification time (oldest first).
                Returns an empty list when no sessions exist yet.
        """
        agent_dir = self._dir / agent_id
        if not agent_dir.exists():
            return []
        files = sorted(agent_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        return files

    def _name_path(self, agent_id: str, session_id: str) -> Path:
        """Return the path of the sidecar name file for a session.

        Names are stored as tiny plain-text files next to their JSONL counterpart,
        e.g. ``sessions/main/abc-123.name`` alongside ``sessions/main/abc-123.jsonl``.
        This keeps the name co-located with the history it describes and means
        ``list_sessions()`` (which globs ``*.jsonl``) never accidentally picks them up.
        """
        return self._dir / agent_id / f"{session_id}.name"

    def get_name(self, agent_id: str, session_id: str) -> str | None:
        """Return the human-readable name for a session, or ``None`` if unset.

        Args:
            agent_id (str): Agent ID.
            session_id (str): Session UUID.

        Returns:
            str | None: The stored name, or ``None`` if the session has no name.
        """
        p = self._name_path(agent_id, session_id)
        if not p.exists():
            return None
        # Strip whitespace so an accidental trailing newline doesn't count as a name.
        return p.read_text(encoding="utf-8").strip() or None

    def set_name(self, agent_id: str, session_id: str, name: str) -> None:
        """Assign a human-readable name to a session.

        Stored as a sidecar ``.name`` file alongside the ``.jsonl`` history file.
        Passing an empty string clears the name (removes the file).

        Args:
            agent_id (str): Agent ID.
            session_id (str): Session UUID to rename.
            name (str): The new display name.  Leading/trailing whitespace is stripped.
        """
        p = self._name_path(agent_id, session_id)
        name = name.strip()
        if not name:
            # Empty name = clear the name — delete the sidecar file if it exists.
            if p.exists():
                p.unlink()
            return
        # Ensure the agent directory exists before writing (it may not exist yet
        # if this is the very first session for this agent).
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding="utf-8")

    def delete_session(self, agent_id: str, session_id: str) -> bool:
        """Delete a session's history file and its sidecar name file.

        Does not touch ``sessions.json`` — that file only tracks the current
        session pointer per agent and is managed by :class:`SessionStore`.

        Args:
            agent_id (str): Agent ID.
            session_id (str): UUID of the session to delete.

        Returns:
            bool: ``True`` if the ``.jsonl`` file existed and was deleted,
                ``False`` if the file was not found (already gone).
        """
        p = self._path(agent_id, session_id)
        name_p = self._name_path(agent_id, session_id)
        deleted = False
        if p.exists():
            p.unlink()
            deleted = True
        # Remove the sidecar name file regardless — it's meaningless without history.
        if name_p.exists():
            name_p.unlink()
        return deleted

    def prune_sessions(self, agent_id: str, keep_n: int = 20) -> int:
        """Delete old session files, keeping only the N most recent.

        Args:
            agent_id (str): Agent ID whose old sessions to prune.
            keep_n (int): Number of most-recent session files to keep. Default 20.

        Returns:
            int: Number of files deleted.
        """
        files = self.list_sessions(agent_id)  # sorted oldest → newest
        to_delete = files[:-keep_n] if len(files) > keep_n else []
        for f in to_delete:
            try:
                f.unlink()
            except OSError:
                pass
        return len(to_delete)
