"""JSONL-backed short-term conversation history per agent.

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
Files are stored at ``{base_dir}/{key}.jsonl``. The key is typically the agent
ID (e.g. ``"main"`` → ``~/.minion-assistant/sessions/main.jsonl``).

Talks to
--------
- ``minion.py`` — creates this, loads history at startup, saves after each turn.
- ``tools/memory.py`` — the long-term counterpart; this handles short-term only.
"""

import json
import os
from pathlib import Path


class ShortTermMemory:
    """Conversation history store backed by JSONL files.

    Args:
        base_dir (Path): Directory where ``.jsonl`` history files are stored.
            Created automatically if it doesn't exist.
    """

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        # Create the sessions directory if it doesn't exist.
        # parents=True handles the case where ~/.minion-assistant itself doesn't exist yet.
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Return the filesystem path for a given agent's history file.

        Args:
            key (str): Agent ID, e.g. ``"main"`` or ``"researcher"``.

        Returns:
            Path: Full path to the JSONL file, e.g. ``/home/user/.minion-assistant/sessions/main.jsonl``.
        """
        return self._dir / f"{key}.jsonl"

    def load(self, key: str) -> list[dict]:
        """Load the full conversation history for an agent.

        Reads the JSONL file and parses each line as a message dict.
        Skips blank lines to handle trailing newlines gracefully.

        Args:
            key (str): Agent ID to load history for.

        Returns:
            list[dict]: List of message dicts in chronological order.
                Returns an empty list if the file doesn't exist yet
                (first run / fresh start).
        """
        p = self._path(key)
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
                # Skip corrupt lines; the rest of the history is still usable.
                pass
        return messages

    def save(self, key: str, messages: list[dict]) -> None:
        """Overwrite the history file with the given message list.

        This does a full rewrite of the file. Used after each conversation turn
        to persist the complete updated history.

        Uses a temp-file swap (write to .tmp then os.replace) so a crash
        mid-write cannot corrupt or truncate an existing history file.

        Args:
            key (str): Agent ID.
            messages (list[dict]): The complete message list to save.
                Serializes each message as one JSON line.
        """
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            "\n".join(json.dumps(m) for m in messages),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    def append(self, key: str, message: dict) -> None:
        """Append a single message to the history file without reading the rest.

        More efficient than ``save()`` for adding one message at a time, because
        it opens the file in append mode (``"a"``) rather than rewriting
        everything. Useful for very long conversations.

        Args:
            key (str): Agent ID.
            message (dict): The message dict to append.
        """
        with self._path(key).open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

    def clear(self, key: str) -> None:
        """Delete the history file for an agent, resetting their conversation.

        Args:
            key (str): Agent ID whose history file should be removed.
                Does nothing if the file doesn't exist.
        """
        p = self._path(key)
        if p.exists():
            p.unlink()  # unlink() is Python's name for "delete a file"
