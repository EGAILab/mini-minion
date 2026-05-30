"""Markdown-file-backed long-term memory store.

Long-term memory is for knowledge the agent intentionally chooses to persist —
notes, research findings, project decisions, or anything worth remembering
across multiple sessions. Unlike short-term memory (the conversation transcript),
long-term memory is selective: the agent explicitly calls the ``save_memory``
tool to write a note.

Why Markdown files?
--------------------
- **Human-readable**: you can open the memory folder and read what the agent
  has saved, edit notes, or delete irrelevant ones.
- **Simple**: no database, no indexing — just files. The tradeoff is that
  ``search()`` does a linear scan of all files, which is fine for a small number
  of notes but would be slow with thousands.
- **LLM-friendly**: Markdown is easy for the model to write and read back.

File layout
-----------
Each note is stored as ``{base_dir}/{key}.md``. Keys with forward slashes or
backslashes are sanitized (replaced with ``_``) to be filesystem-safe.

Example: key ``"api/rest-notes"`` → file ``api_rest-notes.md``.

Talks to
--------
- ``tools/memory.py`` — :class:`SaveMemoryTool` and :class:`SearchMemoryTool`
  call methods on this class to read and write notes.
- ``minion.py`` — creates the :class:`LongTermMemory` instance at startup.
"""

from pathlib import Path

# Linear scan cap — prevents context flooding when many notes match a broad query.
# Fine for dozens of notes; raise or add an index if the store grows beyond ~1 000.
_SEARCH_MAX_RESULTS = 20


class LongTermMemory:
    """Long-term note store backed by Markdown files on disk.

    Args:
        base_dir (Path): Directory where Markdown note files are stored.
            Created automatically if it doesn't exist.
    """

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        # Create the memory directory if it doesn't exist.
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Build a filesystem-safe path for a given memory key.

        Args:
            key (str): The note identifier, e.g. ``"project-goals"`` or
                ``"api/rest-research"``.

        Returns:
            Path: Full path to the Markdown file. Slashes in the key are
                replaced with underscores to keep a flat directory structure.
        """
        # Replace path separators with underscores to keep all files in a flat
        # structure and avoid accidentally creating subdirectories.
        safe = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.md"

    def save(self, key: str, content: str) -> None:
        """Save a note to memory, overwriting any existing note with the same key.

        Args:
            key (str): Identifier for this note, e.g. ``"project-goals"``.
            content (str): Markdown text to store.
        """
        self._path(key).write_text(content, encoding="utf-8")

    def load(self, key: str) -> str | None:
        """Load the content of a memory note by key.

        Args:
            key (str): The note identifier.

        Returns:
            str | None: The note's text content, or ``None`` if no note with
                this key exists.
        """
        p = self._path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def search(self, query: str, max_results: int = _SEARCH_MAX_RESULTS) -> list[tuple[str, str]]:
        """Search all memory notes for any term in the query.

        Splits ``query`` on whitespace and returns notes where at least one
        term appears in the content or key. Case-insensitive. Capped at
        ``max_results`` to prevent flooding the context window.

        Args:
            query (str): One or more keywords to search for (case-insensitive).
            max_results (int): Maximum notes to return. Defaults to
                :data:`_SEARCH_MAX_RESULTS`.

        Returns:
            list[tuple[str, str]]: List of ``(key, content)`` pairs for
                matching notes, sorted alphabetically by key.
        """
        results = []
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return results
        for p in sorted(self._dir.glob("*.md")):
            content = p.read_text(encoding="utf-8")
            content_lower = content.lower()
            stem_lower = p.stem.lower()
            if any(t in content_lower or t in stem_lower for t in terms):
                results.append((p.stem, content))
                if len(results) >= max_results:
                    break
        return results

    def list_keys(self) -> list[str]:
        """Return a sorted list of all stored note keys.

        Returns:
            list[str]: All note identifiers (filenames without ``.md`` extension),
                sorted alphabetically.
        """
        return [p.stem for p in sorted(self._dir.glob("*.md"))]

    def delete(self, key: str) -> bool:
        """Delete a memory note.

        Args:
            key (str): The note identifier to delete.

        Returns:
            bool: ``True`` if the file was deleted, ``False`` if it didn't exist.
        """
        p = self._path(key)
        if p.exists():
            p.unlink()
            return True
        return False
