"""``MemoryService`` — the one object agents/tools depend on for memory.

Stage One Phase 1, slice 2.

Why a facade over ``MemoryFileRepository``?
--------------------------------------------
Today, ``AgentSession`` and every memory tool (``SaveMemoryTool``,
``NoteTool``, ``SearchMemoryTool``, ``WriteDailyMemoryTool``) each hold their
own reference to a raw :class:`~minion_assist.memory.long_term.LongTermMemory`
instance and call its methods directly. That means every consumer has to
know the storage layer's internals (how a key becomes a filename, where
daily notes live) and there is no single place to add cross-cutting
behavior (status reporting, future scope checks) without touching every
call site.

``MemoryService`` is that single place. It wraps one
:class:`~minion_assist.memory.files.MemoryFileRepository` per agent and
exposes the operations tools/session code actually need, translating
``files.py``'s lower-level primitives (path resolution, locators) into
simpler calls a tool can use directly — e.g. :meth:`get` takes a plain path
string rather than requiring the caller to build a
:class:`~minion_assist.memory.models.MemoryLocator` itself.

Scope, deliberately not enforced here
--------------------------------------
Stage One's target design names several memory scopes (agent-private,
user-shared, workspace, session-lineage, channel, import-quarantine — see
``docs/adr/0003-per-agent-memory-scope.md``). This service only enforces
``agent-private``, and it does so *structurally*: one ``MemoryService`` per
agent, each backed by its own repository rooted at that agent's own
workspace directory. There is no runtime scope-check method here, because
nothing in the codebase yet produces a request tagged with any other scope
— adding that plumbing now would be exactly the kind of speculative,
no-current-caller abstraction the project's simplicity rule warns against.
Add real scope enforcement when a real cross-agent/channel use case exists.

Not yet wired in
-----------------
This is slice 2 of Phase 1 — ``AgentSession`` and the tools still use
``LongTermMemory`` directly. Wiring happens in slice 3.

Talks to
--------
- ``memory/files.py`` — :class:`MemoryFileRepository`, the actual storage
  backend this service wraps.
- ``memory/models.py`` — the typed request/result objects passed through.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .files import MemoryFileRepository
from .models import MemoryExcerpt, MemoryHit, MemoryLocator, MemoryStatus

# Mirrors MemoryFileRepository's / LongTermMemory's default search cap.
_SEARCH_MAX_RESULTS = 20


class MemoryService:
    """Thin orchestration facade over :class:`MemoryFileRepository`.

    Args:
        files: The repository backing this service. Construct one
            ``MemoryService`` per agent, each wrapping a repository rooted
            at that agent's own workspace directory.
    """

    def __init__(self, files: MemoryFileRepository) -> None:
        self._files = files

    # -----------------------------------------------------------------
    # Explicit notes
    # -----------------------------------------------------------------

    def remember(self, key: str, content: str) -> None:
        """Save an explicit note under ``key``, overwriting any existing note.

        Args:
            key: Note identifier, e.g. ``"project-goals"``.
            content: Markdown text to store.
        """
        self._files.remember(key, content)

    def load(self, key: str) -> str | None:
        """Load an explicit note's content by key, or ``None`` if it doesn't exist."""
        return self._files.load(key)

    def delete(self, key: str) -> bool:
        """Delete an explicit note. Returns ``True`` if a file was removed."""
        return self._files.delete(key)

    def list_keys(self) -> list[str]:
        """Return every explicit note's key, sorted alphabetically."""
        return self._files.list_keys()

    # -----------------------------------------------------------------
    # Quarantined notes — unreviewed, never auto-promoted (see files.py)
    # -----------------------------------------------------------------

    def remember_import(self, key: str, content: str) -> None:
        """Save quarantined, unreviewed content under ``key``.

        Used by the background extractor and (until retired) the ``note``
        tool — content nobody has reviewed yet, searchable but never
        auto-promoted. See ``docs/adr/0003-per-agent-memory-scope.md``.
        """
        self._files.remember_import(key, content)

    def load_import(self, key: str) -> str | None:
        """Load quarantined content by key, or ``None`` if it doesn't exist."""
        return self._files.load_import(key)

    def list_import_keys(self) -> list[str]:
        """Return every quarantined note's key, sorted alphabetically."""
        return self._files.list_import_keys()

    # -----------------------------------------------------------------
    # Search and recall
    # -----------------------------------------------------------------

    def search(self, query: str, max_results: int = _SEARCH_MAX_RESULTS) -> list[MemoryHit]:
        """Search topic, import, and daily notes for ``query``.

        Returns raw, structured hits — formatting them into a prompt block
        or tool-result string is the caller's job (``session.py`` for
        per-turn injection, ``SearchMemoryTool`` for the explicit
        ``search_memory`` tool call), matching how ``LongTermMemory.search()``
        results are used today. See the module docstring's "Scope,
        deliberately not enforced here" note — this is also where a future
        recall-telemetry hook (Stage One Phase 5) would attach, once one
        exists.

        Args:
            query: One or more keywords, space-separated.
            max_results: Maximum notes to return.

        Returns:
            list[MemoryHit]: Best matches first, tagged by source.
        """
        return self._files.search(query, max_results=max_results)

    # -----------------------------------------------------------------
    # Daily notes
    # -----------------------------------------------------------------

    def append_daily(self, text: str, *, when: date | None = None) -> Path:
        """Append a timestamped entry to today's (or ``when``'s) daily note.

        Args:
            text: The note content to append.
            when: Date to file this entry under. Defaults to today.

        Returns:
            Path: The daily note file that was written to.
        """
        return self._files.append_daily(text, when=when)

    # -----------------------------------------------------------------
    # Exact bounded reads
    # -----------------------------------------------------------------

    def get(
        self,
        path: str,
        *,
        from_line: int | None = None,
        lines: int | None = None,
    ) -> MemoryExcerpt:
        """Read an exact, bounded slice of a memory file by path string.

        Wraps :meth:`MemoryFileRepository.resolve_path` and
        :class:`MemoryLocator` construction so callers never need to know
        about the locator type or perform their own containment check —
        this is what a future ``memory_get`` tool (Phase 1, slice 5) will
        call directly with the path string an LLM provides.

        Args:
            path: A relative (to the agent's workspace root) or absolute
                path. Resolved and validated as inside the memory root.
            from_line: 1-indexed starting line. ``None`` means from the
                start of the file.
            lines: Maximum number of lines to return. ``None`` means to the
                end of the file.

        Returns:
            MemoryExcerpt: The requested slice, plus the file's total line
                count.

        Raises:
            ValueError: ``path`` resolves outside the memory root.
            FileNotFoundError: ``path`` does not exist.
        """
        resolved = self._files.resolve_path(path)
        locator = MemoryLocator(path=resolved, from_line=from_line, lines=lines)
        return self._files.get(locator)

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def status(self) -> MemoryStatus:
        """Return a snapshot of this agent's memory store — note counts by source.

        Returns:
            MemoryStatus: Topic/import/daily note counts and the store's root
                directory. See :class:`MemoryStatus` for why this is
                intentionally minimal in Phase 1.
        """
        counts = self._files.count_notes()
        return MemoryStatus(
            root=self._files.root,
            topic_count=counts["topic"],
            import_count=counts["import"],
            daily_count=counts["daily"],
        )
