"""``PostgresMemoryIndex`` — rebuildable lexical index over memory files.

Stage One Phase 3, slice A.

What this is, and why it's separate from ``session/db.py``
------------------------------------------------------------
``session/db.py``'s ``SessionDB`` already runs PostgreSQL full-text search —
but over *chat messages* (``messages.search_vector``), for the
``session_search`` tool. This module indexes something different: the
curated Markdown *memory files* a ``MemoryFileRepository`` owns (``MEMORY.md``,
topic notes, daily notes, imports). Those are a different corpus with a
different lifecycle (edited occasionally, not appended every turn), so this
gets its own tables and its own class rather than growing ``SessionDB``
further — matching the module split the plan lays out
(``minion-assist-docs/improve/memory-implementation-plan.md``'s "Stage One
proposed module changes").

Why chunk instead of indexing whole files?
--------------------------------------------
See ``memory/chunking.py``'s module docstring — the short version is that a
citation needs to point at a specific section once a file (typically
``MEMORY.md``) grows long, not just "somewhere in this file."

What this slice does *not* yet do
------------------------------------
This class only builds and rebuilds the index — nothing in the running app
calls it yet. Wiring (write-path sync, startup reconciliation, a live
filesystem watcher) is Stage One Phase 3 slice B; crash-safe rebuild-and-swap,
corpus-filtered search, and citations wired into ``MemoryService.search()``
are slice C. Building this in isolation first means the chunking and schema
logic can be tested against a real PostgreSQL instance without needing the
rest of the machinery to exist yet.

Talks to
--------
- ``memory/chunking.py`` — :func:`chunk_markdown`, this module's only
  chunking dependency.
- ``memory/files.py`` — :meth:`MemoryFileRepository.list_indexable_files`
  supplies the ``(source_kind, rel_path, content)`` triples
  :meth:`PostgresMemoryIndex.rebuild_agent` indexes.
"""

from __future__ import annotations

import hashlib
import threading
import time

from .chunking import chunk_markdown

# Thread-local connection cache, kept separate from session/db.py's own
# thread-local so a psycopg connection meant for one module is never
# accidentally reused by the other.
_local = threading.local()


def _hash_text(text: str) -> str:
    """SHA256 hex digest of a file's content — used to detect real changes.

    Mirrors ``memory/migration.py``'s ``_hash_bytes`` (same algorithm, same
    reasoning: a cheap, reliable way to tell "this file's content actually
    changed" apart from "this file's mtime changed for some other reason").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PostgresMemoryIndex:
    """Rebuildable PostgreSQL lexical index over one or more agents' memory files.

    Args:
        url: libpq connection string — the same ``database.url`` config
            value :class:`~minion_assist.session.db.SessionDB` uses. Opening
            a second connection to the same database (rather than sharing
            ``SessionDB``'s) keeps this module fully independent, at the
            cost of one extra idle connection per thread — a deliberate,
            simple trade given how rarely indexing operations happen
            relative to message traffic.
    """

    def __init__(self, url: str) -> None:
        import psycopg  # noqa: PLC0415

        self._url = url
        self._psycopg = psycopg
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return a thread-local autocommit connection, creating it if needed."""
        conn = getattr(_local, "conn", None)
        if conn is None or conn.closed:
            conn = self._psycopg.connect(self._url, autocommit=True)
            _local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                source_kind   TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                heading_path  TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                chunk_hash    TEXT NOT NULL,
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(content, '')), 'B')
                ) STORED
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_fts_idx "
            "ON memory_chunks USING GIN (search_vector)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_file_idx "
            "ON memory_chunks (agent_id, rel_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_corpus_idx "
            "ON memory_chunks (agent_id, source_kind)"
        )

        # Per-file reconciliation ledger — same role as session/db.py's
        # message_mirrors: lets a later slice B diff "what's on disk" against
        # "what's indexed" by content hash rather than reindexing everything
        # unconditionally on every startup.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_files (
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                source_kind  TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at   DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (agent_id, rel_path)
            )
        """)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def reindex_file(self, agent_id: str, rel_path: str, source_kind: str, content: str) -> int:
        """(Re)index one file: replace its chunks and update the reconciliation ledger.

        Always deletes any existing chunks for this ``(agent_id, rel_path)``
        first, then inserts fresh ones — simpler and safer than trying to
        diff old vs. new chunk boundaries, and cheap enough given these are
        curated notes, not a high-frequency write path.

        Args:
            agent_id: The owning agent.
            rel_path: Path relative to the agent's workspace root (as
                returned by :meth:`MemoryFileRepository.list_indexable_files`).
            source_kind: ``"durable"``, ``"daily"``, or ``"import"``.
            content: The file's current full text.

        Returns:
            int: How many chunks were written (0 for empty/whitespace-only
                content — the file's ledger row is still updated so a
                since-emptied file doesn't linger as stale chunks).
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

        chunks = chunk_markdown(content)
        for i, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    agent_id,
                    source_kind,
                    rel_path,
                    i,
                    " > ".join(chunk.heading_path),
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    _hash_text(chunk.content),
                ),
            )

        conn.execute(
            """
            INSERT INTO memory_files (agent_id, rel_path, source_kind, content_hash, indexed_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (agent_id, rel_path)
            DO UPDATE SET source_kind = EXCLUDED.source_kind,
                          content_hash = EXCLUDED.content_hash,
                          indexed_at = EXCLUDED.indexed_at
            """,
            (agent_id, rel_path, source_kind, _hash_text(content), time.time()),
        )
        return len(chunks)

    def remove_file(self, agent_id: str, rel_path: str) -> None:
        """Remove one file's chunks and ledger row (e.g. after on-disk deletion).

        A no-op if this file was never indexed — safe to call unconditionally.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        conn.execute(
            "DELETE FROM memory_files WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

    def rebuild_agent(self, agent_id: str, indexable_files: list[tuple[str, str, str]]) -> int:
        """Rebuild one agent's entire index from a fresh file listing.

        Removes ledger/chunk rows for any file no longer present in
        ``indexable_files`` (i.e. deleted from disk since the last build),
        then reindexes every file in the listing.

        Args:
            agent_id: The agent to rebuild.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                return value.

        Returns:
            int: Total chunk count written across all files.
        """
        conn = self._conn()
        current_paths = {rel_path for _kind, rel_path, _content in indexable_files}

        existing_rows = conn.execute(
            "SELECT rel_path FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        for (rel_path,) in existing_rows:
            if rel_path not in current_paths:
                self.remove_file(agent_id, rel_path)

        total = 0
        for source_kind, rel_path, content in indexable_files:
            total += self.reindex_file(agent_id, rel_path, source_kind, content)
        return total

    def reconcile_agent(self, agent_id: str, indexable_files: list[tuple[str, str, str]]) -> int:
        """Bring one agent's index up to date by content hash, touching only what changed.

        Unlike :meth:`rebuild_agent` (which unconditionally reindexes every
        file — the primitive Phase 3 slice C's crash-safe force-reindex
        builds on), this only reindexes a file whose content hash differs
        from what's already in the ``memory_files`` ledger, and only
        removes ledger/chunk rows for files no longer present. A file
        that's unchanged since the last reconciliation costs one hash
        comparison, not a full rechunk-and-reinsert.

        This is what startup catch-up, ``MemoryService``'s write-path sync,
        and the live filesystem watcher (Phase 3 slice B) all call — the
        same "diff by hash, heal exactly what's missing or stale" shape as
        ``session/db.py``'s ``reconcile_session``/``reconcile_all_sessions``
        for message mirrors.

        Args:
            agent_id: The agent to reconcile.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                current return value.

        Returns:
            int: How many files were actually reindexed or removed (0 means
                the index was already fully up to date).
        """
        conn = self._conn()
        current = {
            rel_path: (source_kind, content) for source_kind, rel_path, content in indexable_files
        }

        existing_rows = conn.execute(
            "SELECT rel_path, content_hash FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        existing_hashes = {rel_path: content_hash for rel_path, content_hash in existing_rows}

        touched = 0
        for rel_path in existing_hashes:
            if rel_path not in current:
                self.remove_file(agent_id, rel_path)
                touched += 1

        for rel_path, (source_kind, content) in current.items():
            if existing_hashes.get(rel_path) != _hash_text(content):
                self.reindex_file(agent_id, rel_path, source_kind, content)
                touched += 1

        return touched

    # ------------------------------------------------------------------
    # Inspection (used by tests and later slices' status reporting)
    # ------------------------------------------------------------------

    def chunk_count(self, agent_id: str) -> int:
        """Total indexed chunk count for one agent — a primitive for tests/status."""
        row = self._conn().execute(
            "SELECT count(*) FROM memory_chunks WHERE agent_id = %s", (agent_id,)
        ).fetchone()
        return row[0] if row else 0

    def indexed_files(self, agent_id: str) -> list[str]:
        """Every ``rel_path`` currently in the reconciliation ledger for one agent."""
        rows = self._conn().execute(
            "SELECT rel_path FROM memory_files WHERE agent_id = %s ORDER BY rel_path", (agent_id,)
        ).fetchall()
        return [r[0] for r in rows]
