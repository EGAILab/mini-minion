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

        # Shadow copies of both tables above, same shape — scratch space for
        # force_rebuild_agent()'s crash-safe rebuild-and-swap (Stage One
        # Phase 3, slice C). See that method's docstring for the swap
        # mechanics. Never queried by search()/chunk_count()/etc. — only
        # force_rebuild_agent() ever touches these.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks_shadow (
                id            BIGSERIAL PRIMARY KEY,
                agent_id      TEXT NOT NULL,
                source_kind   TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                heading_path  TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                chunk_hash    TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_chunks_shadow_agent_idx "
            "ON memory_chunks_shadow (agent_id)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_files_shadow (
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                source_kind  TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at   DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_files_shadow_agent_idx "
            "ON memory_files_shadow (agent_id)"
        )

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
        file in place) or :meth:`force_rebuild_agent` (which does the same
        but crash-safely, via a shadow-table swap), this only reindexes a
        file whose content hash differs from what's already in the
        ``memory_files`` ledger, and only
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

    def force_rebuild_agent(
        self, agent_id: str, indexable_files: list[tuple[str, str, str]]
    ) -> int:
        """Crash-safely rebuild one agent's entire index, atomically swapping it into service.

        Unlike :meth:`rebuild_agent` (which deletes and reinserts each
        file's chunks one at a time, so a crash partway through can leave
        an agent with a genuinely incomplete index — some files reindexed,
        some not), this builds the *entire* new index into a scratch area
        first and only replaces the live tables in a single atomic
        transaction:

        1. Clear this agent's rows from ``memory_chunks_shadow`` /
           ``memory_files_shadow`` (leftover scratch from a previous
           *interrupted* force-rebuild, if any — safe to discard, it was
           never live).
        2. Chunk and write every file in ``indexable_files`` into the
           shadow tables. If this raises partway through (or the process
           crashes), the live ``memory_chunks``/``memory_files`` tables
           are never touched at all — search results are completely
           unaffected by an interrupted rebuild.
        3. Validate that every file was actually processed (a simple
           count check — see the "processed" counter below) before
           proceeding to the swap.
        4. Swap: in one PostgreSQL transaction, delete this agent's live
           rows and copy the shadow rows into their place, then clear the
           shadow rows. PostgreSQL's own transactional guarantees mean
           this is atomic even if the process crashes mid-swap — either
           the whole transaction commits (search immediately sees the new
           index) or none of it does (search keeps seeing the old index,
           unchanged) - there is never a moment where search sees a
           half-old-half-new or empty result set.

        Args:
            agent_id: The agent to rebuild.
            indexable_files: ``(source_kind, rel_path, content)`` triples —
                typically :meth:`MemoryFileRepository.list_indexable_files`'s
                current return value.

        Returns:
            int: Total chunk count written across all files (post-swap).

        Note:
            Single-writer assumption: running two ``force_rebuild_agent``
            calls for the *same* agent concurrently is not safe (both would
            share the same shadow scratch rows). This is a rare, manual
            maintenance operation (``minion-assist memory reindex --force``),
            not something the running app ever triggers concurrently with
            itself, so this is an accepted, documented limitation rather
            than something guarded against with extra locking — the same
            kind of concurrency assumption ``session/db.py``'s
            ``mirror_message`` documents for its own check-then-insert.
        """
        conn = self._conn()
        conn.execute("DELETE FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,))
        conn.execute("DELETE FROM memory_files_shadow WHERE agent_id = %s", (agent_id,))

        processed = 0
        for source_kind, rel_path, content in indexable_files:
            chunks = chunk_markdown(content)
            for i, chunk in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO memory_chunks_shadow
                        (agent_id, source_kind, rel_path, chunk_index, heading_path,
                         content, start_line, end_line, chunk_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        agent_id, source_kind, rel_path, i, " > ".join(chunk.heading_path),
                        chunk.content, chunk.start_line, chunk.end_line,
                        _hash_text(chunk.content),
                    ),
                )
            conn.execute(
                """
                INSERT INTO memory_files_shadow
                    (agent_id, rel_path, source_kind, content_hash, indexed_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (agent_id, rel_path, source_kind, _hash_text(content), time.time()),
            )
            processed += 1

        if processed != len(indexable_files):
            raise RuntimeError(
                f"force_rebuild_agent: processed {processed} of {len(indexable_files)} "
                f"files for agent {agent_id!r} — aborting swap, live index left unchanged."
            )

        with conn.transaction():
            conn.execute("DELETE FROM memory_chunks WHERE agent_id = %s", (agent_id,))
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                SELECT agent_id, source_kind, rel_path, chunk_index, heading_path,
                       content, start_line, end_line, chunk_hash
                FROM memory_chunks_shadow WHERE agent_id = %s
                """,
                (agent_id,),
            )
            conn.execute("DELETE FROM memory_files WHERE agent_id = %s", (agent_id,))
            conn.execute(
                """
                INSERT INTO memory_files (agent_id, rel_path, source_kind, content_hash, indexed_at)
                SELECT agent_id, rel_path, source_kind, content_hash, indexed_at
                FROM memory_files_shadow WHERE agent_id = %s
                """,
                (agent_id,),
            )

        conn.execute("DELETE FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,))
        conn.execute("DELETE FROM memory_files_shadow WHERE agent_id = %s", (agent_id,))

        return self.chunk_count(agent_id)

    # ------------------------------------------------------------------
    # Search — Stage One Phase 3, slice C
    # ------------------------------------------------------------------

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        corpus: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Full-text search across one agent's indexed chunks, ranked by relevance.

        Args:
            agent_id: Which agent's chunks to search.
            query: Free-text query, passed through PostgreSQL's
                ``websearch_to_tsquery`` — quoted phrases, ``AND``/``OR``,
                and ``-exclude`` all work, the same query language
                ``session/db.py``'s ``search_messages`` already uses for
                message search.
            corpus: Restrict results to one ``source_kind`` (``"durable"``,
                ``"daily"``, or ``"import"``), or ``None`` to search every
                corpus. The plan's fourth corpus, "sessions", is
                deliberately not offered here — that's already the
                separate ``session_search`` tool's job; duplicating it into
                this index would mean two divergent ways to search the same
                message data.
            max_results: Maximum chunks to return.

        Returns:
            list[dict]: Best matches first. Each has ``rel_path``,
                ``source_kind``, ``chunk_index``, ``heading_path``,
                ``content``, ``start_line``, ``end_line``, and ``score``
                (the ``ts_rank`` value) — everything
                :class:`~minion_assist.memory.models.MemoryHit`'s optional
                citation fields need.
        """
        conn = self._conn()
        corpus_sql = " AND source_kind = %s" if corpus else ""
        params: list = [query, agent_id, query]
        if corpus:
            params.append(corpus)
        params.append(max_results)

        rows = conn.execute(
            f"""
            SELECT rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line,
                   ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS score
            FROM memory_chunks
            WHERE agent_id = %s
              AND search_vector @@ websearch_to_tsquery('english', %s)
              {corpus_sql}
            ORDER BY score DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "rel_path": r[0], "source_kind": r[1], "chunk_index": r[2],
                "heading_path": r[3], "content": r[4], "start_line": r[5],
                "end_line": r[6], "score": float(r[7]),
            }
            for r in rows
        ]

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

    def index_summary(self, agent_id: str) -> dict:
        """A snapshot of one agent's index health — the primitive behind ``memory status --deep``.

        Everything here is derived from tables a one-shot CLI process can
        actually read; it deliberately does *not* claim to know whether a
        live ``CaptureWorker``/``MemoryIndexWatcher`` thread is currently
        running in some other process — that's not observable from here.

        Returns:
            dict: ``total_chunks``, ``file_count``, ``by_corpus`` (chunk
                count per ``source_kind``), and ``last_indexed_at`` (the
                most recent ``memory_files.indexed_at`` — ``None`` if this
                agent has no indexed files at all).
        """
        conn = self._conn()
        total_chunks = self.chunk_count(agent_id)
        file_count = len(self.indexed_files(agent_id))

        by_corpus_rows = conn.execute(
            "SELECT source_kind, count(*) FROM memory_chunks "
            "WHERE agent_id = %s GROUP BY source_kind",
            (agent_id,),
        ).fetchall()
        by_corpus = {kind: count for kind, count in by_corpus_rows}

        last_indexed_row = conn.execute(
            "SELECT max(indexed_at) FROM memory_files WHERE agent_id = %s", (agent_id,)
        ).fetchone()
        last_indexed_at = last_indexed_row[0] if last_indexed_row else None

        return {
            "total_chunks": total_chunks,
            "file_count": file_count,
            "by_corpus": by_corpus,
            "last_indexed_at": last_indexed_at,
        }
