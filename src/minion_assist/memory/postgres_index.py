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
import logging
import re
import threading
import time
from datetime import date as _date
from typing import TYPE_CHECKING

from .chunking import Chunk, chunk_markdown

if TYPE_CHECKING:
    from ..providers.embeddings import EmbeddingProvider

_log = logging.getLogger("minion_assist.memory_index")

# Thread-local connection cache, kept separate from session/db.py's own
# thread-local so a psycopg connection meant for one module is never
# accidentally reused by the other.
_local = threading.local()

# ---------------------------------------------------------------------------
# Fusion helpers — Stage One Phase 4, slice C
# ---------------------------------------------------------------------------
# Free functions (not methods) since none of them touch the database — they
# operate purely on the lane results hybrid_search() already fetched.

# Standard reciprocal-rank-fusion constant. 60 is the value from the
# original RRF paper (Cormack et al.) and is not sensitive to tuning for
# typical result-list lengths — it just controls how quickly a lower rank's
# contribution falls off, not the fusion's correctness.
_RRF_K = 60

# Half-life for daily-note score decay: a daily note's fused score is
# multiplied by 0.5 for every _DECAY_HALF_LIFE_DAYS days since the date in
# its filename. Only source_kind "daily" decays — "durable" content
# (MEMORY.md, topic notes) does not, per Task 6's "evergreen MEMORY.md and
# topic pages do not decay merely because their file mtime is old."
_DECAY_HALF_LIFE_DAYS = 30

_DAILY_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")


def _reciprocal_rank_fusion(lanes: list[list[dict]]) -> tuple[dict[int, float], dict[int, dict]]:
    """Combine several independently-ranked lanes into one score per chunk id.

    Each lane is a list of chunk dicts already ranked best-first (by
    whatever criterion that lane uses — ts_rank, cosine similarity, recency,
    exact-path match). A chunk's fused score is the sum, across every lane
    it appears in, of ``1 / (_RRF_K + rank)`` — so a chunk ranked highly by
    even one lane scores well, and a chunk multiple lanes agree on scores
    higher still. This is why every lane must be able to surface a
    candidate the others missed (the mem0 "reject semantic-seeded fusion"
    decision the plan cites): a chunk absent from a lane simply doesn't
    collect that lane's contribution, rather than being penalized for it.

    Args:
        lanes: One ranked list per lane. A lane may be empty (e.g. the
            vector lane with no embedding provider configured).

    Returns:
        tuple[dict[int, float], dict[int, dict]]: ``(scores, rows)`` —
            ``scores`` maps chunk id to its fused score, ``rows`` maps
            chunk id to the first lane's dict that mentioned it (all lanes
            describe the same underlying chunk, so any one's fields work).
    """
    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}
    for lane in lanes:
        for rank, row in enumerate(lane, start=1):
            chunk_id = row["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            rows.setdefault(chunk_id, row)
    return scores, rows


def _decay_factor(rel_path: str, source_kind: str) -> float:
    """Score multiplier for temporal decay — 1.0 (no decay) for anything but daily notes.

    Args:
        rel_path: The chunk's file path, e.g. ``"memory/2026-07-20.md"``.
        source_kind: ``"durable"``, ``"daily"``, or ``"import"``.

    Returns:
        float: 1.0 for non-daily content, or ``0.5 ** (days_old /
            _DECAY_HALF_LIFE_DAYS)`` for a daily note — halving every
            ``_DECAY_HALF_LIFE_DAYS`` days since the date in its filename.
            Also 1.0 (no decay applied) if the filename doesn't parse as a
            date or the note is dated today/in the future — decay should
            never fail loudly over an unexpected filename shape.
    """
    if source_kind != "daily":
        return 1.0
    match = _DAILY_DATE_RE.search(rel_path)
    if not match:
        return 1.0
    try:
        note_date = _date.fromisoformat(match.group(1))
    except ValueError:
        return 1.0
    days_old = (_date.today() - note_date).days
    if days_old <= 0:
        return 1.0
    return 0.5 ** (days_old / _DECAY_HALF_LIFE_DAYS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1] (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash_text(text: str) -> str:
    """SHA256 hex digest of a file's content — used to detect real changes.

    Mirrors ``memory/migration.py``'s ``_hash_bytes`` (same algorithm, same
    reasoning: a cheap, reliable way to tell "this file's content actually
    changed" apart from "this file's mtime changed for some other reason").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_query(query: str) -> str:
    """Normalize and hash a search query — the recall-telemetry correlation key.

    Stage One Phase 5, slice A, Task 2: "hash normalized queries rather
    than storing unnecessary raw query text in promotion telemetry."
    Normalizing (lowercase, collapsed whitespace) before hashing means two
    queries that differ only in case or spacing count as the "same"
    query for recall-diversity purposes — what Task 3's later ranking
    calls "query diversity" cares about distinct *intents*, not
    incidental text formatting.

    Public (not a leading-underscore helper) because both
    :meth:`PostgresMemoryIndex.hybrid_search` (recording a surfaced
    result) and :meth:`~minion_assist.memory.service.MemoryService.mark_injected`
    (recording which of those were actually injected, later in the same
    turn) must compute the exact same hash for a given query to correlate
    with each other.
    """
    normalized = " ".join(query.lower().split())
    return _hash_text(normalized)


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
        embedding_dimensions: The configured embedding model's vector
            length (``config.embeddings.dimensions``), or ``None`` if no
            embedding backend is configured. Needed up front because a
            pgvector column's width is fixed at ``CREATE TABLE`` time — it
            can't be discovered later from the data itself. When ``None``
            (the default — matches every call site before Stage One
            Phase 4), :attr:`memory_chunk_embeddings` is never created and
            the vector lane simply doesn't exist for this instance.
        embedding_provider: The :class:`EmbeddingProvider` to call when
            indexing a chunk with no cached embedding yet (Stage One
            Phase 4, slice C). ``None`` (the default) means chunks are
            never embedded — the vector lane stays empty even if
            ``embedding_dimensions`` created the table, exactly as if no
            embedding backend were configured at all.
    """

    def __init__(
        self,
        url: str,
        embedding_dimensions: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        import psycopg  # noqa: PLC0415

        self._url = url
        self._psycopg = psycopg
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self._has_vector = False
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return a thread-local autocommit connection, creating it if needed.

        When ``self._has_vector`` is set, registers pgvector's psycopg
        adapter on every new connection — without it, a ``vector`` column
        round-trips as a raw ``"[0.1,0.2,0.3]"`` string on read instead of a
        usable :class:`pgvector.vector.Vector`, and (less critically) an
        outgoing Python list wouldn't reliably be recognized as a vector
        value on write either.
        """
        conn = getattr(_local, "conn", None)
        if conn is None or conn.closed:
            conn = self._psycopg.connect(self._url, autocommit=True)
            if self._has_vector:
                from pgvector.psycopg import register_vector  # noqa: PLC0415

                register_vector(conn)
            _local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        # pgvector extension (optional — silently skipped if not available,
        # same pattern as session/db.py's SessionDB._has_vector). Registered
        # on *this* connection immediately since _conn() only registers it
        # for connections created *after* self._has_vector becomes True —
        # this first connection was already open before that happened.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._has_vector = True
            from pgvector.psycopg import register_vector  # noqa: PLC0415

            register_vector(conn)
        except Exception:
            pass

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

        # Pinned files — Stage One Phase 4, slice B. Backs the "pinned" fusion
        # lane (slice C): a pinned file's chunks are always surfaced,
        # regardless of query match. Scoped to topic notes only (see
        # memory/service.py's pin()/unpin() docstrings for why).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_pins (
                agent_id   TEXT NOT NULL,
                rel_path   TEXT NOT NULL,
                pinned_at  DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (agent_id, rel_path)
            )
        """)

        # Recall telemetry — Stage One Phase 5, slice A. Keyed by rel_path,
        # not a memory_chunks row id: chunk ids aren't stable across a
        # reindex (the same lesson learned fixing the embedding cache in
        # Phase 4), and promotion decisions (Phase 5 slice C) operate at
        # the file/note level anyway. query_hash is a normalized-and-hashed
        # query (Task 2: "hash normalized queries rather than storing
        # unnecessary raw query text") — see hash_query().
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_recall_events (
                id           BIGSERIAL PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                rel_path     TEXT NOT NULL,
                query_hash   TEXT NOT NULL,
                surfaced_at  DOUBLE PRECISION NOT NULL,
                was_injected BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS memory_recall_events_file_idx "
            "ON memory_recall_events (agent_id, rel_path)"
        )

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

        # Embedding cache — only created when pgvector is available AND an
        # embedding backend is actually configured (Stage One Phase 4,
        # slice A). The vector width is fixed at creation time from the
        # configured model's dimensions; f-string interpolation is required
        # here because PostgreSQL's vector(N) type modifier can't be a bind
        # parameter, but embedding_dimensions is a validated int from
        # config.py, never external input, so this is safe.
        #
        # Keyed by (content_hash, model_identity), NOT chunk_id or agent_id.
        # reindex_file() deletes and reinserts every chunk on every call
        # (even when only one changed), so a chunk_id-keyed cache could
        # never hit — Task 3's "cache embeddings by model identity and
        # content hash" literally means content, not row identity.
        # memory_chunks.chunk_hash already stores this same hash per row,
        # so the vector lane joins through that column rather than needing
        # its own chunk_id reference. No agent_id either: identical chunk
        # text embeds identically regardless of which agent's note it came
        # from, so a cache hit is shared across agents for free.
        if self._has_vector and self._embedding_dimensions:
            # Self-healing one-time migration: an earlier build of this
            # table (before this content-hash-keyed shape existed) used
            # chunk_id as its primary key. Nothing ever wrote real
            # embeddings under that shape — it was never wired into any
            # write path until now — so it's safe to detect and drop it
            # here rather than requiring a manual fix on any machine that
            # already ran the earlier code.
            old_pk_cols = conn.execute("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'memory_chunk_embeddings'
                  AND tc.constraint_type = 'PRIMARY KEY'
            """).fetchall()
            if old_pk_cols and {r[0] for r in old_pk_cols} == {"chunk_id"}:
                conn.execute("DROP TABLE memory_chunk_embeddings")

            dims = int(self._embedding_dimensions)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_chunk_embeddings (
                    content_hash   TEXT NOT NULL,
                    model_identity TEXT NOT NULL,
                    embedding      vector({dims}) NOT NULL,
                    PRIMARY KEY (content_hash, model_identity)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunk_embeddings_hnsw "
                "ON memory_chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
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
        self._maybe_embed_chunks(chunks)
        return len(chunks)

    def remove_file(self, agent_id: str, rel_path: str) -> None:
        """Remove one file's chunks, ledger row, and pin (e.g. after on-disk deletion).

        A no-op if this file was never indexed/pinned — safe to call
        unconditionally. Also clears any pin (Stage One Phase 4, slice B)
        so a deleted note can never linger as an orphaned pin pointing at
        content that no longer exists.
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
        conn.execute(
            "DELETE FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

    # ------------------------------------------------------------------
    # Proposal indexing — Stage One Phase 5, slice B
    # ------------------------------------------------------------------

    def reindex_proposal(self, agent_id: str, proposal_id: int, claim_text: str) -> int:
        """Index one unreviewed capture-job proposal as a searchable chunk.

        Makes ``memory_proposals`` rows (``session/db.py``'s ``SessionDB``)
        reachable through the same lexical/hybrid search as real memory
        files, so Phase 5 slice C's consolidation ranking can use recall
        telemetry (:func:`hash_query`, :meth:`recall_stats`) on proposals
        the same way it does on notes.

        A proposal has no file on disk, so it gets a synthetic
        ``rel_path`` (``"proposals/{proposal_id}"``) instead of a real
        one, and deliberately gets no ``memory_files`` ledger row: that
        ledger exists to reconcile indexed chunks against on-disk content
        (see :meth:`reconcile_agent`), which doesn't apply here — a
        proposal is written once, by
        :class:`~minion_assist.memory.capture_worker.CaptureWorker`, and
        never edited in place. Because it has no ledger row,
        :meth:`rebuild_agent`/:meth:`reconcile_agent` (which only ever
        look at ``memory_files`` rows) never touch it; :meth:`force_rebuild_agent`'s
        live-swap DELETE explicitly excludes ``source_kind = 'proposal'``
        rows for the same reason.

        Args:
            agent_id: The agent the proposal belongs to.
            proposal_id: ``memory_proposals.id`` (returned by
                :meth:`~minion_assist.session.db.SessionDB.complete_capture_job`).
            claim_text: The proposal's claim text.

        Returns:
            int: How many chunks were written (a proposal's claim text is
                short — almost always 1 — but :func:`chunk_markdown` is
                reused unconditionally rather than special-cased).
        """
        conn = self._conn()
        rel_path = f"proposals/{proposal_id}"
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )
        chunks = chunk_markdown(claim_text)
        for i, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO memory_chunks
                    (agent_id, source_kind, rel_path, chunk_index, heading_path,
                     content, start_line, end_line, chunk_hash)
                VALUES (%s, 'proposal', %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    agent_id,
                    rel_path,
                    i,
                    " > ".join(chunk.heading_path),
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    _hash_text(chunk.content),
                ),
            )
        self._maybe_embed_chunks(chunks)
        return len(chunks)

    def remove_proposal(self, agent_id: str, proposal_id: int) -> None:
        """Remove one proposal's indexed chunks (e.g. once reviewed, no longer 'pending').

        A no-op if this proposal was never indexed — safe to call
        unconditionally. Phase 5 slice D (review/apply/reject) is the
        expected caller once a proposal's status leaves ``"pending"``.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
            (agent_id, f"proposals/{proposal_id}"),
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
        all_chunks = []
        for source_kind, rel_path, content in indexable_files:
            chunks = chunk_markdown(content)
            all_chunks.extend(chunks)
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
            # Proposal chunks (Stage One Phase 5, slice B) live in this same
            # table but have no memory_files ledger row and aren't part of
            # indexable_files — excluded here so a force-rebuild (a
            # files-only operation) can't wipe them out.
            conn.execute(
                "DELETE FROM memory_chunks WHERE agent_id = %s AND source_kind != 'proposal'",
                (agent_id,),
            )
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

        self._maybe_embed_chunks(all_chunks)
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
                ``"daily"``, ``"import"``, or ``"proposal"`` — Stage One
                Phase 5, slice B), or ``None`` to search every corpus. The
                plan's fourth corpus, "sessions", is
                deliberately not offered here — that's already the
                separate ``session_search`` tool's job; duplicating it into
                this index would mean two divergent ways to search the same
                message data.
            max_results: Maximum chunks to return.

        Returns:
            list[dict]: Best matches first. Each has ``id`` (the
                ``memory_chunks`` row id — used internally by
                :meth:`hybrid_search`'s fusion, not part of
                :class:`~minion_assist.memory.models.MemoryHit`),
                ``rel_path``, ``source_kind``, ``chunk_index``,
                ``heading_path``, ``content``, ``start_line``, ``end_line``,
                and ``score`` (the ``ts_rank`` value).
        """
        conn = self._conn()
        # Proposals (Stage One Phase 5, slice B) are unreviewed capture-job
        # output — never surfaced by a corpus-agnostic search (which is what
        # per-turn injection and search_memory both do) unless a caller
        # explicitly asks for corpus="proposal" (Phase 5 slice C/D's
        # consolidation review). Every lane below follows this same rule.
        if corpus:
            corpus_sql = " AND source_kind = %s"
        else:
            corpus_sql = " AND source_kind != 'proposal'"
        params: list = [query, agent_id, query]
        if corpus:
            params.append(corpus)
        params.append(max_results)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
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
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6],
                "end_line": r[7], "score": float(r[8]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Hybrid retrieval — Stage One Phase 4, slice C
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        agent_id: str,
        query: str,
        *,
        corpus: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Fuse path, lexical, vector, pinned, and recent lanes into one ranked result list.

        Each lane ranks candidates by a completely different signal (exact
        path match, ``ts_rank``, cosine similarity, pin status, recency), so
        no single lane's score is comparable to another's. Reciprocal-rank
        fusion (:func:`_reciprocal_rank_fusion`) sidesteps that by summing
        each candidate's ``1/(k+rank)`` contribution across every lane it
        appears in, rather than trying to average incompatible scores.

        Order of operations:

        1. Run each lane (each already ranked, best first).
        2. Fuse via RRF.
        3. Apply temporal decay to daily-note chunks (:func:`_decay_factor`).
        4. Re-rank by the decayed fused score.
        5. If an embedding provider is configured, run MMR
           (:meth:`_mmr`) to reduce near-duplicate snippets. Skipped
           without embeddings — MMR's whole purpose is catching *semantic*
           near-duplicates, which requires vectors to measure; there is no
           equivalent lexical-only substitute implemented here.
        6. Guarantee every currently pinned chunk (matching ``corpus``) is
           included, prepended ahead of the ranked results — "pinned"
           means always surfaced, not just boosted.
        7. Truncate to ``max_results``.

        Args:
            agent_id: Which agent's chunks to search.
            query: Free-text query.
            corpus: Restrict to one ``source_kind``, or ``None`` for every
                *reviewed* corpus. Applies to every lane, including pinned
                (in practice only "durable" pins exist, since pinning is
                scoped to topic notes — see ``memory/service.py``'s
                ``pin()``). ``None`` does *not* mean literally every
                chunk: ``"proposal"`` chunks (Stage One Phase 5, slice B —
                unreviewed capture-job output) are excluded unless
                ``corpus="proposal"`` is given explicitly, so a normal
                per-turn search or ``search_memory`` call never surfaces
                an unreviewed claim as if it were a reviewed note.
            max_results: Maximum chunks to return.

        Returns:
            list[dict]: Same shape as :meth:`search`'s results (``id``,
                ``rel_path``, ``source_kind``, ``chunk_index``,
                ``heading_path``, ``content``, ``start_line``, ``end_line``,
                ``score``) — ``score`` here is the fused, decayed RRF score
                (0.0 for a chunk included only because it's pinned and
                wasn't otherwise ranked by any lane).
        """
        lane_limit = max(max_results * 2, 20)  # overfetch each lane so fusion has real signal

        lexical_hits = self.search(agent_id, query, corpus=corpus, max_results=lane_limit)
        path_hits = self._path_lane(agent_id, query, corpus, lane_limit)
        vector_hits = self._vector_lane(agent_id, query, corpus, lane_limit)
        recent_hits = self._recent_lane(agent_id, corpus, lane_limit)

        scores, rows = _reciprocal_rank_fusion([path_hits, lexical_hits, vector_hits, recent_hits])
        for chunk_id, row in rows.items():
            scores[chunk_id] *= _decay_factor(row["rel_path"], row["source_kind"])

        ranked = sorted(rows.values(), key=lambda r: -scores[r["id"]])

        if self._embedding_provider is not None and self.has_vector_lane:
            final = self._mmr(ranked, scores, max_results)
        else:
            final = ranked[:max_results]

        pinned_hits = self._pinned_lane(agent_id, corpus)
        pinned_ids = {r["id"] for r in pinned_hits}
        non_pinned = [r for r in final if r["id"] not in pinned_ids]
        combined = pinned_hits + non_pinned

        results = []
        for row in combined[:max_results]:
            results.append({**row, "score": scores.get(row["id"], 0.0)})

        # Recall telemetry (Stage One Phase 5, slice A) — record every
        # result actually returned, regardless of caller (an explicit
        # search_memory tool call surfaces results just as much as
        # proactive per-turn injection does; mark_injected() is what later
        # distinguishes the subset that was actually injected). Never
        # allowed to turn a working search into a failed one.
        try:
            query_hash = hash_query(query)
            for row in results:
                self.record_recall(agent_id, row["rel_path"], query_hash)
        except Exception as exc:
            _log.debug(
                "Recording recall telemetry failed for agent %s: %s: %s",
                agent_id, type(exc).__name__, exc,
            )

        return results

    def _path_lane(self, agent_id: str, query: str, corpus: str | None, limit: int) -> list[dict]:
        """Exact/substring identifier match — catches a query naming a file directly.

        Complements the lexical lane's content-based ranking: a query like
        "project-goals" should surface a file at that path even if the
        word "project-goals" never appears in the file's *body* text.
        """
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not terms:
            return []
        conn = self._conn()
        term_conditions = " OR ".join(["rel_path ILIKE %s"] * len(terms))
        params: list = [agent_id] + [f"%{t}%" for t in terms]
        # See search()'s comment: proposals are excluded from a
        # corpus-agnostic query unless explicitly requested.
        if corpus:
            corpus_sql = " AND source_kind = %s"
            params.append(corpus)
        else:
            corpus_sql = " AND source_kind != 'proposal'"
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line
            FROM memory_chunks
            WHERE agent_id = %s AND ({term_conditions}) {corpus_sql}
            ORDER BY chunk_index
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _vector_lane(self, agent_id: str, query: str, corpus: str | None, limit: int) -> list[dict]:
        """Cosine-similarity ranking against cached chunk embeddings.

        Returns an empty lane (never raises) if no embedding provider is
        configured, or if embedding the query itself fails — the vector
        lane is optional by design; its absence just means the other lanes
        carry the fusion.
        """
        if self._embedding_provider is None or not self.has_vector_lane:
            return []
        try:
            [query_vector] = self._embedding_provider.embed([query])
        except Exception as exc:
            _log.debug("Embedding the query failed, skipping vector lane: %s: %s",
                       type(exc).__name__, exc)
            return []

        conn = self._conn()
        # See search()'s comment: proposals are excluded from a
        # corpus-agnostic query unless explicitly requested.
        if corpus:
            corpus_sql = " AND mc.source_kind = %s"
        else:
            corpus_sql = " AND mc.source_kind != 'proposal'"
        # Parameter order must match the SQL's %s occurrences top-to-bottom:
        # similarity's query_vector, the JOIN's model_identity, agent_id,
        # (corpus), ORDER BY's query_vector again, then LIMIT.
        params: list = [query_vector, self._embedding_provider.model_identity, agent_id]
        if corpus:
            params.append(corpus)
        params += [query_vector, limit]

        # %s::vector explicit casts: without them, psycopg sends a bare
        # Python list parameter as a plain float8[] array (even with
        # pgvector's adapter registered), and pgvector's <=> operator has
        # no overload for vector <=> float8[] — confirmed by a direct
        # smoke test against the dev database while building this lane.
        rows = conn.execute(
            f"""
            SELECT mc.id, mc.rel_path, mc.source_kind, mc.chunk_index, mc.heading_path,
                   mc.content, mc.start_line, mc.end_line,
                   1 - (mce.embedding <=> %s::vector) AS similarity
            FROM memory_chunks mc
            JOIN memory_chunk_embeddings mce
                ON mc.chunk_hash = mce.content_hash AND mce.model_identity = %s
            WHERE mc.agent_id = %s {corpus_sql}
            ORDER BY mce.embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
                "score": float(r[8]),
            }
            for r in rows
        ]

    def _pinned_lane(self, agent_id: str, corpus: str | None) -> list[dict]:
        """Every chunk of every currently pinned file, most recently pinned first."""
        pinned_paths = self.pinned_files(agent_id)
        if not pinned_paths:
            return []
        conn = self._conn()
        corpus_sql = " AND source_kind = %s" if corpus else ""
        params: list = [agent_id, pinned_paths]
        if corpus:
            params.append(corpus)

        rows = conn.execute(
            f"""
            SELECT id, rel_path, source_kind, chunk_index, heading_path, content,
                   start_line, end_line
            FROM memory_chunks
            WHERE agent_id = %s AND rel_path = ANY(%s) {corpus_sql}
            ORDER BY array_position(%s, rel_path), chunk_index
            """,
            [*params, pinned_paths],
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _recent_lane(self, agent_id: str, corpus: str | None, limit: int) -> list[dict]:
        """Most recently indexed files' chunks, regardless of content match."""
        conn = self._conn()
        corpus_sql = " AND mf.source_kind = %s" if corpus else ""
        params: list = [agent_id]
        if corpus:
            params.append(corpus)
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT mc.id, mc.rel_path, mc.source_kind, mc.chunk_index, mc.heading_path,
                   mc.content, mc.start_line, mc.end_line
            FROM memory_chunks mc
            JOIN memory_files mf ON mc.agent_id = mf.agent_id AND mc.rel_path = mf.rel_path
            WHERE mc.agent_id = %s {corpus_sql}
            ORDER BY mf.indexed_at DESC, mc.chunk_index
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "rel_path": r[1], "source_kind": r[2], "chunk_index": r[3],
                "heading_path": r[4], "content": r[5], "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def _mmr(self, ranked: list[dict], scores: dict[int, float], max_results: int) -> list[dict]:
        """Greedy maximal-marginal-relevance re-ranking to reduce near-duplicate snippets.

        Standard greedy MMR: repeatedly pick whichever remaining candidate
        maximizes ``score - max_similarity_to_already_selected``, so a
        highly-ranked chunk that's nearly identical to one already picked
        gets pushed down in favor of a lower-ranked but more novel one.
        Only ever called when an embedding provider is configured (see
        :meth:`hybrid_search`) — similarity here is cosine similarity
        between cached embeddings, falling back to no penalty (0.0
        similarity) for a candidate whose embedding isn't cached yet
        (e.g. it was written since the last successful embedding pass).

        Args:
            ranked: Candidates already sorted best-first by fused score.
            scores: chunk id -> fused score, from :meth:`hybrid_search`.
            max_results: How many to select.

        Returns:
            list[dict]: Up to ``max_results`` candidates, MMR-reordered.
        """
        model_identity = self._embedding_provider.model_identity
        vectors: dict[int, list[float] | None] = {}
        for row in ranked:
            content_hash = _hash_text(row["content"])
            vectors[row["id"]] = self.get_cached_embedding(content_hash, model_identity)

        selected: list[dict] = []
        selected_vectors: list[list[float]] = []
        remaining = list(ranked)

        while remaining and len(selected) < max_results:
            best_index = 0
            best_mmr_score = None
            for i, row in enumerate(remaining):
                vec = vectors.get(row["id"])
                if vec is not None and selected_vectors:
                    max_sim = max(_cosine_similarity(vec, sv) for sv in selected_vectors)
                else:
                    max_sim = 0.0
                mmr_score = 0.5 * scores[row["id"]] - 0.5 * max_sim
                if best_mmr_score is None or mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_index = i
            picked = remaining.pop(best_index)
            selected.append(picked)
            picked_vector = vectors.get(picked["id"])
            if picked_vector is not None:
                selected_vectors.append(picked_vector)

        return selected

    # ------------------------------------------------------------------
    # Pinning — Stage One Phase 4, slice B
    # ------------------------------------------------------------------
    #
    # Just the storage primitives here — the "pinned" fusion lane that
    # actually surfaces these in search results is Phase 4 slice C's job,
    # once fusion across lanes exists at all.

    def pin_file(self, agent_id: str, rel_path: str) -> None:
        """Pin a file so it's always surfaced by the pinned fusion lane, regardless of query match.

        Idempotent — pinning an already-pinned file just refreshes
        ``pinned_at`` rather than erroring or creating a duplicate row.
        """
        self._conn().execute(
            """
            INSERT INTO memory_pins (agent_id, rel_path, pinned_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (agent_id, rel_path) DO UPDATE SET pinned_at = EXCLUDED.pinned_at
            """,
            (agent_id, rel_path, time.time()),
        )

    def unpin_file(self, agent_id: str, rel_path: str) -> None:
        """Unpin a file. A no-op (not an error) if it wasn't pinned."""
        self._conn().execute(
            "DELETE FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        )

    def is_pinned(self, agent_id: str, rel_path: str) -> bool:
        """Whether a specific file is currently pinned."""
        row = self._conn().execute(
            "SELECT 1 FROM memory_pins WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        ).fetchone()
        return row is not None

    def pinned_files(self, agent_id: str) -> list[str]:
        """Every ``rel_path`` currently pinned for one agent, most recently pinned first."""
        rows = self._conn().execute(
            "SELECT rel_path FROM memory_pins WHERE agent_id = %s ORDER BY pinned_at DESC",
            (agent_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def _maybe_embed_chunks(self, chunks: list[Chunk]) -> None:
        """Best-effort: embed and cache ``chunks`` not already cached under the current model.

        Called from :meth:`reindex_file` and :meth:`force_rebuild_agent`
        right after their writes succeed — indexing itself never depends
        on this step. A no-op with no embedding provider configured. Skips
        any chunk whose content is already cached under the current
        model's identity (the common case on repeated indexing of mostly
        unchanged content), and batches everything else into one
        ``embed()`` call rather than one request per chunk.

        Never raises: a failed embedding call (provider unreachable, bad
        model name, etc.) is logged and swallowed, exactly like
        ``memory/service.py``'s ``_sync_index`` never lets an index-sync
        failure block the write that triggered it. The vector lane simply
        stays incomplete for these chunks until the next successful
        reindex retries them.
        """
        if self._embedding_provider is None or not self.has_vector_lane:
            return

        model_identity = self._embedding_provider.model_identity
        to_embed_hashes: list[str] = []
        to_embed_texts: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            content_hash = _hash_text(chunk.content)
            if content_hash in seen:
                continue  # duplicate content already queued this batch (e.g. two identical chunks)
            seen.add(content_hash)
            if self.get_cached_embedding(content_hash, model_identity) is not None:
                continue
            to_embed_hashes.append(content_hash)
            to_embed_texts.append(chunk.content)

        if not to_embed_texts:
            return

        try:
            vectors = self._embedding_provider.embed(to_embed_texts)
            for content_hash, vector in zip(to_embed_hashes, vectors):
                self.cache_embedding(content_hash, model_identity, vector)
        except Exception as exc:
            _log.debug(
                "Embedding %d chunk(s) failed (model %s): %s: %s",
                len(to_embed_texts), model_identity, type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Recall telemetry — Stage One Phase 5, slice A
    # ------------------------------------------------------------------
    #
    # hybrid_search() calls record_recall() for every result it actually
    # returns (Task 1: "record recall telemetry only for results actually
    # surfaced"). mark_injected() is called separately, later in the same
    # turn, once the caller knows which of those surfaced results were
    # actually selected for prompt injection (build_prompt_section() may
    # only fit a few of them within its token budget) — see
    # memory/service.py's mark_injected() and agents/session.py's send().

    def record_recall(self, agent_id: str, rel_path: str, query_hash: str) -> None:
        """Record that ``rel_path`` was surfaced by a search, keyed by its (already-hashed) query.

        Never raises out of :meth:`hybrid_search` — a telemetry-write
        failure must not turn a working search into a failed one. Callers
        needing that guarantee wrap this themselves (see
        :meth:`hybrid_search`'s own try/except around its recording loop).
        """
        self._conn().execute(
            """
            INSERT INTO memory_recall_events (agent_id, rel_path, query_hash, surfaced_at)
            VALUES (%s, %s, %s, %s)
            """,
            (agent_id, rel_path, query_hash, time.time()),
        )

    def mark_injected(self, agent_id: str, rel_paths: list[str], query_hash: str) -> None:
        """Mark the most recent recall event for each ``rel_path`` under this query as injected.

        Correlates by ``(agent_id, rel_path, query_hash)`` rather than a
        row id passed back through the whole call chain — safe because a
        turn's own ``record_recall`` calls and its later ``mark_injected``
        call for the *same query* happen microseconds apart, serialized by
        ``AgentSession._lock``, so "most recent matching row" is
        unambiguous in practice.

        Args:
            agent_id: The agent whose recall events to update.
            rel_paths: Which surfaced files were actually injected.
            query_hash: Must match the ``query_hash`` :meth:`hybrid_search`
                recorded for this same search (see :func:`hash_query`).
        """
        conn = self._conn()
        for rel_path in rel_paths:
            conn.execute(
                """
                UPDATE memory_recall_events SET was_injected = TRUE
                WHERE id = (
                    SELECT id FROM memory_recall_events
                    WHERE agent_id = %s AND rel_path = %s AND query_hash = %s
                    ORDER BY surfaced_at DESC
                    LIMIT 1
                )
                """,
                (agent_id, rel_path, query_hash),
            )

    def recall_stats(self, agent_id: str, rel_path: str) -> dict:
        """Aggregate recall telemetry for one file — a primitive Phase 5 slice C's ranking uses.

        Returns:
            dict: ``recall_count`` (total times surfaced), ``unique_queries``
                (distinct ``query_hash`` values — Task 3's "query
                diversity"), ``injected_count`` (times actually injected),
                and ``last_recalled_at`` (``None`` if never recalled).
        """
        rows = self._conn().execute(
            "SELECT query_hash, was_injected, surfaced_at FROM memory_recall_events "
            "WHERE agent_id = %s AND rel_path = %s",
            (agent_id, rel_path),
        ).fetchall()
        if not rows:
            return {
                "recall_count": 0,
                "unique_queries": 0,
                "injected_count": 0,
                "last_recalled_at": None,
            }
        return {
            "recall_count": len(rows),
            "unique_queries": len({r[0] for r in rows}),
            "injected_count": sum(1 for r in rows if r[1]),
            "last_recalled_at": max(r[2] for r in rows),
        }

    # ------------------------------------------------------------------
    # Embedding cache — Stage One Phase 4, slices A & C
    # ------------------------------------------------------------------
    #
    # Keyed by (content_hash, model_identity) — see the schema comment in
    # _ensure_schema() for why this is content-addressed rather than tied
    # to a specific memory_chunks row. Guarded by `has_vector_lane` rather
    # than relying on callers to check first, so a caller that forgets
    # simply gets a no-op/None instead of a SQL error from a table that
    # was never created.

    @property
    def has_vector_lane(self) -> bool:
        """Whether the embedding cache table exists (pgvector available and a model configured)."""
        return self._has_vector and bool(self._embedding_dimensions)

    def cache_embedding(
        self, content_hash: str, model_identity: str, embedding: list[float]
    ) -> None:
        """Store (or replace) the embedding for one exact chunk content, under one model.

        Args:
            content_hash: SHA256 of the chunk's content (matches
                ``memory_chunks.chunk_hash`` for the chunk(s) this text
                came from — see :func:`_hash_text`).
            model_identity: Identifies which model/endpoint produced this
                vector (:attr:`EmbeddingProvider.model_identity`) — lets a
                model change be detected rather than silently mixing
                vectors from two different embedding spaces.
            embedding: The vector itself. A no-op if no vector lane is
                configured (see :attr:`has_vector_lane`).
        """
        if not self.has_vector_lane:
            return
        self._conn().execute(
            """
            INSERT INTO memory_chunk_embeddings (content_hash, model_identity, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash, model_identity) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (content_hash, model_identity, embedding),
        )

    def get_cached_embedding(self, content_hash: str, model_identity: str) -> list[float] | None:
        """Look up the cached embedding for this exact chunk content, under this model.

        Returns ``None`` (a cache miss, not an error) both when there is no
        vector lane configured and when this exact ``(content_hash,
        model_identity)`` pair has simply never been embedded.

        Args:
            content_hash: Must match the cached row's value exactly.
            model_identity: Must match the cached row's value exactly.

        Returns:
            list[float] | None: The cached vector, or ``None`` on a miss.
        """
        if not self.has_vector_lane:
            return None
        row = self._conn().execute(
            "SELECT embedding FROM memory_chunk_embeddings "
            "WHERE content_hash = %s AND model_identity = %s",
            (content_hash, model_identity),
        ).fetchone()
        # register_vector() (see _conn()) makes row[0] a pgvector.vector.Vector,
        # not a plain list — .to_list() is that class's conversion method.
        return row[0].to_list() if row else None

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
