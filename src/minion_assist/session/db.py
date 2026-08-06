"""PostgreSQL-backed session and message store with FTS via tsvector.

Optional — disabled when DATABASE_URL is not set.  Call SessionDB(url) directly
or let minion.py create it from config; when unavailable the rest of the code
treats db=None and silently skips persistence.

Schema (created automatically on first connect):
  sessions(id, agent_id, source, started_at, last_active, turn_count, title, parent_id)
  messages(id BIGSERIAL, session_id, role, content, tool_name, timestamp,
           search_vector tsvector GENERATED ALWAYS AS ...)
  message_embeddings(message_id, embedding vector(N))  — only when pgvector available
  message_mirrors(session_id, event_id, message_id, mirrored_at) — idempotency
           ledger for mirroring (Stage One Phase 2, slice A)
  memory_capture_jobs(id, agent_id, session_id, source_from_message_id,
           source_to_message_id, idempotency_key, state, attempts, run_after,
           last_error, created_at, updated_at) — durable extraction queue
           (Stage One Phase 2, slice C)
  memory_proposals(id, job_id, agent_id, claim_text, created_at, status) —
           structured, unreviewed extraction output (Stage One Phase 2,
           slice C); status (Stage One Phase 5, slice B) defaults to
           "pending" until a later slice's review flow assigns
           "promoted"/"rejected"/"superseded"
  memory_commitment_jobs(id, agent_id, session_id, channel,
           source_from_message_id, source_to_message_id, idempotency_key,
           state, attempts, run_after, last_error, created_at, updated_at)
           — durable commitment-extraction queue, same shape as
           memory_capture_jobs plus a channel column (Stage One Phase 6,
           slice B)
  commitments(id, agent_id, session_id, channel, kind, sensitivity,
           source, status, reason, suggested_text, dedupe_key, confidence,
           due_earliest, due_latest, source_job_id, created_at, updated_at,
           sent_at, dismissed_at) — inferred, short-lived social
           follow-ups (Stage One Phase 6, slice B)

Durable capture jobs (Stage One Phase 2, slice C)
----------------------------------------------------
Replaces the per-turn daemon-thread extractor (``memory/extractor.py``) when
a database is configured: rather than firing a new thread every turn,
``agents/session.py`` enqueues a ``memory_capture_jobs`` row referencing the
PostgreSQL message ID range to extract from (using the IDs
:meth:`mirror_message` already returns — slice A). A single long-running
:class:`~minion_assist.memory.capture_worker.CaptureWorker` (started once at
process startup, not per turn) polls for due jobs, extracts facts via the
provider, and records them as ``memory_proposals`` — structured, unreviewed
claims, deliberately *not* written into a note file directly (Stage One
Phase 5 will decide how/whether to promote them; see
``minion-assist-docs/improve/memory-implementation-plan.md``).

Idempotency: :meth:`enqueue_capture_job`'s ``idempotency_key`` includes the
agent, session, message ID range, extraction prompt version, and model
(``memory/extractor.py``'s ``_EXTRACTION_PROMPT_VERSION`` +
``AgentSession``'s configured ``model_id``) — re-enqueuing the same range
under the same prompt/model is a no-op; changing either produces a new key
and a fresh extraction.

Degraded mode: when no database is configured, ``agents/session.py`` keeps
firing the original per-turn daemon-thread extractor unchanged (including
its 50-entry rolling cap) — there is no durable queue to fall back to
without a database (see ``docs/adr/0004-degraded-operation.md``).

Idempotent mirroring (Stage One Phase 2, slice A)
---------------------------------------------------
``add_message()`` alone has no notion of "already mirrored" — calling it
twice with the same content creates two rows. :meth:`mirror_message` adds
that: every mirror attempt is keyed by ``(session_id, event_id)``, where
``event_id`` is a message's stable identity (``messages.py``'s
``EVENT_ID_KEY`` / ``ensure_event_id()`` — a UUID embedded in the message
dict, assigned once and persisted to JSONL from then on). Calling
``mirror_message()`` again with the same ``(session_id, event_id)`` is a
no-op, not a duplicate insert.

This replaces the old ``replay_jsonl()``, which skipped an entire session
the moment it found *any* row already mirrored for that session — meaning a
crash after mirroring 2 of 20 messages left the other 18 unmirrored
forever, silently. :meth:`reconcile_session` / :meth:`reconcile_all_sessions`
mirror exactly the messages still missing, every time they run, so a partial
mirror from any prior crash is always completed rather than left behind.

Concurrency note: :meth:`mirror_message` checks-then-inserts without wrapping
both steps in one transaction. This is safe under this codebase's actual
concurrency model — a session's messages are only ever mirrored from that
session's own ``AgentSession.send()``, which is serialized by
``AgentSession._lock`` — but would need a real transaction (or an
``INSERT ... ON CONFLICT`` upsert) if two processes ever mirrored the same
session concurrently.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..messages import EVENT_ID_KEY, ensure_event_id
from ..schema_migrations import Migration, run_migrations

# Thread-local connection cache — one psycopg connection per OS thread.
_local = threading.local()


def _migration_001_baseline(conn) -> None:
    """Baseline migration (MEM-GAP-010): every table/index SessionDB needs.

    This is exactly what ``_ensure_schema()`` unconditionally ran before
    versioned migrations existed — every statement here is idempotent, so
    replaying it against an existing database (retroactively marking it
    "at version 1" the first time this code runs) is always safe. Do not
    edit this function once it has shipped: add a new, higher-numbered
    migration instead — :func:`~minion_assist.schema_migrations.run_migrations`
    will refuse to start if this function's source ever changes after being
    recorded in the ``schema_migrations`` ledger.

    The pgvector-dependent ``message_embeddings`` table/index are wrapped in
    their own ``try``/``except`` (safe under this class's autocommit
    connections — one failed statement doesn't poison the others) rather
    than a pre-checked flag, since this function has no access to
    ``SessionDB._has_vector`` — it only ever receives a raw connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'cli',
            started_at  DOUBLE PRECISION NOT NULL,
            last_active DOUBLE PRECISION NOT NULL,
            turn_count  INTEGER NOT NULL DEFAULT 0,
            title       TEXT,
            parent_id   TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sessions_agent_idx ON sessions (agent_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sessions_last_active_idx ON sessions (last_active DESC)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id           BIGSERIAL PRIMARY KEY,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT,
            tool_name    TEXT,
            timestamp    DOUBLE PRECISION NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english',
                    coalesce(content, '') || ' ' || coalesce(tool_name, ''))
            ) STORED
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS messages_fts_idx ON messages USING GIN (search_vector)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id)"
    )
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id BIGINT PRIMARY KEY,
                embedding  vector(1536) NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS msg_emb_hnsw
            ON message_embeddings USING hnsw (embedding vector_cosine_ops)
        """)
    except Exception:
        pass

    # Idempotency ledger for mirroring — see the module docstring's
    # "Idempotent mirroring" section.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_mirrors (
            session_id  TEXT NOT NULL,
            event_id    TEXT NOT NULL,
            message_id  BIGINT NOT NULL,
            mirrored_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (session_id, event_id)
        )
    """)

    # Durable capture-job queue — see the module docstring's "Durable
    # capture jobs" section.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_capture_jobs (
            id                     BIGSERIAL PRIMARY KEY,
            agent_id               TEXT NOT NULL,
            session_id             TEXT NOT NULL,
            source_from_message_id BIGINT NOT NULL,
            source_to_message_id   BIGINT NOT NULL,
            idempotency_key        TEXT NOT NULL UNIQUE,
            state                  TEXT NOT NULL DEFAULT 'pending',
            attempts               INTEGER NOT NULL DEFAULT 0,
            run_after              DOUBLE PRECISION NOT NULL,
            last_error             TEXT,
            created_at             DOUBLE PRECISION NOT NULL,
            updated_at             DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS capture_jobs_pending_idx
        ON memory_capture_jobs (state, run_after)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_proposals (
            id              BIGSERIAL PRIMARY KEY,
            job_id          BIGINT NOT NULL,
            agent_id        TEXT NOT NULL,
            claim_text      TEXT NOT NULL,
            created_at      DOUBLE PRECISION NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            rejected_reason TEXT NOT NULL DEFAULT ''
        )
    """)
    # status added in Stage One Phase 5, slice B; rejected_reason in
    # slice D — both ADD COLUMN IF NOT EXISTS so a database that
    # already has this table from an earlier slice picks them up
    # without a manual migration. status: "pending" (not yet
    # reviewed), "promoted"/"rejected"/"superseded" (assigned by
    # MemoryConsolidator's approve()/reject()/rollback()).
    # rejected_reason: the human-readable reason passed to reject() —
    # cleared back to "" on any other status transition (approve,
    # rollback) so it can never linger as a stale explanation for a
    # decision that's since been reversed.
    conn.execute(
        "ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS "
        "status TEXT NOT NULL DEFAULT 'pending'"
    )
    conn.execute(
        "ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS "
        "rejected_reason TEXT NOT NULL DEFAULT ''"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS proposals_agent_idx ON memory_proposals (agent_id)"
    )

    # Durable commitment-extraction job queue — Stage One Phase 6, slice
    # B. Structurally identical to memory_capture_jobs above (same
    # claim/complete/fail lifecycle, same idempotency-key mechanism),
    # but a separate table/queue: commitment output (kind, sensitivity,
    # due window, confidence, dedupe key) is a completely different
    # shape than memory_proposals' plain claim strings, so the two
    # pipelines stay cleanly separated rather than overloading one
    # table/worker with two incompatible output schemas. The extra
    # `channel` column (e.g. a Matrix room id, or "cli" outside Matrix)
    # is what memory_capture_jobs never needed — commitments must be
    # scoped to "exact agent and channel context" (the plan's Task 4).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_commitment_jobs (
            id                     BIGSERIAL PRIMARY KEY,
            agent_id               TEXT NOT NULL,
            session_id             TEXT NOT NULL,
            channel                TEXT NOT NULL,
            source_from_message_id BIGINT NOT NULL,
            source_to_message_id   BIGINT NOT NULL,
            idempotency_key        TEXT NOT NULL UNIQUE,
            state                  TEXT NOT NULL DEFAULT 'pending',
            attempts               INTEGER NOT NULL DEFAULT 0,
            run_after              DOUBLE PRECISION NOT NULL,
            last_error             TEXT,
            created_at             DOUBLE PRECISION NOT NULL,
            updated_at             DOUBLE PRECISION NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitment_jobs_pending_idx
        ON memory_commitment_jobs (state, run_after)
    """)

    # Commitments — inferred, short-lived social follow-ups (Stage One
    # Phase 6, Task 3). Scaled down from OpenClaw's real
    # src/commitments/types.ts CommitmentRecord (verified against that
    # source, not just the plan doc): same kind/sensitivity/source/
    # status vocabulary and due-window shape, without OpenClaw's
    # richer accountId/to/threadId/senderId scope fields minion-assist
    # has no equivalent concept for yet.
    #
    # due_earliest/due_latest are epoch seconds, not a single instant —
    # a window, so "check in tomorrow afternoon" has somewhere to land
    # rather than needing to resolve to one exact second.
    # dedupe_key + (agent_id, channel) is what
    # SessionDB.complete_commitment_job() upserts against, so a
    # near-identical inferred follow-up extends an existing pending
    # commitment's window rather than creating a duplicate.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS commitments (
            id             BIGSERIAL PRIMARY KEY,
            agent_id       TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            channel        TEXT NOT NULL,
            kind           TEXT NOT NULL,
            sensitivity    TEXT NOT NULL,
            source         TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'pending',
            reason         TEXT NOT NULL,
            suggested_text TEXT NOT NULL,
            dedupe_key     TEXT NOT NULL,
            confidence     DOUBLE PRECISION NOT NULL,
            due_earliest   DOUBLE PRECISION NOT NULL,
            due_latest     DOUBLE PRECISION NOT NULL,
            source_job_id  BIGINT NOT NULL,
            created_at     DOUBLE PRECISION NOT NULL,
            updated_at     DOUBLE PRECISION NOT NULL,
            sent_at        DOUBLE PRECISION,
            dismissed_at   DOUBLE PRECISION
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitments_scope_idx
        ON commitments (agent_id, channel, status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS commitments_dedupe_idx
        ON commitments (agent_id, channel, dedupe_key)
    """)


# Every migration SessionDB knows about, in the order they were introduced.
# Append new migrations here — never edit an existing entry's `apply`
# function once it has shipped (see _migration_001_baseline's docstring).
_SESSION_DB_MIGRATIONS = [
    Migration(1, "baseline", _migration_001_baseline),
]


def _msg_text(msg: dict) -> str | None:
    """Extract plain text from an OpenAI-format message dict."""
    content = msg.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts) or None
    return None


class SessionDB:
    """Thin wrapper around a PostgreSQL database for session + message storage.

    Args:
        url: libpq connection string, e.g. ``postgresql://user:pw@host/db``.
    """

    def __init__(self, url: str) -> None:
        import psycopg  # noqa: PLC0415
        self._url = url
        self._psycopg = psycopg
        self._has_vector = False
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
        # pgvector extension (optional — silently skipped if not available).
        # Kept here rather than inside a migration: this mutates self, which
        # a migration function (checksummed, self-less by design — see
        # schema_migrations.py) has no access to.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._has_vector = True
        except Exception:
            pass

        run_migrations(conn, "session_db", _SESSION_DB_MIGRATIONS)

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        agent_id: str,
        parent_id: str | None = None,
        source: str = "cli",
        started_at: float | None = None,
    ) -> None:
        """Insert a session row if it doesn't exist yet."""
        now = time.time()
        self._conn().execute(
            """
            INSERT INTO sessions (id, agent_id, source, started_at, last_active, parent_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (session_id, agent_id, source, started_at or now, now, parent_id),
        )

    def update_session(
        self,
        session_id: str,
        last_active: float | None = None,
        turn_count: int | None = None,
        title: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if last_active is not None:
            sets.append("last_active = %s")
            params.append(last_active)
        if turn_count is not None:
            sets.append("turn_count = %s")
            params.append(turn_count)
        if title is not None:
            sets.append("title = %s")
            params.append(title)
        if not sets:
            return
        params.append(session_id)
        self._conn().execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = %s", params
        )

    def get_sessions_by_ids(self, session_ids: list[str], agent_id: str) -> dict[str, dict]:
        """Return session metadata for ``session_ids``, filtered to ``agent_id``.

        The ``agent_id`` filter (MEM-GAP-002) means a session belonging to a
        different agent is silently omitted from the result rather than
        raising — callers (e.g. the session-search tool) treat "not in the
        dict" the same as "doesn't exist", so a cross-agent id can't be used
        to read another agent's session metadata.
        """
        if not session_ids:
            return {}
        rows = self._conn().execute(
            """
            SELECT id, agent_id, title, started_at, last_active, turn_count
            FROM sessions WHERE id = ANY(%s) AND agent_id = %s
            """,
            (session_ids, agent_id),
        ).fetchall()
        return {
            r[0]: {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4], "turn_count": r[5],
            }
            for r in rows
        }

    def list_sessions(self, agent_id: str, limit: int = 20) -> list[dict]:
        """Return this agent's most recent sessions, newest first.

        ``agent_id`` is mandatory (MEM-GAP-002) so one agent's browse/search
        tool can never enumerate another agent's session history — every
        agent's memory stays private to that agent, even from other agents
        in the same deployment.
        """
        rows = self._conn().execute(
            """
            SELECT s.id, s.agent_id, s.title, s.started_at, s.last_active, s.turn_count,
                (SELECT content FROM messages
                 WHERE session_id = s.id AND role = 'user'
                 ORDER BY id LIMIT 1) AS first_message
            FROM sessions s
            WHERE s.agent_id = %s
            ORDER BY s.last_active DESC
            LIMIT %s
            """,
            (agent_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4],
                "turn_count": r[5], "first_message": r[6],
            }
            for r in rows
        ]

    def delete_session(self, agent_id: str, session_id: str) -> dict | None:
        """Delete every row this database owns for one session (MEM-GAP-003).

        Removes, in dependency order: ``message_embeddings`` (keyed by
        message id, no ``session_id`` column of its own), ``message_mirrors``,
        ``messages``, this session's ``memory_proposals`` (via their
        ``memory_capture_jobs``), ``memory_capture_jobs``,
        ``memory_commitment_jobs``, ``commitments``, and finally the
        ``sessions`` row itself.

        Deliberately does **not** touch anything outside this database, and
        does **not** touch any durable memory note a *promoted* proposal's
        content was already merged into (``memory/topics/*.md``) — that
        content became independent, reviewed memory the moment it was
        approved, and deleting the source conversation doesn't retroactively
        invalidate it. The proposal *bookkeeping row* is still deleted here
        regardless of its status (pending/rejected/promoted): it's pure
        session-derived tracking data, not the note itself. Cleaning up the
        matching indexed proposal chunk, draft previews, and knowledge-graph
        evidence citations in :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
        is the caller's job (:meth:`~minion_assist.memory.service.MemoryService.forget_proposals`)
        — a separate class/connection this method has no access to, using
        the ``proposal_ids`` this method returns.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to delete.

        Returns:
            dict | None: ``None`` if ``session_id`` isn't owned by
                ``agent_id`` (fail closed — matches every other MEM-GAP-002
                scoped method's behavior; a missing/foreign session_id is
                treated as "nothing to delete," not an error). Otherwise
                ``{"messages": int, "proposal_ids": list[int]}`` — the
                message count deleted, and every deleted
                ``memory_proposals`` row's id, for the caller's PostgresMemoryIndex
                cleanup pass.
        """
        if not self._session_owned_by(session_id, agent_id):
            return None
        conn = self._conn()

        message_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM messages WHERE session_id = %s", (session_id,)
            ).fetchall()
        ]
        if message_ids and self._has_vector:
            conn.execute(
                "DELETE FROM message_embeddings WHERE message_id = ANY(%s)", (message_ids,)
            )
        conn.execute("DELETE FROM message_mirrors WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))

        job_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM memory_capture_jobs WHERE session_id = %s", (session_id,)
            ).fetchall()
        ]
        proposal_ids: list[int] = []
        if job_ids:
            proposal_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM memory_proposals WHERE job_id = ANY(%s)", (job_ids,)
                ).fetchall()
            ]
            if proposal_ids:
                conn.execute(
                    "DELETE FROM memory_proposals WHERE id = ANY(%s)", (proposal_ids,)
                )
        conn.execute("DELETE FROM memory_capture_jobs WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM memory_commitment_jobs WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM commitments WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

        return {"messages": len(message_ids), "proposal_ids": proposal_ids}

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        tool_name: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert a message row unconditionally; returns the new message id.

        No idempotency check — calling this twice with the same content
        creates two rows. Use :meth:`mirror_message` for the idempotent
        version (mirroring live turns and reconciliation both go through
        that, not this).
        """
        row = self._conn().execute(
            """
            INSERT INTO messages (session_id, role, content, tool_name, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (session_id, role, content, tool_name, timestamp or time.time()),
        ).fetchone()
        return row[0] if row else -1

    def is_mirrored(self, session_id: str, event_id: str) -> bool:
        """Return ``True`` if this exact message has already been mirrored."""
        row = self._conn().execute(
            "SELECT 1 FROM message_mirrors WHERE session_id = %s AND event_id = %s",
            (session_id, event_id),
        ).fetchone()
        return row is not None

    def mirror_message(
        self,
        session_id: str,
        event_id: str,
        role: str,
        content: str | None,
        tool_name: str | None = None,
        timestamp: float | None = None,
    ) -> int | None:
        """Idempotently insert a message, keyed by ``(session_id, event_id)``.

        Args:
            session_id: The session this message belongs to.
            event_id: The message's stable identity (``messages.py``'s
                ``ensure_event_id()``).
            role, content, tool_name, timestamp: Same as :meth:`add_message`.

        Returns:
            int | None: The new message row's id, or ``None`` if this exact
                ``(session_id, event_id)`` was already mirrored (a no-op,
                not an error — see the module docstring's concurrency note
                for why this check-then-insert is safe here).
        """
        if self.is_mirrored(session_id, event_id):
            return None
        message_id = self.add_message(
            session_id, role, content, tool_name=tool_name, timestamp=timestamp
        )
        self._conn().execute(
            """
            INSERT INTO message_mirrors (session_id, event_id, message_id, mirrored_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id, event_id) DO NOTHING
            """,
            (session_id, event_id, message_id, time.time()),
        )
        return message_id

    def search_messages(self, query: str, agent_id: str, limit: int = 10) -> list[dict]:
        """FTS search across this agent's own message content only.

        Joins to ``sessions`` and filters on ``agent_id`` (MEM-GAP-002) so a
        full-text query can never surface another agent's conversation
        history — agents are meant to be completely isolated from each
        other's memory, even though they share one PostgreSQL database.
        """
        rows = self._conn().execute(
            """
            SELECT
                m.id,
                m.session_id,
                m.role,
                m.content,
                m.tool_name,
                m.timestamp,
                ts_headline('english', coalesce(m.content, ''),
                    websearch_to_tsquery('english', %s),
                    'MaxWords=30, MinWords=10, StartSel=[, StopSel=]'
                ) AS snippet,
                ts_rank(m.search_vector, websearch_to_tsquery('english', %s)) AS rank
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.agent_id = %s
              AND m.search_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, agent_id, query, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "role": r[2],
                "content": r[3], "tool_name": r[4], "timestamp": r[5],
                "snippet": r[6], "rank": float(r[7]),
            }
            for r in rows
        ]

    def _session_owned_by(self, session_id: str, agent_id: str) -> bool:
        """Return ``True`` if ``session_id`` exists and belongs to ``agent_id``.

        Used to fail closed (MEM-GAP-002) before reading any message content
        keyed only by ``session_id`` — ``messages`` has no ``agent_id``
        column of its own, so this ownership check is what stops a caller
        from guessing another agent's session_id and scrolling through it.
        """
        row = self._conn().execute(
            "SELECT 1 FROM sessions WHERE id = %s AND agent_id = %s",
            (session_id, agent_id),
        ).fetchone()
        return row is not None

    def get_messages_around(
        self, session_id: str, agent_id: str, anchor_id: int, window: int = 5
    ) -> list[dict]:
        """Return messages in [anchor_id-window, anchor_id+window] for a session.

        Returns an empty list (rather than raising) if ``session_id`` isn't
        owned by ``agent_id`` — see :meth:`_session_owned_by`.
        """
        if not self._session_owned_by(session_id, agent_id):
            return []
        if anchor_id <= 0:
            row = self._conn().execute(
                "SELECT id FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            anchor_id = row[0] if row else 0
        if anchor_id <= 0:
            return []
        rows = self._conn().execute(
            """
            SELECT id, role, content, tool_name, timestamp
            FROM messages
            WHERE session_id = %s AND id BETWEEN %s AND %s
            ORDER BY id
            """,
            (session_id, anchor_id - window, anchor_id + window),
        ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "tool_name": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def get_session_bookends(
        self, session_id: str, agent_id: str, n: int = 3
    ) -> tuple[list[dict], list[dict]]:
        """Return first n and last n user/assistant messages for a session.

        Returns ``([], [])`` if ``session_id`` isn't owned by ``agent_id``
        (MEM-GAP-002) — see :meth:`_session_owned_by`.
        """
        if not self._session_owned_by(session_id, agent_id):
            return [], []
        conn = self._conn()
        first = conn.execute(
            """
            SELECT id, role, content FROM messages
            WHERE session_id = %s AND role IN ('user', 'assistant')
            ORDER BY id LIMIT %s
            """,
            (session_id, n),
        ).fetchall()
        last = conn.execute(
            """
            SELECT id, role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = %s AND role IN ('user', 'assistant')
                ORDER BY id DESC LIMIT %s
            ) sub ORDER BY id
            """,
            (session_id, n),
        ).fetchall()
        to_dict = lambda r: {"id": r[0], "role": r[1], "content": r[2]}  # noqa: E731
        return [to_dict(r) for r in first], [to_dict(r) for r in last]

    def get_messages_in_range(self, session_id: str, from_id: int, to_id: int) -> list[dict]:
        """Return every message in ``[from_id, to_id]`` for a session, in order.

        Used by :class:`~minion_assist.memory.capture_worker.CaptureWorker`
        to fetch a capture job's source messages (Stage One Phase 2, slice C).

        Args:
            session_id: The session to read from.
            from_id: Lowest message id to include (inclusive).
            to_id: Highest message id to include (inclusive).

        Returns:
            list[dict]: ``{"id", "role", "content", "tool_name", "timestamp"}``
                dicts, ordered by id. These are raw DB rows, not
                provider-ready message dicts — callers must build a clean
                ``{"role", "content"}`` dict before passing anything to
                ``provider.chat()`` (extra keys here would otherwise leak
                into the API request the same way ``EVENT_ID_KEY`` almost
                did — see ``providers/openai_compatible.py``'s
                ``_prepare_messages_for_openai``).
        """
        rows = self._conn().execute(
            """
            SELECT id, role, content, tool_name, timestamp
            FROM messages
            WHERE session_id = %s AND id BETWEEN %s AND %s
            ORDER BY id
            """,
            (session_id, from_id, to_id),
        ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "tool_name": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def list_message_ids(self, session_id: str) -> list[int]:
        """Every message id in one session, ascending.

        Stage One Phase 5, slice D: ``memory/consolidation.py``'s
        ``backfill_agent`` diffs this against
        :meth:`list_capture_job_ranges` to find message ranges no capture
        job has ever covered.
        """
        rows = self._conn().execute(
            "SELECT id FROM messages WHERE session_id = %s ORDER BY id", (session_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def list_capture_job_ranges(self, session_id: str) -> list[tuple[int, int]]:
        """Every ``(source_from_message_id, source_to_message_id)`` pair ever enqueued for one session.

        Every job state counts as "covered" here, including ``failed`` —
        backfill's job is to catch messages that were *never attempted*,
        not to retry a job that already ran and failed (that's
        ``CaptureWorker``'s own retry/backoff loop's responsibility, up to
        ``_MAX_ATTEMPTS``).

        Stage One Phase 5, slice D: the other half of
        ``backfill_agent``'s gap computation.
        """
        rows = self._conn().execute(
            """
            SELECT source_from_message_id, source_to_message_id
            FROM memory_capture_jobs
            WHERE session_id = %s
            """,
            (session_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def list_session_ids_for_agent(self, agent_id: str) -> list[str]:
        """Every session id ever recorded for one agent — no recency limit.

        Unlike :meth:`list_sessions` (most recent N, for display), this is
        exhaustive — Stage One Phase 5, slice D's ``backfill_agent`` needs
        every session, including ones long past any "recent" window, to
        find historical gaps.
        """
        rows = self._conn().execute(
            "SELECT id FROM sessions WHERE agent_id = %s", (agent_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def set_proposal_status(self, proposal_id: int, status: str, reason: str = "") -> None:
        """Set one proposal's review status (and optional rejection reason).

        Stage One Phase 5, slice D: used by
        :class:`~minion_assist.memory.consolidation.MemoryConsolidator`'s
        ``approve()`` (→ ``"promoted"``), ``reject()`` (→ ``"rejected"``,
        with a reason), and ``rollback()`` (→ back to ``"pending"``).

        Args:
            proposal_id: Which proposal to update.
            status: ``"pending"``, ``"promoted"``, ``"rejected"``, or
                ``"superseded"`` — not validated against this list here
                (this is an internal primitive; validation, if any, is the
                caller's job) the same way ``fail_capture_job`` doesn't
                re-validate ``state`` either.
            reason: Stored as ``rejected_reason`` — always set (default
                ``""``), so any transition *other than* a fresh
                ``reject()`` call clears a stale reason from a previous
                decision rather than leaving it to linger.
        """
        self._conn().execute(
            "UPDATE memory_proposals SET status = %s, rejected_reason = %s WHERE id = %s",
            (status, reason, proposal_id),
        )

    # ------------------------------------------------------------------
    # Reconciliation (Stage One Phase 2, slice A)
    # ------------------------------------------------------------------

    def reconcile_session(
        self,
        session_id: str,
        agent_id: str,
        messages: list[dict],
        mtime: float,
    ) -> int:
        """Mirror every message in ``messages`` that isn't mirrored yet.

        Unlike the old ``replay_jsonl()``, this never skips a session just
        because it already has some rows — it checks each message
        individually via :meth:`mirror_message`'s idempotency, so a crash
        that left a session partially mirrored is completed here rather
        than left behind forever.

        Messages without an existing event ID are assigned one in place
        (``messages.py``'s ``ensure_event_id()``) — callers MUST persist
        ``messages`` back to JSONL if this mutated any message, or those
        IDs will be regenerated (and mirrored as "new") on the next call.
        :meth:`reconcile_all_sessions` handles this automatically.

        Args:
            session_id: The session being reconciled.
            agent_id: The owning agent — used to upsert the session row.
            messages: The session's full message list, as loaded from JSONL.
            mtime: Used as the session's ``started_at``/mirrored timestamp
                when a value isn't otherwise available.

        Returns:
            int: Number of messages newly mirrored by this call.
        """
        self.upsert_session(session_id, agent_id, started_at=mtime)

        mirrored = 0
        for msg in messages:
            role = msg.get("role", "")
            if not role:
                continue
            event_id = ensure_event_id(msg)
            content = _msg_text(msg)
            tool_name = msg.get("name") or msg.get("tool_name")
            result = self.mirror_message(
                session_id, event_id, role, content, tool_name=tool_name, timestamp=mtime
            )
            if result is not None:
                mirrored += 1
        return mirrored

    def reconcile_all_sessions(
        self,
        short_term: object,
        agent_ids: list[str],
    ) -> int:
        """Reconcile every JSONL session file for every agent against ``message_mirrors``.

        Runs at startup (replacing the old one-time ``replay_jsonl()``) and
        is safe to call repeatedly — already-mirrored messages are no-ops.
        Re-saves a session's JSONL file via ``short_term.save()`` whenever
        :meth:`reconcile_session` assigned any message a new event ID, so
        the ID is never lost to a subsequent restart.

        Args:
            short_term: The agent's ``ShortTermMemory`` instance (duck-typed
                — only ``list_sessions``/``load``/``save`` are used).
            agent_ids: Every configured agent ID to reconcile.

        Returns:
            int: Total number of messages newly mirrored across all sessions.
        """
        total = 0
        for agent_id in agent_ids:
            for path in short_term.list_sessions(agent_id):
                session_id = path.stem
                messages = short_term.load(agent_id, session_id)
                if not messages:
                    continue

                ids_before = [m.get(EVENT_ID_KEY) for m in messages]
                mtime = path.stat().st_mtime
                total += self.reconcile_session(session_id, agent_id, messages, mtime)

                if [m.get(EVENT_ID_KEY) for m in messages] != ids_before:
                    short_term.save(agent_id, session_id, messages)
        return total

    # ------------------------------------------------------------------
    # Durable capture jobs (Stage One Phase 2, slice C)
    # ------------------------------------------------------------------

    def enqueue_capture_job(
        self,
        agent_id: str,
        session_id: str,
        source_from_message_id: int,
        source_to_message_id: int,
        idempotency_key: str,
    ) -> int | None:
        """Enqueue a capture job, idempotently.

        Args:
            agent_id: The agent whose exchange should be extracted from.
            session_id: The session the source messages belong to.
            source_from_message_id: Lowest message id in the source range.
            source_to_message_id: Highest message id in the source range.
            idempotency_key: Typically ``"{agent}:{session}:{from}-{to}:{prompt_version}:{model}"``
                (see ``memory/extractor.py``). Enqueuing the same key twice
                is a no-op, not a duplicate job.

        Returns:
            int | None: The new job's id, or ``None`` if a job with this
                ``idempotency_key`` already exists.
        """
        now = time.time()
        row = self._conn().execute(
            """
            INSERT INTO memory_capture_jobs
                (agent_id, session_id, source_from_message_id, source_to_message_id,
                 idempotency_key, state, attempts, run_after, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                agent_id, session_id, source_from_message_id, source_to_message_id,
                idempotency_key, now, now, now,
            ),
        ).fetchone()
        return row[0] if row else None

    def claim_next_capture_job(self) -> dict | None:
        """Atomically claim one pending, due capture job for processing.

        Uses ``FOR UPDATE SKIP LOCKED`` so multiple workers (should more than
        one ever run) never claim the same job twice. The whole claim is one
        SQL statement, so it is atomic even under this class's ``autocommit``
        connections — no explicit transaction needed.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id",
                "source_from_message_id", "source_to_message_id", "attempts"}``,
                or ``None`` if no job is currently due.
        """
        now = time.time()
        row = self._conn().execute(
            """
            UPDATE memory_capture_jobs
            SET state = 'running', updated_at = %s
            WHERE id = (
                SELECT id FROM memory_capture_jobs
                WHERE state = 'pending' AND run_after <= %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, agent_id, session_id, source_from_message_id,
                      source_to_message_id, attempts
            """,
            (now, now),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "agent_id": row[1],
            "session_id": row[2],
            "source_from_message_id": row[3],
            "source_to_message_id": row[4],
            "attempts": row[5],
        }

    def complete_capture_job(self, job_id: int, proposals: list[str]) -> list[dict]:
        """Mark a capture job done and record its extracted proposals.

        Args:
            job_id: The job to complete.
            proposals: 0 or more extracted claim strings. An empty list is a
                normal, successful outcome ("nothing worth remembering this
                time"), not a failure.

        Returns:
            list[dict]: One ``{"id", "agent_id", "claim_text"}`` per newly
                inserted proposal row, in the same order as ``proposals``.
                Stage One Phase 5, slice B: :class:`~minion_assist.memory.capture_worker.CaptureWorker`
                uses these ids to index each new proposal as searchable
                right after this call returns, without a second query.
        """
        now = time.time()
        conn = self._conn()
        job_row = conn.execute(
            "SELECT agent_id FROM memory_capture_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        agent_id = job_row[0] if job_row else ""
        new_proposals = []
        for claim in proposals:
            row = conn.execute(
                """
                INSERT INTO memory_proposals (job_id, agent_id, claim_text, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (job_id, agent_id, claim, now),
            ).fetchone()
            new_proposals.append({"id": row[0], "agent_id": agent_id, "claim_text": claim})
        conn.execute(
            "UPDATE memory_capture_jobs SET state = 'done', updated_at = %s WHERE id = %s",
            (now, job_id),
        )
        return new_proposals

    def fail_capture_job(
        self, job_id: int, error: str, backoff_seconds: float, max_attempts: int = 5
    ) -> None:
        """Record a failed capture-job attempt — retry with backoff, or give up.

        Args:
            job_id: The job that failed.
            error: Short description of the failure, stored as ``last_error``.
            backoff_seconds: How long to wait before this job is eligible to
                be claimed again (``run_after = now + backoff_seconds``).
            max_attempts: Once ``attempts`` reaches this count, the job is
                marked ``'failed'`` instead of ``'pending'`` — it stops being
                retried automatically but remains inspectable in the table.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            "SELECT attempts FROM memory_capture_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        state = "failed" if attempts >= max_attempts else "pending"
        conn.execute(
            """
            UPDATE memory_capture_jobs
            SET state = %s, attempts = %s, run_after = %s, last_error = %s, updated_at = %s
            WHERE id = %s
            """,
            (state, attempts, now + backoff_seconds, error, now, job_id),
        )

    # ------------------------------------------------------------------
    # Durable commitment-extraction jobs (Stage One Phase 6, slice B)
    # ------------------------------------------------------------------
    #
    # Structurally identical to the capture-job methods above — see the
    # module docstring's commitments schema note for why this is a
    # separate table/queue rather than an extension of memory_capture_jobs.

    def enqueue_commitment_job(
        self,
        agent_id: str,
        session_id: str,
        channel: str,
        source_from_message_id: int,
        source_to_message_id: int,
        idempotency_key: str,
    ) -> int | None:
        """Enqueue a commitment-extraction job, idempotently.

        Args mirror :meth:`enqueue_capture_job` with one addition:
        ``channel`` — a Matrix room id, or ``"cli"`` outside Matrix. This
        is what scopes a later-extracted commitment to "exact agent and
        channel context" (the plan's Task 4).

        Returns:
            int | None: The new job's id, or ``None`` if a job with this
                ``idempotency_key`` already exists.
        """
        now = time.time()
        row = self._conn().execute(
            """
            INSERT INTO memory_commitment_jobs
                (agent_id, session_id, channel, source_from_message_id,
                 source_to_message_id, idempotency_key, state, attempts,
                 run_after, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                agent_id, session_id, channel, source_from_message_id,
                source_to_message_id, idempotency_key, now, now, now,
            ),
        ).fetchone()
        return row[0] if row else None

    def claim_next_commitment_job(self) -> dict | None:
        """Atomically claim one pending, due commitment-extraction job.

        Same ``FOR UPDATE SKIP LOCKED`` mechanics as
        :meth:`claim_next_capture_job` — see its docstring.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id", "channel",
                "source_from_message_id", "source_to_message_id",
                "attempts"}``, or ``None`` if no job is currently due.
        """
        now = time.time()
        row = self._conn().execute(
            """
            UPDATE memory_commitment_jobs
            SET state = 'running', updated_at = %s
            WHERE id = (
                SELECT id FROM memory_commitment_jobs
                WHERE state = 'pending' AND run_after <= %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, agent_id, session_id, channel, source_from_message_id,
                      source_to_message_id, attempts
            """,
            (now, now),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "agent_id": row[1],
            "session_id": row[2],
            "channel": row[3],
            "source_from_message_id": row[4],
            "source_to_message_id": row[5],
            "attempts": row[6],
        }

    def list_pending_commitments_for_scope(
        self, agent_id: str, channel: str, limit: int = 8
    ) -> list[dict]:
        """List pending commitments already tracked for one (agent, channel) scope.

        Fed into the extraction prompt as ``existing_pending`` context
        (``memory/commitments.py``'s ``build_commitment_extraction_prompt``)
        so the model can extend an already-tracked follow-up instead of
        proposing a near-duplicate — the same anti-duplication input
        OpenClaw's real extractor uses (verified against
        ``ref-repos/openclaw/src/commitments/extraction.ts``'s
        ``hydrateCommitmentExtractionItem``).

        Returns:
            list[dict]: ``{"kind", "reason", "dedupe_key", "due_earliest",
                "due_latest"}`` dicts, most recently created first, capped
                at ``limit``.
        """
        rows = self._conn().execute(
            """
            SELECT kind, reason, dedupe_key, due_earliest, due_latest
            FROM commitments
            WHERE agent_id = %s AND channel = %s AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (agent_id, channel, limit),
        ).fetchall()
        return [
            {
                "kind": r[0], "reason": r[1], "dedupe_key": r[2],
                "due_earliest": r[3], "due_latest": r[4],
            }
            for r in rows
        ]

    def complete_commitment_job(self, job_id: int, candidates: list[dict]) -> list[dict]:
        """Mark a commitment job done and record its extracted candidates.

        A candidate whose ``dedupe_key`` matches an existing *pending*
        commitment in the same ``(agent_id, channel)`` scope is upserted
        into that row (widening the due window to
        ``min(earliest)``/``max(latest)``, keeping the higher confidence)
        rather than inserted as a duplicate — mirrors OpenClaw's real
        ``upsertInferredCommitments`` (verified against
        ``ref-repos/openclaw/src/commitments/store.ts``), scaled down to
        this codebase's simpler needs.

        Args:
            job_id: The job to complete.
            candidates: 0 or more validated candidate dicts (see
                ``memory/commitments.py``'s ``extract_commitments`` —
                already confidence-gated and due-window-clamped by the
                time they reach here). Each needs ``kind``, ``sensitivity``,
                ``source``, ``reason``, ``suggested_text``, ``dedupe_key``,
                ``confidence``, ``due_earliest``, ``due_latest`` (all as
                produced by ``extract_commitments``).

        Returns:
            list[dict]: ``{"id", "created"}`` per candidate — ``created``
                is ``False`` when the candidate was merged into an
                existing pending commitment rather than inserted fresh.
        """
        now = time.time()
        conn = self._conn()
        job_row = conn.execute(
            "SELECT agent_id, session_id, channel FROM memory_commitment_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        agent_id, session_id, channel = job_row if job_row else ("", "", "")

        results = []
        for candidate in candidates:
            existing = conn.execute(
                """
                SELECT id, due_earliest, due_latest, confidence FROM commitments
                WHERE agent_id = %s AND channel = %s AND dedupe_key = %s AND status = 'pending'
                """,
                (agent_id, channel, candidate["dedupe_key"]),
            ).fetchone()
            if existing is not None:
                existing_id, existing_earliest, existing_latest, existing_confidence = existing
                conn.execute(
                    """
                    UPDATE commitments
                    SET due_earliest = %s, due_latest = %s, confidence = %s,
                        reason = %s, suggested_text = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (
                        min(existing_earliest, candidate["due_earliest"]),
                        max(existing_latest, candidate["due_latest"]),
                        max(existing_confidence, candidate["confidence"]),
                        candidate["reason"], candidate["suggested_text"], now,
                        existing_id,
                    ),
                )
                results.append({"id": existing_id, "created": False})
                continue
            row = conn.execute(
                """
                INSERT INTO commitments
                    (agent_id, session_id, channel, kind, sensitivity, source, reason,
                     suggested_text, dedupe_key, confidence, due_earliest, due_latest,
                     source_job_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    agent_id, session_id, channel, candidate["kind"], candidate["sensitivity"],
                    candidate["source"], candidate["reason"], candidate["suggested_text"],
                    candidate["dedupe_key"], candidate["confidence"], candidate["due_earliest"],
                    candidate["due_latest"], job_id, now, now,
                ),
            ).fetchone()
            results.append({"id": row[0], "created": True})

        conn.execute(
            "UPDATE memory_commitment_jobs SET state = 'done', updated_at = %s WHERE id = %s",
            (now, job_id),
        )
        return results

    def fail_commitment_job(
        self, job_id: int, error: str, backoff_seconds: float, max_attempts: int = 5
    ) -> None:
        """Record a failed commitment-extraction attempt — retry with backoff, or give up.

        Identical mechanics to :meth:`fail_capture_job` — see its
        docstring.
        """
        now = time.time()
        conn = self._conn()
        row = conn.execute(
            "SELECT attempts FROM memory_commitment_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        state = "failed" if attempts >= max_attempts else "pending"
        conn.execute(
            """
            UPDATE memory_commitment_jobs
            SET state = %s, attempts = %s, run_after = %s, last_error = %s, updated_at = %s
            WHERE id = %s
            """,
            (state, attempts, now + backoff_seconds, error, now, job_id),
        )

    # ------------------------------------------------------------------
    # Queue health (MEM-GAP-016)
    # ------------------------------------------------------------------

    def queue_lag_summary(self, agent_id: str) -> dict:
        """Report how far behind the capture/commitment job queues are for one agent.

        Unlike :class:`~minion_assist.worker_health.WorkerHealth` (in-process
        liveness, only visible to the same running process), this is a plain
        SQL aggregate over ``memory_capture_jobs``/``memory_commitment_jobs``
        — a fact anyone with a database connection can observe, including a
        separate ``minion-assist memory status --deep`` CLI invocation.
        "Pending" here means ``state = 'pending'`` (claimed-but-running or
        already-finished jobs are excluded — a stuck ``'running'`` job is a
        different, rarer failure mode this doesn't attempt to detect).

        Args:
            agent_id: Which agent's queues to summarize.

        Returns:
            dict: ``{"capture": {"pending_count", "oldest_pending_age_s"},
                "commitment": {"pending_count", "oldest_pending_age_s"}}``.
                ``oldest_pending_age_s`` is seconds since the oldest pending
                job was created, or ``None`` if the queue is empty.
        """
        conn = self._conn()
        now = time.time()

        def _lane(table: str) -> dict:
            row = conn.execute(
                f"SELECT count(*), MIN(created_at) FROM {table} "
                "WHERE agent_id = %s AND state = 'pending'",
                (agent_id,),
            ).fetchone()
            pending_count = row[0] or 0
            oldest_created_at = row[1]
            oldest_pending_age_s = (
                now - oldest_created_at if oldest_created_at is not None else None
            )
            return {"pending_count": pending_count, "oldest_pending_age_s": oldest_pending_age_s}

        return {
            "capture": _lane("memory_capture_jobs"),
            "commitment": _lane("memory_commitment_jobs"),
        }

    def find_uncovered_capture_range(
        self, agent_id: str, session_id: str
    ) -> tuple[int, int] | None:
        """Return the range of mirrored messages newer than this session's last capture job.

        MEM-GAP-007: ``AgentSession._send_locked()``'s own
        ``enqueue_capture_job()`` call can fail (e.g. a transient
        connection drop) and, before this method existed, that turn's
        capture job would simply never exist — nothing ever retried it.
        :class:`~minion_assist.memory.reconciliation_scheduler.ReconciliationScheduler`
        calls this periodically per session and enqueues one catch-up job
        covering the whole gap when it finds one.

        Deliberately coarser than ``_send_locked()``'s own per-turn range
        (exactly "the last exchange"): reconstructing exact per-turn
        boundaries from message content alone (which messages belong to
        which turn, given tool calls can appear in between) would risk
        getting the pairing subtly wrong. Returning "everything mirrored
        since the last job's covered end" is simpler and safe — however
        many turns accumulated in the gap (normally zero; only nonzero
        after a genuine enqueue failure) become one catch-up job instead
        of exactly reconstructed per-turn ones.

        A job counts as "covering" its range regardless of its current
        ``state`` (pending/running/done/failed) — a failed extraction
        attempt still means a job was successfully *enqueued* for that
        range; re-enqueuing here would create a duplicate under a new
        idempotency key, not retry the original failed attempt.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to check.

        Returns:
            tuple[int, int] | None: ``(from_id, to_id)`` for uncovered
                mirrored user/assistant messages, or ``None`` if
                ``session_id`` isn't owned by ``agent_id`` or there's
                nothing to catch up on.
        """
        if not self._session_owned_by(session_id, agent_id):
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT MAX(source_to_message_id) FROM memory_capture_jobs "
            "WHERE agent_id = %s AND session_id = %s",
            (agent_id, session_id),
        ).fetchone()
        last_covered = row[0] if row and row[0] is not None else 0
        row = conn.execute(
            "SELECT MIN(id), MAX(id) FROM messages "
            "WHERE session_id = %s AND role IN ('user', 'assistant') "
            "AND content IS NOT NULL AND id > %s",
            (session_id, last_covered),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return (row[0], row[1])

    def find_uncovered_commitment_range(
        self, agent_id: str, session_id: str
    ) -> tuple[str, int, int] | None:
        """Return the range of mirrored messages newer than this session's last commitment job.

        Same reasoning and "coarser catch-up, not exact per-turn
        reconstruction" trade-off as :meth:`find_uncovered_capture_range` —
        see its docstring.

        Commitment jobs are additionally channel-scoped (a Matrix room id,
        or ``"cli"``), so this only reconciles a session that has *at
        least one* prior commitment job to infer the channel from. A
        session with zero commitment jobs ever most likely means
        commitments were disabled (``config.json``'s
        ``commitments.enabled``) when its messages were created, not a
        missed enqueue — left alone rather than guessing at a channel.

        Args:
            agent_id: The agent this session must belong to.
            session_id: The session to check.

        Returns:
            tuple[str, int, int] | None: ``(channel, from_id, to_id)`` for
                uncovered mirrored user/assistant messages, using the most
                recently covered job's channel, or ``None`` if
                ``session_id`` isn't owned by ``agent_id``, this session
                has no prior commitment job, or there's nothing to catch
                up on.
        """
        if not self._session_owned_by(session_id, agent_id):
            return None
        conn = self._conn()
        row = conn.execute(
            """
            SELECT channel, MAX(source_to_message_id)
            FROM memory_commitment_jobs
            WHERE agent_id = %s AND session_id = %s
            GROUP BY channel
            ORDER BY MAX(source_to_message_id) DESC
            LIMIT 1
            """,
            (agent_id, session_id),
        ).fetchone()
        if row is None:
            return None
        channel, last_covered = row[0], row[1] or 0
        row2 = conn.execute(
            "SELECT MIN(id), MAX(id) FROM messages "
            "WHERE session_id = %s AND role IN ('user', 'assistant') "
            "AND content IS NOT NULL AND id > %s",
            (session_id, last_covered),
        ).fetchone()
        if row2 is None or row2[0] is None:
            return None
        return (channel, row2[0], row2[1])

    # ------------------------------------------------------------------
    # Commitment lifecycle — Stage One Phase 6, slice C
    # ------------------------------------------------------------------

    # How long past its own due_latest a still-pending commitment survives
    # before being auto-expired. OpenClaw's own shipped default
    # (ref-repos/openclaw/src/commitments/config.ts's
    # DEFAULT_COMMITMENT_EXPIRE_AFTER_HOURS), reused for the same
    # "no evaluation data of our own yet" reason
    # memory/commitments.py's confidence thresholds already document.
    _COMMITMENT_EXPIRE_AFTER_SECONDS = 72 * 3600.0

    def expire_stale_commitments(self, agent_id: str, now: float) -> int:
        """Mark any ``pending`` commitment whose window closed long ago as ``"expired"``.

        Called at the top of :meth:`list_due_commitments_for_agent` (lazy
        sweep on every read, mirroring OpenClaw's own
        ``expireStaleCommitments`` — verified against
        ``ref-repos/openclaw/src/commitments/store.ts``) rather than
        needing a dedicated expiry scheduler.

        Args:
            agent_id: Which agent's commitments to sweep.
            now: Epoch seconds "now" is evaluated at.

        Returns:
            int: How many commitments were just expired.
        """
        cutoff = now - self._COMMITMENT_EXPIRE_AFTER_SECONDS
        result = self._conn().execute(
            """
            UPDATE commitments
            SET status = 'expired', updated_at = %s
            WHERE agent_id = %s AND status = 'pending' AND due_latest < %s
            """,
            (now, agent_id, cutoff),
        )
        return result.rowcount

    def list_due_commitments_for_agent(
        self, agent_id: str, now: float, max_per_day: int = 3, max_per_heartbeat: int = 3
    ) -> list[dict]:
        """List commitments currently due for one agent, across every channel, rate-limited.

        "Due" means ``status = 'pending'`` and ``due_earliest <= now``.
        Unlike :meth:`list_pending_commitments_for_scope` (a single
        ``(agent_id, channel)`` scope, used for extraction-time
        deduplication), this spans every channel the agent has commitments
        in — the delivery side of "multi-room-aware": one heartbeat pass
        can see commitments from several different rooms at once, each
        later resolved to its own room at send time (see
        ``heartbeat.py``'s ``_deliver_to_channel``), rather than needing a
        separate heartbeat run per room.

        Rate-limited the same way OpenClaw's own
        ``listDueCommitmentsForSession`` is (verified against
        ``ref-repos/openclaw/src/commitments/store.ts``): capped at
        ``max_per_heartbeat`` per call *and* at ``max_per_day`` total
        ``"sent"`` commitments in the trailing 24 hours — whichever is
        smaller wins.

        Args:
            agent_id: Which agent's commitments to list.
            now: Epoch seconds "now" is evaluated at.
            max_per_day: Maximum commitments *sent* per rolling 24 hours.
            max_per_heartbeat: Maximum commitments returned by one call.

        Returns:
            list[dict]: ``{"id", "agent_id", "session_id", "channel",
                "kind", "sensitivity", "source", "reason",
                "suggested_text", "dedupe_key", "confidence",
                "due_earliest", "due_latest"}`` dicts, earliest-due first
                (ties broken by ``created_at`` then ``id`` for
                deterministic ordering) — empty once the day's quota is
                used up.
        """
        self.expire_stale_commitments(agent_id, now)
        conn = self._conn()
        sent_today = conn.execute(
            """
            SELECT count(*) FROM commitments
            WHERE agent_id = %s AND status = 'sent' AND sent_at >= %s
            """,
            (agent_id, now - 86400.0),
        ).fetchone()[0]
        remaining_today = max_per_day - sent_today
        if remaining_today <= 0:
            return []
        limit = min(max_per_heartbeat, remaining_today)

        rows = conn.execute(
            """
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, reason,
                   suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments
            WHERE agent_id = %s AND status = 'pending' AND due_earliest <= %s
            ORDER BY due_earliest ASC, created_at ASC, id ASC
            LIMIT %s
            """,
            (agent_id, now, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "session_id": r[2], "channel": r[3],
                "kind": r[4], "sensitivity": r[5], "source": r[6], "reason": r[7],
                "suggested_text": r[8], "dedupe_key": r[9], "confidence": r[10],
                "due_earliest": r[11], "due_latest": r[12],
            }
            for r in rows
        ]

    def get_commitment(self, commitment_id: int) -> dict | None:
        """Look up one commitment by id.

        Returns:
            dict | None: ``{"id", "agent_id", "session_id", "channel",
                "kind", "sensitivity", "source", "status", "reason",
                "suggested_text", "dedupe_key", "confidence",
                "due_earliest", "due_latest"}``, or ``None`` if it doesn't
                exist.
        """
        row = self._conn().execute(
            """
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, status,
                   reason, suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments WHERE id = %s
            """,
            (commitment_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "agent_id": row[1], "session_id": row[2], "channel": row[3],
            "kind": row[4], "sensitivity": row[5], "source": row[6], "status": row[7],
            "reason": row[8], "suggested_text": row[9], "dedupe_key": row[10],
            "confidence": row[11], "due_earliest": row[12], "due_latest": row[13],
        }

    def mark_commitment_sent(self, commitment_id: int, now: float | None = None) -> None:
        """Mark a commitment ``"sent"`` — a check-in was actually delivered for it.

        Called by ``tools/commitment_response.py``'s
        ``RespondToCommitmentTool`` right after delivery succeeds.
        """
        now = time.time() if now is None else now
        self._conn().execute(
            "UPDATE commitments SET status = 'sent', sent_at = %s, updated_at = %s WHERE id = %s",
            (now, now, commitment_id),
        )

    def mark_commitment_dismissed(self, commitment_id: int, now: float | None = None) -> None:
        """Mark a commitment ``"dismissed"`` — reviewed and deliberately not sent.

        Called by ``tools/commitment_response.py``'s
        ``DismissCommitmentTool``.
        """
        now = time.time() if now is None else now
        self._conn().execute(
            "UPDATE commitments SET status = 'dismissed', dismissed_at = %s, updated_at = %s "
            "WHERE id = %s",
            (now, now, commitment_id),
        )

    def list_commitments(
        self, agent_id: str, status: str | None = None, channel: str | None = None
    ) -> list[dict]:
        """List commitments for one agent, most recently created first — the CLI's primitive.

        Args:
            agent_id: Which agent's commitments to list.
            status: Restrict to one status (``"pending"``/``"sent"``/
                ``"dismissed"``/``"snoozed"``/``"expired"``), or ``None``
                for every status.
            channel: Restrict to one channel, or ``None`` for every
                channel.

        Returns:
            list[dict]: Same shape as :meth:`get_commitment`'s return
                value, newest first.
        """
        clauses = ["agent_id = %s"]
        params: list = [agent_id]
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if channel is not None:
            clauses.append("channel = %s")
            params.append(channel)
        where_sql = " AND ".join(clauses)
        rows = self._conn().execute(
            f"""
            SELECT id, agent_id, session_id, channel, kind, sensitivity, source, status,
                   reason, suggested_text, dedupe_key, confidence, due_earliest, due_latest
            FROM commitments
            WHERE {where_sql}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "session_id": r[2], "channel": r[3],
                "kind": r[4], "sensitivity": r[5], "source": r[6], "status": r[7],
                "reason": r[8], "suggested_text": r[9], "dedupe_key": r[10],
                "confidence": r[11], "due_earliest": r[12], "due_latest": r[13],
            }
            for r in rows
        ]

    def delete_commitment(self, agent_id: str, commitment_id: int) -> bool:
        """Permanently delete one commitment — Task 7's "complete scoped deletion."

        Scoped to ``agent_id`` (not just the raw id) so an operator can't
        accidentally delete a different agent's commitment by guessing an
        id — the same scoping discipline ``memory/service.py``'s
        ``pin()``/``unpin()`` already applies.

        Args:
            agent_id: The commitment's owning agent — must match, or
                nothing is deleted.
            commitment_id: Which commitment to delete.

        Returns:
            bool: ``True`` if a row was actually deleted.
        """
        result = self._conn().execute(
            "DELETE FROM commitments WHERE id = %s AND agent_id = %s",
            (commitment_id, agent_id),
        )
        return result.rowcount > 0

    def list_pending_proposals(self, agent_id: str) -> list[dict]:
        """List every not-yet-reviewed proposal for one agent, oldest first.

        Stage One Phase 5, slice C: the source list
        ``memory/consolidation.py``'s ``rank_proposals`` scores and orders
        into a review queue. Only ``status = 'pending'`` rows are returned —
        a proposal already ``promoted``/``rejected``/``superseded`` (a later
        slice's review flow assigns these) has already been decided and
        shouldn't be re-offered for review.

        Args:
            agent_id: Which agent's proposals to list.

        Returns:
            list[dict]: ``{"id", "job_id", "agent_id", "claim_text",
                "created_at"}`` dicts, ordered by ``id`` (oldest first).
        """
        rows = self._conn().execute(
            """
            SELECT id, job_id, agent_id, claim_text, created_at
            FROM memory_proposals
            WHERE agent_id = %s AND status = 'pending'
            ORDER BY id
            """,
            (agent_id,),
        ).fetchall()
        return [
            {"id": r[0], "job_id": r[1], "agent_id": r[2], "claim_text": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_proposal(self, proposal_id: int) -> dict | None:
        """Look up one proposal by id.

        Stage One Phase 5, slice C: ``MemoryConsolidator.preview()`` needs a
        single proposal's claim text and status without listing every
        pending proposal first.

        Returns:
            dict | None: ``{"id", "job_id", "agent_id", "claim_text",
                "created_at", "status", "rejected_reason"}``, or ``None``
                if no proposal with this id exists.
        """
        row = self._conn().execute(
            """
            SELECT id, job_id, agent_id, claim_text, created_at, status, rejected_reason
            FROM memory_proposals
            WHERE id = %s
            """,
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "job_id": row[1], "agent_id": row[2], "claim_text": row[3],
            "created_at": row[4], "status": row[5], "rejected_reason": row[6],
        }
