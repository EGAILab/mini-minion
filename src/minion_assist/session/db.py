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

# Thread-local connection cache — one psycopg connection per OS thread.
_local = threading.local()


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
        # pgvector extension (optional — silently skipped if not available)
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._has_vector = True
        except Exception:
            pass

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
        if self._has_vector:
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
                id         BIGSERIAL PRIMARY KEY,
                job_id     BIGINT NOT NULL,
                agent_id   TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending'
            )
        """)
        # status added in Stage One Phase 5, slice B — ADD COLUMN IF NOT
        # EXISTS so a database that already has this table from Phase 2
        # picks it up without a manual migration. Values: "pending" (not
        # yet reviewed), "promoted"/"rejected"/"superseded" (Phase 5 slice
        # D's consolidation review will assign these; nothing does yet).
        conn.execute(
            "ALTER TABLE memory_proposals ADD COLUMN IF NOT EXISTS "
            "status TEXT NOT NULL DEFAULT 'pending'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS proposals_agent_idx ON memory_proposals (agent_id)"
        )

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

    def get_sessions_by_ids(self, session_ids: list[str]) -> dict[str, dict]:
        if not session_ids:
            return {}
        rows = self._conn().execute(
            """
            SELECT id, agent_id, title, started_at, last_active, turn_count
            FROM sessions WHERE id = ANY(%s)
            """,
            (session_ids,),
        ).fetchall()
        return {
            r[0]: {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4], "turn_count": r[5],
            }
            for r in rows
        }

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """Return most recent sessions, newest first."""
        rows = self._conn().execute(
            """
            SELECT s.id, s.agent_id, s.title, s.started_at, s.last_active, s.turn_count,
                (SELECT content FROM messages
                 WHERE session_id = s.id AND role = 'user'
                 ORDER BY id LIMIT 1) AS first_message
            FROM sessions s
            ORDER BY s.last_active DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0], "agent_id": r[1], "title": r[2],
                "started_at": r[3], "last_active": r[4],
                "turn_count": r[5], "first_message": r[6],
            }
            for r in rows
        ]

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

    def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        """FTS search across all message content. Returns ranked matches."""
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
            WHERE m.search_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, query, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "role": r[2],
                "content": r[3], "tool_name": r[4], "timestamp": r[5],
                "snippet": r[6], "rank": float(r[7]),
            }
            for r in rows
        ]

    def get_messages_around(
        self, session_id: str, anchor_id: int, window: int = 5
    ) -> list[dict]:
        """Return messages in [anchor_id-window, anchor_id+window] for a session."""
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
        self, session_id: str, n: int = 3
    ) -> tuple[list[dict], list[dict]]:
        """Return first n and last n user/assistant messages for a session."""
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
