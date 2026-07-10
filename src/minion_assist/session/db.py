"""PostgreSQL-backed session and message store with FTS via tsvector.

Optional — disabled when DATABASE_URL is not set.  Call SessionDB(url) directly
or let minion.py create it from config; when unavailable the rest of the code
treats db=None and silently skips persistence.

Schema (created automatically on first connect):
  sessions(id, agent_id, source, started_at, last_active, turn_count, title, parent_id)
  messages(id BIGSERIAL, session_id, role, content, tool_name, timestamp,
           search_vector tsvector GENERATED ALWAYS AS ...)
  message_embeddings(message_id, embedding vector(N))  — only when pgvector available
"""
from __future__ import annotations

import threading
import time
from typing import Any

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
        """Insert a message row; returns the new message id."""
        row = self._conn().execute(
            """
            INSERT INTO messages (session_id, role, content, tool_name, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (session_id, role, content, tool_name, timestamp or time.time()),
        ).fetchone()
        return row[0] if row else -1

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

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def replay_jsonl(
        self,
        short_term: object,
        agent_ids: list[str],
    ) -> int:
        """One-time migration: replay existing JSONL session files into the DB.

        Skips sessions that already have rows in the messages table.
        Returns total number of messages inserted.
        """
        import json  # noqa: PLC0415

        total = 0
        for agent_id in agent_ids:
            for path in short_term.list_sessions(agent_id):
                session_id = path.stem
                # Skip sessions already in DB
                count = self._conn().execute(
                    "SELECT count(*) FROM messages WHERE session_id = %s", (session_id,)
                ).fetchone()[0]
                if count:
                    continue

                mtime = path.stat().st_mtime
                self.upsert_session(session_id, agent_id, started_at=mtime)

                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = _msg_text(msg)
                    tool_name = msg.get("name") or msg.get("tool_name")
                    role = msg.get("role", "")
                    if not role:
                        continue
                    self.add_message(session_id, role, content, tool_name=tool_name, timestamp=mtime)
                    total += 1
        return total
