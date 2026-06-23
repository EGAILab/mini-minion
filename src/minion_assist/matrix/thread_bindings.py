"""Matrix thread binding manager.

Maps Matrix thread root event IDs to isolated AgentSession keys so a threaded
conversation always continues in the same agent session regardless of how many
unrelated messages arrive in the same room.

Bindings are stored in SQLite and evicted when:
- ``last_activity_at`` is older than ``idle_hours`` (idle timeout), OR
- ``created_at`` is older than ``max_age_hours`` (absolute max age).

Requires ``aiosqlite``.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from .config import MatrixThreadBindingsConfig

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS thread_bindings (
    thread_event_id TEXT PRIMARY KEY,
    room_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    session_key     TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    last_activity_at INTEGER NOT NULL
)
"""


class MatrixThreadBindingManager:
    """SQLite-backed thread-to-session-key mapping with idle/max-age eviction.

    Args:
        db_path: Path to the SQLite database file.
        config:  Thread bindings settings from config.
    """

    def __init__(self, db_path: Path, config: MatrixThreadBindingsConfig) -> None:
        self._db_path = db_path
        self._config = config
        self._conn = None

    async def start(self) -> None:
        """Open the database and create the schema."""
        import aiosqlite  # noqa: PLC0415

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        # _CREATE_TABLE uses IF NOT EXISTS so this is safe to run on every startup.
        await self._conn.execute(_CREATE_TABLE)
        await self._conn.commit()
        # Remove stale bindings so old threads don't use up memory/storage.
        await self.evict_expired()

    async def stop(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get_or_create_session_key(
        self, thread_event_id: str, room_id: str, agent_id: str
    ) -> str:
        """Return the session key bound to ``thread_event_id``, creating one if needed.

        Args:
            thread_event_id: Root event ID of the Matrix thread.
            room_id:         Room containing the thread.
            agent_id:        Agent that should handle this thread.

        Returns:
            A stable session key string unique to this thread.
        """
        if self._conn is None:
            raise RuntimeError("MatrixThreadBindingManager.start() has not been called.")
        now = int(time.time())
        # Look up an existing binding for this thread root event ID.
        async with self._conn.execute(
            "SELECT session_key FROM thread_bindings WHERE thread_event_id = ?",
            (thread_event_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            # Thread already has a session — refresh its idle timer and reuse.
            await self.touch(thread_event_id)
            return row[0]
        # No binding yet: create a new unique session key for this thread.
        # uuid4().hex is a random 32-character hex string — extremely unlikely
        # to collide, and human-readable enough to appear in debug logs.
        session_key = f"matrix-thread-{uuid.uuid4().hex}"
        await self._conn.execute(
            "INSERT INTO thread_bindings "
            "(thread_event_id, room_id, agent_id, session_key, created_at, last_activity_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_event_id, room_id, agent_id, session_key, now, now),
        )
        await self._conn.commit()
        return session_key

    async def touch(self, thread_event_id: str) -> None:
        """Update ``last_activity_at`` for an existing binding."""
        if self._conn is None:
            return
        now = int(time.time())
        await self._conn.execute(
            "UPDATE thread_bindings SET last_activity_at = ? WHERE thread_event_id = ?",
            (now, thread_event_id),
        )
        await self._conn.commit()

    async def evict_expired(self) -> None:
        """Delete bindings that have exceeded idle or max-age limits."""
        if self._conn is None:
            return
        now = int(time.time())
        # Convert hours → seconds for the SQL comparison.
        idle_cutoff = now - int(self._config.idle_hours * 3600)
        age_cutoff = now - int(self._config.max_age_hours * 3600)
        # A binding is evicted if either:
        #   - It hasn't been touched for more than idle_hours (stale thread), OR
        #   - It was created more than max_age_hours ago (absolute expiry).
        await self._conn.execute(
            "DELETE FROM thread_bindings WHERE last_activity_at < ? OR created_at < ?",
            (idle_cutoff, age_cutoff),
        )
        await self._conn.commit()
