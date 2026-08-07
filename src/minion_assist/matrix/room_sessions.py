"""Matrix room session binding manager.

Maps ``(room_id, agent_id)`` to a stable ``session_id`` so every Matrix room
gets its own isolated conversation with an agent (MEM-GAP-001), rather than
every room routed to that agent sharing one in-memory session.

This deployment treats a Matrix room as a persistent, deliberately-created
topic (e.g. a "Movie" room, always exactly Ada + one person) — not an
ephemeral Matrix *thread* — so bindings here never expire; a room's session
lasts as long as the room does. Replaces the earlier
``thread_bindings.py``/``MatrixThreadBindingManager``, which bound Matrix
*threads* (a feature this deployment doesn't use — see
``docs/adr/0006-room-scoped-matrix-sessions.md``) and whose resulting key
was never actually wired into session selection.

Bindings are stored in SQLite (via ``aiosqlite``, matching every other
Matrix-channel sidecar database).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS room_sessions (
    room_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (room_id, agent_id)
)
"""


class MatrixRoomSessionManager:
    """SQLite-backed ``(room_id, agent_id) -> session_id`` mapping.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = None

    async def start(self) -> None:
        """Open the database and create the schema."""
        import aiosqlite  # noqa: PLC0415

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        # _CREATE_TABLE uses IF NOT EXISTS so this is safe to run on every startup.
        await self._conn.execute(_CREATE_TABLE)
        await self._conn.commit()

    async def stop(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get_or_create_session_id(self, room_id: str, agent_id: str) -> str:
        """Return the session_id bound to ``(room_id, agent_id)``, creating one if needed.

        Args:
            room_id:  The Matrix room this message arrived in.
            agent_id: The agent handling this room.

        Returns:
            A stable session_id (a UUID string, the same format
            :class:`~minion_assist.session.store.SessionStore` and
            ``short_term.py`` already use elsewhere) unique to this room and
            agent — safe to pass straight into
            :class:`~minion_assist.agents.session.AgentSession`.
        """
        if self._conn is None:
            raise RuntimeError("MatrixRoomSessionManager.start() has not been called.")
        async with self._conn.execute(
            "SELECT session_id FROM room_sessions WHERE room_id = ? AND agent_id = ?",
            (room_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            return row[0]
        # No binding yet for this room+agent: mint a fresh session, never one
        # inherited from the old shared/default session — a new room starts
        # with a clean slate rather than silently picking up unrelated
        # history mixed in from every other room this agent has ever seen.
        session_id = str(uuid.uuid4())
        await self._conn.execute(
            "INSERT INTO room_sessions (room_id, agent_id, session_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (room_id, agent_id, session_id, int(time.time())),
        )
        await self._conn.commit()
        return session_id

    async def rebind(self, room_id: str, agent_id: str, session_id: str) -> None:
        """Point ``(room_id, agent_id)`` at a different, already-existing ``session_id``.

        R2-GAP-004: ``/session <arg>`` switches a room's live
        :class:`~minion_assist.agents.session.AgentSession` to a different
        session in place (``switch_session()``), but that's purely
        in-memory — without also updating the row this method writes, the
        switch would silently revert to whatever was last persisted the
        next time the bot restarts and re-resolves this room's binding via
        :meth:`get_or_create_session_id`.

        Unlike that method, this always overwrites — there is no
        get-or-create branch, since the caller already knows exactly which
        session_id it wants bound.

        Args:
            room_id: The Matrix room whose binding should change.
            agent_id: The agent this room routes to.
            session_id: The session_id to bind, replacing whatever was
                bound before.
        """
        if self._conn is None:
            raise RuntimeError("MatrixRoomSessionManager.start() has not been called.")
        await self._conn.execute(
            "INSERT INTO room_sessions (room_id, agent_id, session_id, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (room_id, agent_id) DO UPDATE SET session_id = excluded.session_id",
            (room_id, agent_id, session_id, int(time.time())),
        )
        await self._conn.commit()
