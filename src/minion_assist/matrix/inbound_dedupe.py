"""Inbound event deduplication for the Matrix channel.

Matrix clients may replay events on reconnect.  This module tracks seen event
IDs in a SQLite database so duplicate events are silently discarded.

Entries older than 24 hours are pruned on startup to keep the database small.

Requires ``aiosqlite`` (installed with the ``matrix`` optional dependency group).
"""

from __future__ import annotations

import time
from pathlib import Path

_PRUNE_AGE_SECONDS = 86_400  # 24 hours


class MatrixInboundDeduper:
    """SQLite-backed seen-event-ID tracker.

    Args:
        db_path: Path to the SQLite database file.  Created on first start.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = None

    async def start(self) -> None:
        """Open the database connection and create the schema if needed."""
        # aiosqlite is an async wrapper around Python's built-in sqlite3.
        # We import it lazily here because it's an optional dependency only
        # needed when the Matrix channel is actually enabled.
        import aiosqlite  # noqa: PLC0415 — optional dependency, imported lazily

        # Create the workspace/matrix/ directory if it doesn't exist yet.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        # Create the table only if it doesn't already exist — safe to call on
        # every startup.  event_id is the PRIMARY KEY so duplicates are rejected.
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_events "
            "(event_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)"
        )
        await self._conn.commit()
        # Remove old entries on startup so the database doesn't grow forever.
        await self._prune()

    async def stop(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def is_seen(self, event_id: str) -> bool:
        """Check whether ``event_id`` has been seen before and mark it seen.

        The check-and-mark is done in a single SQL statement (INSERT OR IGNORE)
        so there is no race condition between check and write.

        Args:
            event_id: Matrix event ID from the sync response.

        Returns:
            True if the event was already in the database (duplicate).
            False if it was newly inserted (first time seen).
        """
        if self._conn is None:
            raise RuntimeError("MatrixInboundDeduper.start() has not been called.")
        now = int(time.time())
        # INSERT OR IGNORE atomically tries to insert the event ID.
        # If it already exists (PRIMARY KEY conflict), the insert is silently
        # skipped and rowcount == 0.  This avoids a separate SELECT + INSERT
        # which would have a race condition between the two queries.
        async with self._conn.execute(
            "INSERT OR IGNORE INTO seen_events (event_id, seen_at) VALUES (?, ?)",
            (event_id, now),
        ) as cursor:
            inserted = cursor.rowcount > 0
        await self._conn.commit()
        # If nothing was inserted, this event was already seen → it's a duplicate.
        return not inserted

    async def _prune(self) -> None:
        """Delete entries older than 24 hours."""
        cutoff = int(time.time()) - _PRUNE_AGE_SECONDS
        await self._conn.execute("DELETE FROM seen_events WHERE seen_at < ?", (cutoff,))
        await self._conn.commit()
