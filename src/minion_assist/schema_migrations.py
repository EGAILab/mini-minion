"""Versioned, checksum-verified PostgreSQL schema evolution (MEM-GAP-010).

Before this module, ``session/db.py``'s ``SessionDB`` and
``memory/postgres_index.py``'s ``PostgresMemoryIndex`` each bootstrapped
their ~21 combined tables via unconditional ``CREATE TABLE IF NOT EXISTS``/
``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` statements, re-run in full on
every process startup. That's safe for purely additive changes, but has no
version ledger, no ordering guarantee, no way to detect a partially-applied
migration, and no mechanism at all for a non-additive change (rename, drop,
type change, backfill) — see
``minion-assist-docs/improve/openclaw-memory-gap-analysis.md``'s
MEM-GAP-010.

This module doesn't replace that bootstrap SQL — it wraps it. Each
class's entire existing ``_ensure_schema()`` body becomes a single
"baseline" :class:`Migration` (version 1), recorded in a new
``schema_migrations`` ledger table the first time it runs. An existing
database just gets retroactively marked "already at version 1" — every
statement in the baseline was already idempotent, so replaying it is
harmless. Going forward, any *new* schema change is a new, higher-numbered
:class:`Migration` appended to that component's list, never an edit to an
already-applied one.

Checksum-verified, fail actionably
------------------------------------
Every applied migration's source code (via ``inspect.getsource`` on its
``apply`` function, not just its output) is hashed and stored alongside its
version in the ledger. :func:`run_migrations` re-hashes on every startup and
raises :class:`SchemaMigrationError` — refusing to start, not just warning
— if an already-applied migration's source no longer matches (someone
edited migration history instead of adding a new migration), or if the
ledger contains a version this code doesn't define at all (the database was
touched by a newer version of the app). Silently continuing with an
unverifiable schema is exactly the failure mode this exists to prevent.

Shared ledger, per-component versioning
-------------------------------------------
One ``schema_migrations`` table, keyed by ``(component, version)`` — not
two separate tables — since it's the same physical database either way.
``SessionDB`` uses ``component="session_db"``, ``PostgresMemoryIndex`` uses
``component="memory_index"``; each has its own independent version
sequence starting at 1.

Talks to
--------
- ``session/db.py`` — ``SessionDB._ensure_schema()`` calls
  :func:`run_migrations` with its baseline migration.
- ``memory/postgres_index.py`` — ``PostgresMemoryIndex._ensure_schema()``
  does the same for its own tables.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass

# Ledger table itself — created (idempotently) by run_migrations() before it
# does anything else, so it never depends on being part of any component's
# own migration list.
_LEDGER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    component   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    applied_at  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (component, version)
)
"""


class SchemaMigrationError(RuntimeError):
    """Raised when a component's migration ledger can't be trusted.

    Two distinct causes, both deliberately fatal (refuse to start) rather
    than a warning:

    - An already-applied migration's source no longer matches the checksum
      recorded when it ran — its code was edited after the fact instead of
      adding a new migration.
    - The ledger records a migration version this code doesn't define at
      all — this database was touched by a newer version of the app than
      the one currently running.
    """


@dataclass(frozen=True)
class Migration:
    """One versioned, checksummed schema change for one component.

    Args:
        version: Strictly increasing per component, starting at 1 — the
            order migrations are applied in, not just a label.
        name: Short human-readable description, shown in ledger rows and
            error messages.
        apply: Applies this migration's schema change, given the raw
            connection. Every statement inside must be idempotent
            (``CREATE TABLE IF NOT EXISTS``, ``ADD COLUMN IF NOT EXISTS``,
            etc.) so replaying a migration that already fully or partially
            applied (e.g. before a crash) is always safe.
    """

    version: int
    name: str
    apply: Callable[[object], None]


def _checksum(fn: Callable) -> str:
    """Hash a migration's actual source code, not its behavior/output.

    Hashing source (via ``inspect.getsource``) rather than, say, the
    resulting table structure catches *any* edit to an already-applied
    migration's body — including one that looks harmless (reordering
    statements, tweaking a comment) — since the whole point is that
    migration history must never be edited in place, only appended to.
    """
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def run_migrations(conn, component: str, migrations: list[Migration]) -> list[int]:
    """Bring one component's schema up to date, verifying ledger history as it goes.

    R2-GAP-008: the whole body runs under a session-level PostgreSQL
    advisory lock keyed by ``component`` (``pg_advisory_lock(hashtext(component))``,
    acquired before even the ledger table is created and released in a
    ``finally``). Without it, two processes starting at once (e.g. the bot
    and a maintenance CLI invocation, or two bot instances during a
    deployment) could both read "version N missing" before either had
    applied it, then both call ``migration.apply(conn)`` and both try to
    insert the same ``(component, version)`` ledger row — one succeeds, the
    other hits the ledger's primary-key conflict having already run
    ``apply()`` a second time, which is only actually safe because every
    migration's own SQL is independently idempotent (``IF NOT EXISTS``
    etc.) — this lock removes the need to rely on that as the *only*
    protection. A second caller simply blocks on ``pg_advisory_lock`` until
    the first finishes, then finds the ledger already up to date and
    applies nothing.

    Args:
        conn: An open, autocommit database connection.
        component: This migration set's ledger key (e.g. ``"session_db"``).
        migrations: Every migration this version of the code knows about
            for this component, in any order (sorted internally by
            ``version``).

    Returns:
        list[int]: Versions actually applied during this call (empty if the
            component was already fully up to date).

    Raises:
        SchemaMigrationError: The ledger contains a version not present in
            ``migrations``, or an already-applied migration's checksum no
            longer matches its current source.
    """
    conn.execute("SELECT pg_advisory_lock(hashtext(%s)::bigint)", (component,))
    try:
        conn.execute(_LEDGER_TABLE_SQL)

        applied_checksums: dict[int, str] = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT version, checksum FROM schema_migrations WHERE component = %s",
                (component,),
            ).fetchall()
        }

        known_versions = {m.version for m in migrations}
        unknown = sorted(set(applied_checksums) - known_versions)
        if unknown:
            raise SchemaMigrationError(
                f"{component}: the database has migration(s) {unknown} recorded as applied, "
                "but this version of minion-assist doesn't define them. Refusing to start — "
                "this usually means the database was touched by a newer version of the app."
            )

        newly_applied: list[int] = []
        for migration in sorted(migrations, key=lambda m: m.version):
            checksum = _checksum(migration.apply)
            if migration.version in applied_checksums:
                if applied_checksums[migration.version] != checksum:
                    raise SchemaMigrationError(
                        f"{component}: migration {migration.version} ('{migration.name}') has "
                        "changed since it was applied — checksum mismatch. Editing an "
                        "already-applied migration is unsafe; add a new migration instead of "
                        "modifying history."
                    )
                continue
            migration.apply(conn)
            conn.execute(
                "INSERT INTO schema_migrations (component, version, name, checksum, applied_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (component, migration.version, migration.name, checksum, time.time()),
            )
            newly_applied.append(migration.version)
        return newly_applied
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s)::bigint)", (component,))
