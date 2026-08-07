"""Tests for schema_migrations.py — run_migrations()/Migration (MEM-GAP-010).

Uses a small in-memory fake connection rather than a live PostgreSQL
instance: run_migrations() only ever calls conn.execute(sql, params) and
.fetchall() on the result, so a fake that understands exactly the three
statement shapes it issues (create ledger table, select applied versions,
insert a ledger row) is enough to exercise the real ledger logic —
apply/skip/checksum-mismatch/unknown-version — fast and without a database
dependency.
"""

from __future__ import annotations

import pytest

from minion_assist.schema_migrations import Migration, SchemaMigrationError, run_migrations


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """In-memory stand-in for an autocommit psycopg connection."""

    def __init__(self):
        self.ledger: list[tuple] = []  # (component, version, name, checksum, applied_at)
        self.executed: list[tuple] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        stripped = sql.strip()
        if stripped.startswith("CREATE TABLE IF NOT EXISTS schema_migrations"):
            return _FakeResult([])
        if stripped.startswith("SELECT version, checksum"):
            (component,) = params
            rows = [(v, c) for (comp, v, _n, c, _a) in self.ledger if comp == component]
            return _FakeResult(rows)
        if stripped.startswith("INSERT INTO schema_migrations"):
            self.ledger.append(tuple(params))
            return _FakeResult([])
        # Any other statement is a migration's own content SQL (CREATE
        # TABLE/ALTER TABLE for its own tables) — just record it, no-op.
        return _FakeResult([])


# ---------------------------------------------------------------------------
# Migration functions used as fixtures — module-level so inspect.getsource()
# has real, stable source text to hash (a locally-defined closure would work
# too, but module-level functions are the clearest stand-in for how
# SessionDB/PostgresMemoryIndex will actually define these).
# ---------------------------------------------------------------------------

def _migration_a_v1(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS a (id INT)")


def _migration_a_v2(conn):
    conn.execute("ALTER TABLE a ADD COLUMN IF NOT EXISTS name TEXT")


def _migration_a_v1_edited(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS a (id INT, extra TEXT)")


def _migration_b_v1(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS b (id INT)")


# ---------------------------------------------------------------------------
# Applying migrations
# ---------------------------------------------------------------------------

def test_run_migrations_applies_a_single_migration():
    conn = _FakeConn()

    applied = run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    assert applied == [1]
    assert len(conn.ledger) == 1
    assert conn.ledger[0][0] == "session_db"
    assert conn.ledger[0][1] == 1


def test_run_migrations_applies_migrations_in_version_order_even_if_list_is_unsorted():
    conn = _FakeConn()
    order: list[int] = []

    def _v1(c):
        order.append(1)

    def _v2(c):
        order.append(2)

    run_migrations(conn, "session_db", [Migration(2, "second", _v2), Migration(1, "first", _v1)])

    assert order == [1, 2]


def test_run_migrations_creates_the_ledger_table_right_after_acquiring_the_lock():
    conn = _FakeConn()

    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    # R2-GAP-008: the advisory lock is acquired before anything else,
    # including the ledger table itself.
    assert conn.executed[0][0].strip().startswith("SELECT pg_advisory_lock")
    assert conn.executed[1][0].strip().startswith("CREATE TABLE IF NOT EXISTS schema_migrations")


def test_run_migrations_second_call_applies_nothing_new():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    applied = run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    assert applied == []
    assert len(conn.ledger) == 1  # not duplicated


def test_run_migrations_applies_only_the_new_migration_on_a_later_call():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    applied = run_migrations(
        conn, "session_db",
        [Migration(1, "baseline", _migration_a_v1), Migration(2, "add name", _migration_a_v2)],
    )

    assert applied == [2]


def test_run_migrations_is_scoped_to_its_own_component():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    applied = run_migrations(conn, "memory_index", [Migration(1, "baseline", _migration_b_v1)])

    assert applied == [1]  # memory_index's version 1 is independent of session_db's


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

def test_run_migrations_raises_when_an_applied_migrations_source_changed():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    with pytest.raises(SchemaMigrationError, match="checksum mismatch"):
        run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1_edited)])


def test_run_migrations_does_not_raise_when_source_is_unchanged():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])  # must not raise


def test_run_migrations_raises_on_a_ledger_version_this_code_does_not_define():
    conn = _FakeConn()
    run_migrations(
        conn, "session_db",
        [Migration(1, "baseline", _migration_a_v1), Migration(2, "add name", _migration_a_v2)],
    )

    with pytest.raises(SchemaMigrationError, match="doesn't define them"):
        run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])


# ---------------------------------------------------------------------------
# Checksum helper behavior (indirectly, via run_migrations' mismatch detection)
# ---------------------------------------------------------------------------

def test_two_migrations_with_identical_source_produce_the_same_checksum():
    # Renaming a migration (name field) without touching its apply function's
    # source must NOT be treated as a content change.
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "old name", _migration_a_v1)])

    run_migrations(conn, "session_db", [Migration(1, "new name", _migration_a_v1)])  # must not raise


# ---------------------------------------------------------------------------
# Advisory lock (R2-GAP-008)
# ---------------------------------------------------------------------------

def test_run_migrations_acquires_and_releases_the_advisory_lock_for_its_component():
    conn = _FakeConn()

    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])

    lock_calls = [e for e in conn.executed if "pg_advisory_lock" in e[0]]
    unlock_calls = [e for e in conn.executed if "pg_advisory_unlock" in e[0]]
    assert len(lock_calls) == 1
    assert lock_calls[0][1] == ("session_db",)
    assert len(unlock_calls) == 1
    assert unlock_calls[0][1] == ("session_db",)


def test_run_migrations_releases_the_lock_even_when_a_migration_raises():
    conn = _FakeConn()

    def _boom(c):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_migrations(conn, "session_db", [Migration(1, "baseline", _boom)])

    unlock_calls = [e for e in conn.executed if "pg_advisory_unlock" in e[0]]
    assert len(unlock_calls) == 1


def test_run_migrations_locks_are_scoped_independently_per_component():
    conn = _FakeConn()
    run_migrations(conn, "session_db", [Migration(1, "baseline", _migration_a_v1)])
    run_migrations(conn, "memory_index", [Migration(1, "baseline", _migration_b_v1)])

    lock_calls = [e for e in conn.executed if "pg_advisory_lock" in e[0]]
    components_locked = {params[0] for _sql, params in lock_calls}
    assert components_locked == {"session_db", "memory_index"}
