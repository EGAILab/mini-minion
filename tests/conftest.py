"""Root pytest configuration — isolated-schema database test hermeticity (R2-GAP-015).

Before this, every DB-backed test connected either to a hardcoded
``_DB_URL`` literal (``test_session_db.py``, ``memory/test_postgres_index.py``,
``memory/test_reconciliation_scheduler.py``, ``test_tools_session_search.py``)
or to whatever the real ``config.json`` on this machine configures
(``test_minion.py``'s live-wiring tests, via ``config.database.url``) — both
pointed at the *same* real dev PostgreSQL database, shared with any other
concurrently-running process (a real bot instance, a concurrent test run).
Global, unscoped queries (``claim_next_capture_job()`` and friends — see
``session/db.py``'s "FOR UPDATE SKIP LOCKED" claim methods, none of which
filter by agent/session) could pick up rows left behind by *anything* else
touching that database, not just this test run — directly observed during
the round-two remediation session: ``test_minion.py``'s own live-wiring
tests left real pending ``memory_capture_jobs`` rows behind for a
"researcher" agent, which then made an unrelated, pre-existing
``queue_lag_summary`` test intermittently claim the wrong job.

``pytest_configure`` runs once, before any test module is collected
(imported) — creates a fresh, uniquely-named PostgreSQL schema on the same
dev database/server, then monkeypatches the *already-loaded*
``minion_assist.config`` module's ``database`` attribute so every
subsequent ``SessionDB``/``PostgresMemoryIndex`` this test session
constructs — whether via a test file's own ``_DB_URL`` (now sourced from
this same patched config, not a separate hardcoded literal) or via
``config.database.url`` directly (``test_minion.py``) — connects with that
schema first in its ``search_path``. Every unqualified table/type reference
this codebase already uses (no test or application code needed to change
beyond where ``_DB_URL`` gets its value) transparently resolves inside the
isolated schema instead of the shared ``public`` one; ``public`` stays
second in the search path purely so already-installed extension types
(``vector``) still resolve without needing to reinstall the extension
per schema.

``pytest_unconfigure`` drops the schema (``CASCADE``) once the whole test
session finishes — nothing this session created durably touches the shared
database at all.

If the dev database is unreachable when ``pytest_configure`` runs, this
is a silent no-op: ``minion_assist.config.database`` is left exactly as
config.json says, and every DB-gated test's own ``_DB_AVAILABLE``-style
skip marker (already established convention — see ``test_session_db.py``'s
module docstring) handles "no database" exactly as it always did.
"""

from __future__ import annotations

import uuid

_BASE_DB_URL = "postgresql://minion:minion@localhost:5433/minion_assist"

# Set by pytest_configure, read by pytest_unconfigure — module-level state
# is fine here since pytest only ever runs one test session per process.
_schema_name: str | None = None


def pytest_configure(config) -> None:
    global _schema_name

    try:
        import psycopg
    except ImportError:
        return  # psycopg not installed -- DB-gated tests will skip themselves anyway

    candidate_schema = f"pytest_{uuid.uuid4().hex[:12]}"
    try:
        conn = psycopg.connect(_BASE_DB_URL, autocommit=True, connect_timeout=2)
        conn.execute(f'CREATE SCHEMA "{candidate_schema}"')
        conn.close()
    except Exception:
        return  # dev database unreachable -- leave config.database untouched

    _schema_name = candidate_schema
    isolated_url = (
        f"{_BASE_DB_URL}?options=-c%20search_path%3D{_schema_name}%2Cpublic"
    )

    import minion_assist.config as _config_mod  # noqa: PLC0415

    _config_mod.database = _config_mod.DatabaseConfig(url=isolated_url)


def pytest_unconfigure(config) -> None:
    if _schema_name is None:
        return
    try:
        import psycopg  # noqa: PLC0415

        conn = psycopg.connect(_BASE_DB_URL, autocommit=True, connect_timeout=2)
        conn.execute(f'DROP SCHEMA "{_schema_name}" CASCADE')
        conn.close()
    except Exception:
        pass
