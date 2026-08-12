"""Tests for tests/conftest.py's DB isolation setup (R2-GAP-015 follow-up).

Covers the fail-closed fix: pytest_configure() must abort the whole test
session (via pytest.exit) when the database is reachable but isolation
schema creation itself fails, rather than silently leaving
minion_assist.config.database unpatched — which would let every DB-gated
test run for real against whatever database config.json points at. See
conftest.py's module docstring for the incident this guards against.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests import conftest as _conftest_mod


def test_pytest_configure_exits_when_schema_creation_fails_on_reachable_db(monkeypatch):
    """DB reachable but CREATE SCHEMA fails -> abort the session, don't fall through."""
    monkeypatch.setattr(_conftest_mod, "_schema_name", None)

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = RuntimeError("permission denied to create schema")

    import minion_assist.config as config_mod
    original_database = config_mod.database

    with patch("psycopg.connect", return_value=fake_conn):
        with pytest.raises(pytest.exit.Exception):
            _conftest_mod.pytest_configure(config=None)

    # Must not have patched config.database on the failure path — a caller
    # that ignored the Exit exception must still see the untouched config,
    # never a half-applied isolation URL.
    assert config_mod.database is original_database
    assert _conftest_mod._schema_name is None
    fake_conn.close.assert_called_once()


def test_pytest_configure_silently_returns_when_db_unreachable(monkeypatch):
    """DB unreachable entirely -> unchanged pre-existing behavior: silent no-op,
    no exception, config.database left untouched (every DB-gated test's own
    reachability check handles the skip)."""
    monkeypatch.setattr(_conftest_mod, "_schema_name", None)

    import minion_assist.config as config_mod
    original_database = config_mod.database

    with patch("psycopg.connect", side_effect=ConnectionRefusedError("no server")):
        _conftest_mod.pytest_configure(config=None)  # must not raise

    assert config_mod.database is original_database
    assert _conftest_mod._schema_name is None


def test_pytest_configure_patches_database_on_success(monkeypatch):
    """Happy path: schema creation succeeds -> config.database is repointed at
    the isolated schema and _schema_name is recorded for pytest_unconfigure."""
    monkeypatch.setattr(_conftest_mod, "_schema_name", None)

    fake_conn = MagicMock()  # execute() succeeds by default (no side_effect)

    import minion_assist.config as config_mod
    original_database = config_mod.database

    try:
        with patch("psycopg.connect", return_value=fake_conn):
            _conftest_mod.pytest_configure(config=None)

        assert _conftest_mod._schema_name is not None
        assert _conftest_mod._schema_name in config_mod.database.url
        assert config_mod.database is not original_database
    finally:
        config_mod.database = original_database
        monkeypatch.setattr(_conftest_mod, "_schema_name", None)
