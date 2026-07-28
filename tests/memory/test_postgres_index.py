"""Tests for memory/postgres_index.py: PostgresMemoryIndex (Stage One Phase 3, slice A).

Requires a live PostgreSQL instance matching config.json's configured URL
(same instance tests/test_session_db.py uses) — the whole module is skipped
if one isn't reachable. Every test uses a fresh, random agent_id and cleans
up its own rows in an autouse fixture, so tests never collide with each
other or leave rows behind in the shared dev database.
"""

from __future__ import annotations

import uuid

import pytest

_DB_URL = "postgresql://minion:minion@localhost:5433/minion_assist"

try:
    import psycopg as _psycopg

    _test_conn = _psycopg.connect(_DB_URL, connect_timeout=2)
    _test_conn.close()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _DB_AVAILABLE, reason="requires a live PostgreSQL instance")

from minion_assist.memory.postgres_index import PostgresMemoryIndex  # noqa: E402


@pytest.fixture
def index():
    return PostgresMemoryIndex(_DB_URL)


@pytest.fixture
def agent_id():
    """A fresh, unique agent_id per test so tests never collide."""
    return f"test-{uuid.uuid4()}"


@pytest.fixture(autouse=True)
def _cleanup_after(index, agent_id):
    yield
    conn = index._conn()
    conn.execute("DELETE FROM memory_chunks WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_files WHERE agent_id = %s", (agent_id,))


# ---------------------------------------------------------------------------
# reindex_file
# ---------------------------------------------------------------------------

def test_reindex_file_writes_chunks_and_ledger_row(index, agent_id):
    count = index.reindex_file(agent_id, "MEMORY.md", "durable", "Some memory content.")

    assert count == 1
    assert index.chunk_count(agent_id) == 1
    assert index.indexed_files(agent_id) == ["MEMORY.md"]


def test_reindex_file_replaces_previous_chunks_for_the_same_file(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "Original short content.")
    assert index.chunk_count(agent_id) == 1

    lines = [
        f"filler line {i} with extra padding words to push past the token budget"
        for i in range(200)
    ]
    big_content = "\n".join(lines)
    count = index.reindex_file(agent_id, "MEMORY.md", "durable", big_content)

    # Old chunk is gone, replaced entirely by the new content's chunks —
    # never accumulates stale rows from a prior version of the same file.
    assert index.chunk_count(agent_id) == count
    assert count > 1


def test_reindex_file_with_empty_content_writes_zero_chunks_but_updates_ledger(index, agent_id):
    count = index.reindex_file(agent_id, "empty.md", "durable", "   ")

    assert count == 0
    assert index.chunk_count(agent_id) == 0
    assert index.indexed_files(agent_id) == ["empty.md"]


def test_reindex_file_stores_source_kind_and_heading_path(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "# A Heading\nSome body text.")

    row = index._conn().execute(
        "SELECT source_kind, heading_path, start_line, end_line "
        "FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()

    assert row[0] == "durable"
    assert row[1] == "A Heading"
    assert row[2] == 1
    assert row[3] == 2


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------

def test_remove_file_deletes_chunks_and_ledger_row(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "content")
    index.remove_file(agent_id, "MEMORY.md")

    assert index.chunk_count(agent_id) == 0
    assert index.indexed_files(agent_id) == []


def test_remove_file_is_a_no_op_for_a_file_never_indexed(index, agent_id):
    index.remove_file(agent_id, "never-existed.md")  # must not raise
    assert index.chunk_count(agent_id) == 0


# ---------------------------------------------------------------------------
# rebuild_agent
# ---------------------------------------------------------------------------

def test_rebuild_agent_indexes_every_file_in_the_listing(index, agent_id):
    files = [
        ("durable", "MEMORY.md", "Durable memory content."),
        ("daily", "memory/2026-07-20.md", "## 2026-07-20\n\n- 09:00: did a thing"),
        ("import", "memory/imports/_auto_extracted.md", "User prefers dark mode."),
    ]

    total = index.rebuild_agent(agent_id, files)

    assert total == 3
    assert sorted(index.indexed_files(agent_id)) == sorted(rel for _k, rel, _c in files)


def test_rebuild_agent_removes_files_no_longer_present(index, agent_id):
    index.rebuild_agent(agent_id, [("durable", "MEMORY.md", "content one")])
    assert index.indexed_files(agent_id) == ["MEMORY.md"]

    # Second rebuild's listing no longer includes MEMORY.md (as if it were
    # deleted from disk) — it must disappear from the index too.
    index.rebuild_agent(agent_id, [("daily", "memory/2026-07-20.md", "daily content")])

    assert index.indexed_files(agent_id) == ["memory/2026-07-20.md"]


def test_rebuild_agent_with_empty_listing_clears_the_agents_index(index, agent_id):
    index.rebuild_agent(agent_id, [("durable", "MEMORY.md", "content")])
    index.rebuild_agent(agent_id, [])

    assert index.indexed_files(agent_id) == []
    assert index.chunk_count(agent_id) == 0


def test_rebuild_agent_does_not_touch_another_agents_files(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.reindex_file(other_agent, "MEMORY.md", "durable", "other agent's content")

    index.rebuild_agent(agent_id, [("durable", "MEMORY.md", "this agent's content")])

    try:
        assert index.indexed_files(other_agent) == ["MEMORY.md"]
        assert index.chunk_count(other_agent) == 1
    finally:
        index.remove_file(other_agent, "MEMORY.md")
