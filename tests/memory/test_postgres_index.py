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
def vector_index():
    """A PostgresMemoryIndex with an embedding dimension configured (Stage One Phase 4, slice A)."""
    return PostgresMemoryIndex(_DB_URL, embedding_dimensions=3)


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
    conn.execute("DELETE FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_files_shadow WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_pins WHERE agent_id = %s", (agent_id,))


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


def test_remove_file_also_clears_a_pin(index, agent_id):
    index.reindex_file(agent_id, "memory/topics/goal.md", "durable", "content")
    index.pin_file(agent_id, "memory/topics/goal.md")

    index.remove_file(agent_id, "memory/topics/goal.md")

    assert index.is_pinned(agent_id, "memory/topics/goal.md") is False
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


# ---------------------------------------------------------------------------
# reconcile_agent
# ---------------------------------------------------------------------------

def test_reconcile_agent_indexes_new_files(index, agent_id):
    touched = index.reconcile_agent(agent_id, [("durable", "MEMORY.md", "new content")])

    assert touched == 1
    assert index.indexed_files(agent_id) == ["MEMORY.md"]


def test_reconcile_agent_is_a_no_op_when_nothing_changed(index, agent_id):
    files = [("durable", "MEMORY.md", "stable content")]
    index.reconcile_agent(agent_id, files)

    touched = index.reconcile_agent(agent_id, files)  # same content, same hash

    assert touched == 0


def test_reconcile_agent_reindexes_only_the_file_whose_content_changed(index, agent_id):
    files = [
        ("durable", "MEMORY.md", "memory content"),
        ("durable", "memory/topics/goals.md", "goals content"),
    ]
    index.reconcile_agent(agent_id, files)

    changed = [
        ("durable", "MEMORY.md", "memory content"),  # unchanged
        ("durable", "memory/topics/goals.md", "updated goals content"),  # changed
    ]
    touched = index.reconcile_agent(agent_id, changed)

    assert touched == 1
    [chunk] = index._conn().execute(
        "SELECT content FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "memory/topics/goals.md"),
    ).fetchall()
    assert chunk[0] == "updated goals content"


def test_reconcile_agent_removes_files_no_longer_present(index, agent_id):
    index.reconcile_agent(agent_id, [("durable", "MEMORY.md", "content")])

    touched = index.reconcile_agent(agent_id, [])

    assert touched == 1
    assert index.indexed_files(agent_id) == []


def test_reconcile_agent_does_not_reindex_files_it_only_removes_or_only_adds(index, agent_id):
    index.reconcile_agent(agent_id, [
        ("durable", "MEMORY.md", "stays the same"),
        ("daily", "memory/2026-07-20.md", "will be removed"),
    ])

    touched = index.reconcile_agent(agent_id, [
        ("durable", "MEMORY.md", "stays the same"),
        ("import", "memory/imports/new.md", "brand new file"),
    ])

    # One removed (2026-07-20.md), one added (new.md), MEMORY.md untouched.
    assert touched == 2
    assert sorted(index.indexed_files(agent_id)) == ["MEMORY.md", "memory/imports/new.md"]


# ---------------------------------------------------------------------------
# force_rebuild_agent
# ---------------------------------------------------------------------------

def test_force_rebuild_agent_indexes_every_file(index, agent_id):
    files = [
        ("durable", "MEMORY.md", "Durable memory content."),
        ("import", "memory/imports/_auto_extracted.md", "User prefers dark mode."),
    ]

    total = index.force_rebuild_agent(agent_id, files)

    assert total == 2
    assert sorted(index.indexed_files(agent_id)) == sorted(rel for _k, rel, _c in files)


def test_force_rebuild_agent_replaces_previous_content(index, agent_id):
    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "original content")])
    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "replaced content")])

    [chunk] = index._conn().execute(
        "SELECT content FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "MEMORY.md"),
    ).fetchall()
    assert chunk[0] == "replaced content"
    assert index.chunk_count(agent_id) == 1  # old chunk didn't linger alongside the new one


def test_force_rebuild_agent_removes_files_no_longer_present(index, agent_id):
    index.force_rebuild_agent(agent_id, [
        ("durable", "MEMORY.md", "content"),
        ("daily", "memory/2026-07-20.md", "daily content"),
    ])

    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "content")])

    assert index.indexed_files(agent_id) == ["MEMORY.md"]


def test_force_rebuild_agent_clears_shadow_tables_after_a_successful_swap(index, agent_id):
    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "content")])

    conn = index._conn()
    shadow_chunks = conn.execute(
        "SELECT count(*) FROM memory_chunks_shadow WHERE agent_id = %s", (agent_id,)
    ).fetchone()
    shadow_files = conn.execute(
        "SELECT count(*) FROM memory_files_shadow WHERE agent_id = %s", (agent_id,)
    ).fetchone()
    assert shadow_chunks[0] == 0
    assert shadow_files[0] == 0


def test_force_rebuild_agent_raises_and_leaves_the_live_index_unchanged_on_failure(
    index, agent_id, monkeypatch
):
    # Establish a known-good live index first.
    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "original content")])

    def _boom(text, **kwargs):
        raise RuntimeError("chunker exploded")

    monkeypatch.setattr("minion_assist.memory.postgres_index.chunk_markdown", _boom)

    with pytest.raises(RuntimeError, match="chunker exploded"):
        index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "new content")])

    # The live index is untouched -- still serving the original content.
    [chunk] = index._conn().execute(
        "SELECT content FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "MEMORY.md"),
    ).fetchall()
    assert chunk[0] == "original content"


def test_force_rebuild_agent_does_not_touch_another_agents_files(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.force_rebuild_agent(other_agent, [("durable", "MEMORY.md", "other agent's content")])

    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "this agent's content")])

    try:
        assert index.indexed_files(other_agent) == ["MEMORY.md"]
    finally:
        index.remove_file(other_agent, "MEMORY.md")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_a_matching_chunk(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode in the editor.")

    results = index.search(agent_id, "dark mode")

    assert len(results) == 1
    assert results[0]["rel_path"] == "MEMORY.md"
    assert "dark mode" in results[0]["content"]
    assert results[0]["score"] > 0


def test_search_returns_empty_list_for_no_match(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode.")

    assert index.search(agent_id, "coffee preferences") == []


def test_search_restricts_to_one_corpus(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "shared keyword content")
    index.reindex_file(agent_id, "memory/imports/x.md", "import", "shared keyword content")

    durable_only = index.search(agent_id, "shared keyword", corpus="durable")

    assert len(durable_only) == 1
    assert durable_only[0]["source_kind"] == "durable"


def test_search_respects_max_results(index, agent_id):
    for i in range(5):
        index.reindex_file(agent_id, f"memory/topics/note-{i}.md", "durable", "shared keyword")

    results = index.search(agent_id, "shared keyword", max_results=2)

    assert len(results) == 2


def test_search_only_searches_the_given_agent(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.reindex_file(other_agent, "MEMORY.md", "durable", "shared keyword content")

    try:
        assert index.search(agent_id, "shared keyword") == []
    finally:
        index.remove_file(other_agent, "MEMORY.md")


# ---------------------------------------------------------------------------
# index_summary
# ---------------------------------------------------------------------------

def test_index_summary_for_agent_with_no_files(index, agent_id):
    summary = index.index_summary(agent_id)

    assert summary["total_chunks"] == 0
    assert summary["file_count"] == 0
    assert summary["by_corpus"] == {}
    assert summary["last_indexed_at"] is None


def test_index_summary_reports_counts_and_last_indexed_at(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "durable content")
    index.reindex_file(agent_id, "memory/imports/x.md", "import", "import content")

    summary = index.index_summary(agent_id)

    assert summary["file_count"] == 2
    assert summary["total_chunks"] == 2
    assert summary["by_corpus"] == {"durable": 1, "import": 1}
    assert summary["last_indexed_at"] is not None


# ---------------------------------------------------------------------------
# Embedding cache (Stage One Phase 4, slice A)
# ---------------------------------------------------------------------------

def test_has_vector_lane_is_false_without_configured_dimensions(index):
    assert index.has_vector_lane is False


def test_has_vector_lane_is_true_with_configured_dimensions(vector_index):
    assert vector_index.has_vector_lane is True


def test_cache_embedding_is_a_no_op_without_a_vector_lane(index):
    index.cache_embedding(999001, "test/model", "hash-a", [0.1, 0.2, 0.3])  # must not raise
    assert index.get_cached_embedding(999001, "test/model", "hash-a") is None


def test_get_cached_embedding_returns_none_for_a_never_cached_chunk(vector_index):
    assert vector_index.get_cached_embedding(999002, "test/model", "hash-a") is None


def test_cache_embedding_round_trips(vector_index):
    chunk_id = 999003
    try:
        vector_index.cache_embedding(chunk_id, "test/model", "hash-a", [0.1, 0.2, 0.3])

        result = vector_index.get_cached_embedding(chunk_id, "test/model", "hash-a")

        assert result is not None
        assert len(result) == 3
        assert result[0] == pytest.approx(0.1, abs=1e-4)
        assert result[1] == pytest.approx(0.2, abs=1e-4)
        assert result[2] == pytest.approx(0.3, abs=1e-4)
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE chunk_id = %s", (chunk_id,)
        )


def test_get_cached_embedding_misses_on_wrong_model_identity(vector_index):
    chunk_id = 999004
    try:
        vector_index.cache_embedding(chunk_id, "test/model-a", "hash-a", [0.1, 0.2, 0.3])

        assert vector_index.get_cached_embedding(chunk_id, "test/model-b", "hash-a") is None
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE chunk_id = %s", (chunk_id,)
        )


def test_get_cached_embedding_misses_on_wrong_content_hash(vector_index):
    chunk_id = 999005
    try:
        vector_index.cache_embedding(chunk_id, "test/model", "hash-a", [0.1, 0.2, 0.3])

        assert vector_index.get_cached_embedding(chunk_id, "test/model", "hash-stale") is None
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE chunk_id = %s", (chunk_id,)
        )


def test_cache_embedding_replaces_a_previous_value_for_the_same_chunk(vector_index):
    chunk_id = 999006
    try:
        vector_index.cache_embedding(chunk_id, "test/model", "hash-a", [0.1, 0.2, 0.3])
        vector_index.cache_embedding(chunk_id, "test/model", "hash-b", [0.4, 0.5, 0.6])

        assert vector_index.get_cached_embedding(chunk_id, "test/model", "hash-a") is None
        result = vector_index.get_cached_embedding(chunk_id, "test/model", "hash-b")
        assert result[0] == pytest.approx(0.4, abs=1e-4)
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE chunk_id = %s", (chunk_id,)
        )


# ---------------------------------------------------------------------------
# Pinning (Stage One Phase 4, slice B)
# ---------------------------------------------------------------------------

def test_is_pinned_is_false_for_a_never_pinned_file(index, agent_id):
    assert index.is_pinned(agent_id, "memory/topics/goal.md") is False


def test_pin_file_makes_is_pinned_true(index, agent_id):
    index.pin_file(agent_id, "memory/topics/goal.md")
    assert index.is_pinned(agent_id, "memory/topics/goal.md") is True


def test_unpin_file_makes_is_pinned_false(index, agent_id):
    index.pin_file(agent_id, "memory/topics/goal.md")
    index.unpin_file(agent_id, "memory/topics/goal.md")
    assert index.is_pinned(agent_id, "memory/topics/goal.md") is False


def test_unpin_file_is_a_no_op_for_a_never_pinned_file(index, agent_id):
    index.unpin_file(agent_id, "memory/topics/goal.md")  # must not raise
    assert index.is_pinned(agent_id, "memory/topics/goal.md") is False


def test_pin_file_is_idempotent(index, agent_id):
    index.pin_file(agent_id, "memory/topics/goal.md")
    index.pin_file(agent_id, "memory/topics/goal.md")  # must not raise or duplicate

    assert index.pinned_files(agent_id) == ["memory/topics/goal.md"]


def test_pinned_files_lists_every_pin_most_recently_first(index, agent_id):
    index.pin_file(agent_id, "memory/topics/a.md")
    index.pin_file(agent_id, "memory/topics/b.md")

    assert index.pinned_files(agent_id) == ["memory/topics/b.md", "memory/topics/a.md"]


def test_pinned_files_is_empty_for_an_agent_with_no_pins(index, agent_id):
    assert index.pinned_files(agent_id) == []


def test_pins_do_not_leak_across_agents(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.pin_file(other_agent, "memory/topics/goal.md")

    try:
        assert index.is_pinned(agent_id, "memory/topics/goal.md") is False
        assert index.pinned_files(agent_id) == []
    finally:
        index.unpin_file(other_agent, "memory/topics/goal.md")
