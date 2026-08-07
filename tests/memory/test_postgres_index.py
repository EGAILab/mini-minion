"""Tests for memory/postgres_index.py: PostgresMemoryIndex (Stage One Phase 3, slice A).

Requires a live PostgreSQL instance matching config.json's configured URL
(same instance tests/test_session_db.py uses) — the whole module is skipped
if one isn't reachable. Every test uses a fresh, random agent_id and cleans
up its own rows in an autouse fixture, so tests never collide with each
other or leave rows behind in the shared dev database. ``_DB_URL`` is
sourced from ``minion_assist.config.database.url`` (patched to an isolated
per-session schema by ``tests/conftest.py`` — see its module docstring,
R2-GAP-015), not a literal, so this connects to that isolated schema when
run through the normal ``pytest`` entry point.
"""

from __future__ import annotations

import time
import uuid
from datetime import date as _date

import pytest

from minion_assist.config import database as _database_cfg

_DB_URL = _database_cfg.url or "postgresql://minion:minion@localhost:5433/minion_assist"

try:
    import psycopg as _psycopg

    _test_conn = _psycopg.connect(_DB_URL, connect_timeout=2)
    _test_conn.close()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _DB_AVAILABLE, reason="requires a live PostgreSQL instance")

from minion_assist.memory.postgres_index import (  # noqa: E402
    PostgresMemoryIndex,
    _cosine_similarity,
    _decay_factor,
    _reciprocal_rank_fusion,
    hash_query,
)


def _iso(epoch: float) -> str:
    """ISO-format an epoch-seconds timestamp for embedding in a claim marker's observed= field."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch).isoformat()


@pytest.fixture
def index():
    return PostgresMemoryIndex(_DB_URL)


@pytest.fixture
def vector_index():
    """A PostgresMemoryIndex with an embedding dimension configured (Stage One Phase 4, slice A)."""
    return PostgresMemoryIndex(_DB_URL, embedding_dimensions=3)


@pytest.fixture
def mock_embedding_provider():
    """A fake EmbeddingProvider returning one fixed 3-dim vector per input text."""
    from unittest.mock import Mock

    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    return provider


@pytest.fixture
def embedding_index(mock_embedding_provider):
    """A PostgresMemoryIndex fully wired for embeddings — dimensions + a mock provider."""
    idx = PostgresMemoryIndex(
        _DB_URL, embedding_dimensions=3, embedding_provider=mock_embedding_provider
    )
    yield idx
    # The mock's model_identity is a fixed test-only value never used by a
    # real embedding backend, so this can never touch real cached data.
    idx._conn().execute(
        "DELETE FROM memory_chunk_embeddings WHERE model_identity = 'test-endpoint::test-model'"
    )


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
    conn.execute("DELETE FROM memory_recall_events WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_consolidation_previews WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_topic_revisions WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM memory_import_previews WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM kb_evidence WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM kb_relationships WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM kb_claims WHERE agent_id = %s", (agent_id,))
    conn.execute("DELETE FROM kb_entities WHERE agent_id = %s", (agent_id,))


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
# Action-boundary frontmatter (Stage One Phase 6, slice A)
# ---------------------------------------------------------------------------

def test_reindex_file_extracts_boundary_metadata_from_frontmatter(index, agent_id):
    content = "---\nowner: main\nrequired_approval: user\n---\nBody text."
    index.reindex_file(agent_id, "topic.md", "durable", content)

    boundary = index.get_boundary(agent_id, "topic.md")

    assert boundary == {"owner": "main", "required_approval": "user"}


def test_reindex_file_without_frontmatter_has_no_boundary(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "Just a normal note.")

    assert index.get_boundary(agent_id, "topic.md") is None


def test_get_boundary_returns_none_for_a_file_never_indexed(index, agent_id):
    assert index.get_boundary(agent_id, "never-indexed.md") is None


def test_reindex_file_strips_frontmatter_from_chunk_content(index, agent_id):
    content = "---\nowner: main\n---\nBody text only."
    index.reindex_file(agent_id, "topic.md", "durable", content)

    row = index._conn().execute(
        "SELECT content FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()

    assert "owner" not in row[0]
    assert row[0] == "Body text only."


def test_reindex_file_offsets_chunk_line_numbers_past_the_frontmatter_block(index, agent_id):
    # Original file: ["---", "owner: main", "---", "# Heading", "Body."]
    # "# Heading" is line 4 of the original file, not line 1 of the body.
    content = "---\nowner: main\n---\n# Heading\nBody."
    index.reindex_file(agent_id, "topic.md", "durable", content)

    row = index._conn().execute(
        "SELECT start_line, end_line FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()

    assert row[0] == 4
    assert row[1] == 5


def test_reindex_file_replaces_boundary_metadata_when_frontmatter_changes(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "---\nowner: main\n---\nBody.")
    index.reindex_file(agent_id, "topic.md", "durable", "---\nowner: researcher\n---\nBody.")

    assert index.get_boundary(agent_id, "topic.md") == {"owner": "researcher"}


def test_reindex_file_content_hash_reflects_the_full_file_including_frontmatter(index, agent_id):
    # A frontmatter-only edit (body unchanged) must still be a real content
    # change for reconciliation purposes -- reconcile_agent() diffs by hash.
    index.reindex_file(agent_id, "topic.md", "durable", "---\nowner: main\n---\nBody.")
    first_hash = index._conn().execute(
        "SELECT content_hash FROM memory_files WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()[0]

    index.reindex_file(agent_id, "topic.md", "durable", "---\nowner: researcher\n---\nBody.")
    second_hash = index._conn().execute(
        "SELECT content_hash FROM memory_files WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()[0]

    assert first_hash != second_hash


def test_force_rebuild_agent_preserves_boundary_metadata(index, agent_id):
    content = "---\nowner: main\n---\nBody text."
    index.force_rebuild_agent(agent_id, [("durable", "topic.md", content)])

    assert index.get_boundary(agent_id, "topic.md") == {"owner": "main"}


def test_force_rebuild_agent_strips_frontmatter_from_chunk_content(index, agent_id):
    content = "---\nowner: main\n---\nBody text only."
    index.force_rebuild_agent(agent_id, [("durable", "topic.md", content)])

    row = index._conn().execute(
        "SELECT content FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchone()

    assert row[0] == "Body text only."


# ---------------------------------------------------------------------------
# Knowledge layer: claim sync (Stage One Phase 7, slice A)
# ---------------------------------------------------------------------------

def test_reindex_file_syncs_a_claim_marker(index, agent_id):
    content = "- User's dog is named Biscuit.\n  <!-- claim:c-1 status=supported confidence=0.9 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    assert claim is not None
    assert claim["text"] == "User's dog is named Biscuit."
    assert claim["status"] == "supported"
    assert claim["confidence"] == 0.9
    assert claim["rel_path"] == "topic.md"


def test_reindex_file_with_no_claim_markers_syncs_nothing(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "Just a plain note, no markers.")

    assert index.list_claims(agent_id) == []


def test_reindex_file_only_syncs_claims_for_durable_source_kind(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    index.reindex_file(agent_id, "2026-01-01.md", "daily", content)

    assert index.get_claim(agent_id, "c-1") is None


def test_reindex_file_syncs_claim_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42,message:1189 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    assert claim["evidence"] == [
        {"source_kind": "proposal", "source_ref": "42"},
        {"source_kind": "message", "source_ref": "1189"},
    ]


def test_reindex_file_replacing_content_updates_the_existing_claim(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable", "- Some claim.\n  <!-- claim:c-1 status=unknown -->"
    )
    index.reindex_file(
        agent_id, "topic.md", "durable", "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    )

    claim = index.get_claim(agent_id, "c-1")

    assert claim["status"] == "supported"
    # Still exactly one row -- an upsert, not a duplicate.
    assert len(index.list_claims(agent_id, rel_path="topic.md")) == 1


def test_reindex_file_removes_a_claim_whose_marker_was_deleted(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable", "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", "No more claims here.")

    assert index.get_claim(agent_id, "c-1") is None


def test_reindex_file_removing_a_claim_also_removes_its_evidence(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable",
        "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42 -->",
    )
    index.reindex_file(agent_id, "topic.md", "durable", "No more claims here.")

    row = index._conn().execute(
        "SELECT count(*) FROM kb_evidence WHERE agent_id = %s AND claim_id = %s", (agent_id, "c-1")
    ).fetchone()
    assert row[0] == 0


def test_reindex_file_defaults_observed_at_to_sync_time_when_absent(index, agent_id):
    before = time.time()
    content = "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)
    after = time.time()

    claim = index.get_claim(agent_id, "c-1")

    assert before <= claim["observed_at"] <= after


def test_reindex_file_uses_the_marker_observed_time_when_present(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 observed=2026-06-01 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    from datetime import datetime as _dt
    assert claim["observed_at"] == _dt.fromisoformat("2026-06-01").timestamp()


def test_reindex_file_offsets_claim_line_number_past_the_frontmatter_block(index, agent_id):
    content = "---\nowner: main\n---\n- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    assert claim["line_number"] == 5


def test_remove_file_removes_its_claims(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable", "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    )
    index.remove_file(agent_id, "topic.md")

    assert index.get_claim(agent_id, "c-1") is None


def test_force_rebuild_agent_syncs_claims(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    index.force_rebuild_agent(agent_id, [("durable", "topic.md", content)])

    assert index.get_claim(agent_id, "c-1") is not None


# ---------------------------------------------------------------------------
# Knowledge layer: relationships (Stage One Phase 7, slice B)
# ---------------------------------------------------------------------------

def test_sync_claims_records_a_contradicts_relationship(index, agent_id):
    content = (
        "- Old claim.\n  <!-- claim:c-1 status=supported -->\n"
        "- New claim.\n  <!-- claim:c-2 status=contested contradicts=c-1 -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-2")

    assert claim["contradicts"] == ["c-1"]
    assert claim["supersedes"] == []


def test_sync_claims_records_a_supersedes_relationship(index, agent_id):
    content = (
        "- Old claim.\n  <!-- claim:c-1 status=superseded -->\n"
        "- New claim.\n  <!-- claim:c-2 supersedes=c-1 -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-2")

    assert claim["supersedes"] == ["c-1"]


def test_sync_claims_records_multiple_relationship_targets(index, agent_id):
    content = "- New claim.\n  <!-- claim:c-3 supersedes=c-1,c-2 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-3")

    assert claim["supersedes"] == ["c-1", "c-2"]


def test_list_relationships_to_finds_claims_pointing_at_a_given_claim(index, agent_id):
    content = (
        "- Old claim.\n  <!-- claim:c-1 status=supported -->\n"
        "- New claim.\n  <!-- claim:c-2 status=contested contradicts=c-1 -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    incoming = index.list_relationships_to(agent_id, "c-1")

    assert incoming == [{"from_claim_id": "c-2", "kind": "contradicts"}]


def test_list_relationships_to_is_empty_for_a_claim_nothing_points_at(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_relationships_to(agent_id, "c-1") == []


def test_reindex_file_replacing_content_updates_relationships(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable",
        "- New claim.\n  <!-- claim:c-2 contradicts=c-1 -->",
    )
    index.reindex_file(
        agent_id, "topic.md", "durable",
        "- New claim.\n  <!-- claim:c-2 -->",  # relationship removed on re-sync
    )

    claim = index.get_claim(agent_id, "c-2")

    assert claim["contradicts"] == []


def test_remove_file_removes_relationships_from_its_claims(index, agent_id):
    content = "- New claim.\n  <!-- claim:c-2 contradicts=c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    index.remove_file(agent_id, "topic.md")

    row = index._conn().execute(
        "SELECT count(*) FROM kb_relationships WHERE agent_id = %s AND from_claim_id = %s",
        (agent_id, "c-2"),
    ).fetchone()
    assert row[0] == 0


def test_removing_a_claim_marker_also_removes_its_outgoing_relationships(index, agent_id):
    index.reindex_file(
        agent_id, "topic.md", "durable",
        "- New claim.\n  <!-- claim:c-2 contradicts=c-1 -->",
    )
    index.reindex_file(agent_id, "topic.md", "durable", "No more claims here.")

    row = index._conn().execute(
        "SELECT count(*) FROM kb_relationships WHERE agent_id = %s AND from_claim_id = %s",
        (agent_id, "c-2"),
    ).fetchone()
    assert row[0] == 0


# ---------------------------------------------------------------------------
# Knowledge layer: entities (Stage One Phase 7, slice A)
# ---------------------------------------------------------------------------

def test_get_or_create_entity_creates_a_new_entity(index, agent_id):
    entity_id = index.get_or_create_entity(agent_id, "Biscuit")

    assert entity_id.startswith("e-")
    row = index._conn().execute(
        "SELECT name FROM kb_entities WHERE id = %s", (entity_id,)
    ).fetchone()
    assert row[0] == "Biscuit"


def test_get_or_create_entity_is_idempotent(index, agent_id):
    first = index.get_or_create_entity(agent_id, "Biscuit")
    second = index.get_or_create_entity(agent_id, "Biscuit")

    assert first == second


def test_get_or_create_entity_matches_case_insensitively(index, agent_id):
    first = index.get_or_create_entity(agent_id, "Biscuit")
    second = index.get_or_create_entity(agent_id, "biscuit")

    assert first == second


def test_get_or_create_entity_preserves_the_first_seen_casing(index, agent_id):
    index.get_or_create_entity(agent_id, "Biscuit")
    index.get_or_create_entity(agent_id, "BISCUIT")

    row = index._conn().execute(
        "SELECT name FROM kb_entities WHERE agent_id = %s AND name_normalized = 'biscuit'",
        (agent_id,),
    ).fetchone()
    assert row[0] == "Biscuit"


def test_get_or_create_entity_scoped_to_agent(index, agent_id):
    other_agent = f"other-{agent_id}"
    other_id = index.get_or_create_entity(other_agent, "Biscuit")
    try:
        this_id = index.get_or_create_entity(agent_id, "Biscuit")
        assert this_id != other_id
    finally:
        index._conn().execute("DELETE FROM kb_entities WHERE agent_id = %s", (other_agent,))


def test_reindex_file_resolves_a_claims_entity(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 entity=Biscuit -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    assert claim["entity_id"] is not None
    entity = index._conn().execute(
        "SELECT name FROM kb_entities WHERE id = %s", (claim["entity_id"],)
    ).fetchone()
    assert entity[0] == "Biscuit"


def test_reindex_file_without_entity_field_leaves_entity_id_none(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    claim = index.get_claim(agent_id, "c-1")

    assert claim["entity_id"] is None


# ---------------------------------------------------------------------------
# Knowledge layer: get_claim / list_claims (Stage One Phase 7, slice A)
# ---------------------------------------------------------------------------

def test_get_claim_returns_none_for_an_unknown_id(index, agent_id):
    assert index.get_claim(agent_id, "c-does-not-exist") is None


def test_list_claims_scoped_to_rel_path(index, agent_id):
    index.reindex_file(agent_id, "a.md", "durable", "- Claim A.\n  <!-- claim:c-a -->")
    index.reindex_file(agent_id, "b.md", "durable", "- Claim B.\n  <!-- claim:c-b -->")

    results = index.list_claims(agent_id, rel_path="a.md")

    assert [c["id"] for c in results] == ["c-a"]


def test_list_claims_scoped_to_status(index, agent_id):
    content = (
        "- Claim A.\n  <!-- claim:c-a status=supported -->\n"
        "- Claim B.\n  <!-- claim:c-b status=contested -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims(agent_id, status="contested")

    assert [c["id"] for c in results] == ["c-b"]


def test_list_claims_scoped_to_agent(index, agent_id):
    other_agent = f"other-{agent_id}"
    try:
        index.reindex_file(other_agent, "topic.md", "durable", "- Claim.\n  <!-- claim:c-1 -->")
        assert index.list_claims(agent_id) == []
    finally:
        index._conn().execute("DELETE FROM kb_claims WHERE agent_id = %s", (other_agent,))
        index._conn().execute("DELETE FROM memory_chunks WHERE agent_id = %s", (other_agent,))
        index._conn().execute("DELETE FROM memory_files WHERE agent_id = %s", (other_agent,))


# ---------------------------------------------------------------------------
# Knowledge dashboards (Stage One Phase 7, slice C)
# ---------------------------------------------------------------------------

def test_list_contradictions_returns_a_recorded_contradiction(index, agent_id):
    content = (
        "- Old claim.\n  <!-- claim:c-1 status=contested -->\n"
        "- New claim.\n  <!-- claim:c-2 status=contested contradicts=c-1 -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    contradictions = index.list_contradictions(agent_id)

    assert len(contradictions) == 1
    row = contradictions[0]
    assert row["from_claim_id"] == "c-2"
    assert row["from_text"] == "New claim."
    assert row["to_claim_id"] == "c-1"
    assert row["to_text"] == "Old claim."


def test_list_contradictions_surfaces_a_dangling_reference(index, agent_id):
    content = "- New claim.\n  <!-- claim:c-2 status=contested contradicts=c-does-not-exist -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    [row] = index.list_contradictions(agent_id)

    assert row["to_claim_id"] == "c-does-not-exist"
    assert row["to_text"] is None
    assert row["to_status"] is None


def test_list_contradictions_is_empty_with_no_relationships(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "- Some claim.\n  <!-- claim:c-1 -->")

    assert index.list_contradictions(agent_id) == []


def test_list_stale_claims_finds_an_old_claim(index, agent_id):
    old = time.time() - 200 * 86400  # 200 days ago, well past a 90-day half-life
    content = f"- Old fact.\n  <!-- claim:c-1 observed={_iso(old)} -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    stale = index.list_stale_claims(agent_id, time.time())

    assert [c["id"] for c in stale] == ["c-1"]
    assert stale[0]["freshness"] < 0.5


def test_list_stale_claims_excludes_a_recent_claim(index, agent_id):
    content = f"- Recent fact.\n  <!-- claim:c-1 observed={_iso(time.time())} -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_stale_claims(agent_id, time.time()) == []


def test_list_stale_claims_orders_stalest_first(index, agent_id):
    now = time.time()
    content = (
        f"- Somewhat old.\n  <!-- claim:c-1 observed={_iso(now - 150 * 86400)} -->\n"
        f"- Very old.\n  <!-- claim:c-2 observed={_iso(now - 300 * 86400)} -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    stale = index.list_stale_claims(agent_id, now)

    assert [c["id"] for c in stale] == ["c-2", "c-1"]


def test_list_low_confidence_claims_finds_a_below_threshold_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 confidence=0.2 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_low_confidence_claims(agent_id)

    assert [c["id"] for c in results] == ["c-1"]


def test_list_low_confidence_claims_includes_unrated_claims(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_low_confidence_claims(agent_id)

    assert results[0]["confidence"] is None


def test_list_low_confidence_claims_excludes_a_high_confidence_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 confidence=0.95 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_low_confidence_claims(agent_id) == []


def test_list_claims_missing_evidence_finds_a_claim_with_no_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims_missing_evidence(agent_id)

    assert [c["id"] for c in results] == ["c-1"]


def test_list_claims_missing_evidence_excludes_a_claim_with_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_claims_missing_evidence(agent_id) == []


def test_list_claims_needing_privacy_review_finds_an_unclassified_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims_needing_privacy_review(agent_id)

    assert [c["id"] for c in results] == ["c-1"]


def test_list_claims_needing_privacy_review_excludes_a_classified_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 privacy=private -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_claims_needing_privacy_review(agent_id) == []


def test_list_claims_needing_reevaluation_finds_an_unknown_claim_with_no_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 status=unknown -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims_needing_reevaluation(agent_id)

    assert [c["id"] for c in results] == ["c-1"]


def test_list_claims_needing_reevaluation_excludes_an_unknown_claim_with_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 status=unknown evidence=proposal:42 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_claims_needing_reevaluation(agent_id) == []


def test_list_claims_needing_reevaluation_excludes_a_supported_claim_with_no_evidence(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 status=supported -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_claims_needing_reevaluation(agent_id) == []


def test_list_claims_citing_evidence_finds_a_matching_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims_citing_evidence(agent_id, "proposal", "42")

    assert [c["id"] for c in results] == ["c-1"]


def test_list_claims_citing_evidence_excludes_a_non_matching_claim(index, agent_id):
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:99 -->"
    index.reindex_file(agent_id, "topic.md", "durable", content)

    assert index.list_claims_citing_evidence(agent_id, "proposal", "42") == []


def test_list_claims_citing_evidence_finds_every_claim_citing_a_shared_source(index, agent_id):
    content = (
        "- Claim one.\n  <!-- claim:c-1 evidence=import:_auto_extracted -->\n\n"
        "- Claim two.\n  <!-- claim:c-2 evidence=import:_auto_extracted -->"
    )
    index.reindex_file(agent_id, "topic.md", "durable", content)

    results = index.list_claims_citing_evidence(agent_id, "import", "_auto_extracted")

    assert {c["id"] for c in results} == {"c-1", "c-2"}


def test_list_claims_citing_evidence_is_empty_with_no_matching_evidence(index, agent_id):
    assert index.list_claims_citing_evidence(agent_id, "proposal", "1") == []


def test_reindex_file_without_a_provider_never_embeds(index, agent_id):
    index.reindex_file(agent_id, "topic.md", "durable", "some content")
    # No provider configured -- nothing to assert against a real cache
    # table, but this documents/locks the no-op behavior explicitly.
    assert index._embedding_provider is None


def test_reindex_file_embeds_new_chunks_when_a_provider_is_configured(
    embedding_index, mock_embedding_provider, agent_id
):
    embedding_index.reindex_file(agent_id, "topic.md", "durable", "some content to embed")

    mock_embedding_provider.embed.assert_called_once()
    [chunk_row] = embedding_index._conn().execute(
        "SELECT chunk_hash FROM memory_chunks WHERE agent_id = %s AND rel_path = %s",
        (agent_id, "topic.md"),
    ).fetchall()
    cached = embedding_index.get_cached_embedding(chunk_row[0], "test-endpoint::test-model")
    assert cached == pytest.approx([0.1, 0.2, 0.3])


def test_reindex_file_skips_embedding_already_cached_content(
    embedding_index, mock_embedding_provider, agent_id
):
    embedding_index.reindex_file(agent_id, "a.md", "durable", "shared content")
    mock_embedding_provider.embed.reset_mock()

    # A second file with IDENTICAL content -- its chunk's hash is already
    # cached from the first call, so embed() must not be called again.
    embedding_index.reindex_file(agent_id, "b.md", "durable", "shared content")

    mock_embedding_provider.embed.assert_not_called()


def test_reindex_file_embedding_failure_does_not_block_indexing(
    embedding_index, mock_embedding_provider, agent_id
):
    mock_embedding_provider.embed.side_effect = RuntimeError("provider unavailable")

    count = embedding_index.reindex_file(agent_id, "topic.md", "durable", "some content")

    assert count == 1  # the chunk itself was still indexed
    assert embedding_index.chunk_count(agent_id) == 1


def test_force_rebuild_agent_embeds_chunks_across_all_files(
    embedding_index, mock_embedding_provider, agent_id
):
    files = [
        ("durable", "MEMORY.md", "durable content"),
        ("import", "memory/imports/x.md", "import content"),
    ]

    embedding_index.force_rebuild_agent(agent_id, files)

    mock_embedding_provider.embed.assert_called_once()
    call_args = mock_embedding_provider.embed.call_args.args[0]
    assert sorted(call_args) == sorted(["durable content", "import content"])


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
# reindex_proposal / remove_proposal (Stage One Phase 5, slice B)
# ---------------------------------------------------------------------------

def test_reindex_proposal_writes_a_searchable_chunk(index, agent_id):
    count = index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    assert count == 1
    assert index.chunk_count(agent_id) == 1
    results = index.search(agent_id, "dark mode", corpus="proposal")
    assert len(results) == 1
    assert results[0]["rel_path"] == "proposals/42"
    assert results[0]["source_kind"] == "proposal"


def test_reindex_proposal_does_not_create_a_memory_files_ledger_row(index, agent_id):
    # Proposals have no on-disk file to reconcile against — see the
    # method's docstring for why this matters to rebuild_agent/
    # reconcile_agent/force_rebuild_agent.
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    assert index.indexed_files(agent_id) == []


def test_reindex_proposal_replaces_previous_chunks_for_the_same_proposal(index, agent_id):
    index.reindex_proposal(agent_id, 42, "Original claim.")
    index.reindex_proposal(agent_id, 42, "Updated claim.")

    assert index.chunk_count(agent_id) == 1
    results = index.search(agent_id, "Updated", corpus="proposal")
    assert len(results) == 1


def test_remove_proposal_deletes_its_chunks(index, agent_id):
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    index.remove_proposal(agent_id, 42)

    assert index.chunk_count(agent_id) == 0


def test_remove_proposal_is_a_no_op_for_a_proposal_never_indexed(index, agent_id):
    index.remove_proposal(agent_id, 999)  # must not raise


def test_search_excludes_proposals_by_default(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    results = index.search(agent_id, "dark mode")

    assert [r["rel_path"] for r in results] == ["MEMORY.md"]


def test_search_finds_proposals_when_explicitly_requested(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    results = index.search(agent_id, "dark mode", corpus="proposal")

    assert [r["rel_path"] for r in results] == ["proposals/42"]


def test_hybrid_search_excludes_proposals_by_default(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    results = index.hybrid_search(agent_id, "dark mode")

    assert all(r["rel_path"] != "proposals/42" for r in results)


def test_hybrid_search_finds_proposals_when_explicitly_requested(index, agent_id):
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    results = index.hybrid_search(agent_id, "dark mode", corpus="proposal")

    assert [r["rel_path"] for r in results] == ["proposals/42"]


def test_search_excludes_imports_by_default(index, agent_id):
    # MEM-GAP-004: unreviewed imports must not surface in a corpus-agnostic
    # search, same as proposals above.
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_file(agent_id, "memory/imports/wiki-note.md", "import", "User prefers dark mode.")

    results = index.search(agent_id, "dark mode")

    assert [r["rel_path"] for r in results] == ["MEMORY.md"]


def test_search_finds_imports_when_explicitly_requested(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_file(agent_id, "memory/imports/wiki-note.md", "import", "User prefers dark mode.")

    results = index.search(agent_id, "dark mode", corpus="import")

    assert [r["rel_path"] for r in results] == ["memory/imports/wiki-note.md"]


def test_hybrid_search_excludes_imports_by_default(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User likes dark mode.")
    index.reindex_file(agent_id, "memory/imports/wiki-note.md", "import", "User prefers dark mode.")

    results = index.hybrid_search(agent_id, "dark mode")

    assert all(r["rel_path"] != "memory/imports/wiki-note.md" for r in results)


def test_hybrid_search_finds_imports_when_explicitly_requested(index, agent_id):
    index.reindex_file(agent_id, "memory/imports/wiki-note.md", "import", "User prefers dark mode.")

    results = index.hybrid_search(agent_id, "dark mode", corpus="import")

    assert [r["rel_path"] for r in results] == ["memory/imports/wiki-note.md"]


def test_force_rebuild_agent_does_not_delete_proposal_chunks(index, agent_id):
    # force_rebuild_agent is a files-only operation (see its docstring) —
    # proposal chunks share memory_chunks but must survive it untouched.
    index.reindex_file(agent_id, "MEMORY.md", "durable", "Some content.")
    index.reindex_proposal(agent_id, 42, "User prefers dark mode.")

    index.force_rebuild_agent(agent_id, [("durable", "MEMORY.md", "Some content.")])

    assert index.search(agent_id, "dark mode", corpus="proposal") != []


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


def test_old_chunk_id_keyed_table_self_heals_into_the_new_schema():
    # Simulates a machine that already ran the earlier chunk_id-keyed
    # build of memory_chunk_embeddings (Stage One Phase 4, slice A) before
    # this content-hash-keyed shape existed.
    conn = _psycopg.connect(_DB_URL, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS memory_chunk_embeddings")
    conn.execute("""
        CREATE TABLE memory_chunk_embeddings (
            chunk_id BIGINT PRIMARY KEY,
            model_identity TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding vector(3) NOT NULL
        )
    """)
    conn.close()

    try:
        migrated = PostgresMemoryIndex(_DB_URL, embedding_dimensions=3)
        assert migrated.has_vector_lane is True

        migrated.cache_embedding("hash-migration-test", "test/model", [0.1, 0.2, 0.3])
        result = migrated.get_cached_embedding("hash-migration-test", "test/model")

        assert result is not None
        assert result[0] == pytest.approx(0.1, abs=1e-4)
    finally:
        conn = _psycopg.connect(_DB_URL, autocommit=True)
        conn.execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = 'hash-migration-test'"
        )
        conn.close()


def test_self_healing_migration_is_scoped_to_the_active_schema_only():
    # R2-GAP-015 regression: found via the round-two remediation's own test-
    # isolation work actually exercising a second schema for the first time.
    # The self-healing detection query used to filter only by table_name,
    # not table_schema -- a second schema's own same-named
    # memory_chunk_embeddings table (with an identically-named
    # auto-generated "..._pkey" constraint, the common case) silently mixed
    # both schemas' primary-key columns into one result, so the *correctly*
    # shaped table's columns polluted the check and the genuinely
    # old-shaped table in the target schema was never healed.
    other_schema = f"test_other_{uuid.uuid4().hex[:8]}"
    target_schema = f"test_target_{uuid.uuid4().hex[:8]}"
    conn = _psycopg.connect(_DB_URL, autocommit=True)
    conn.execute(f'CREATE SCHEMA "{other_schema}"')
    conn.execute(f'CREATE SCHEMA "{target_schema}"')
    # A second schema with the CORRECT (already-migrated) shape -- same
    # table name, and (Postgres' default naming) likely the same
    # auto-generated primary-key constraint name too.
    conn.execute(f"""
        CREATE TABLE "{other_schema}".memory_chunk_embeddings (
            content_hash TEXT NOT NULL,
            model_identity TEXT NOT NULL,
            embedding vector(3) NOT NULL,
            PRIMARY KEY (content_hash, model_identity)
        )
    """)
    # The schema this test actually targets still has the OLD
    # (chunk_id-keyed) shape that must be detected and healed.
    conn.execute(f"""
        CREATE TABLE "{target_schema}".memory_chunk_embeddings (
            chunk_id BIGINT PRIMARY KEY,
            model_identity TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding vector(3) NOT NULL
        )
    """)
    conn.close()

    target_url = f"{_DB_URL}?options=-c%20search_path%3D{target_schema}"
    try:
        migrated = PostgresMemoryIndex(target_url, embedding_dimensions=3)
        assert migrated.has_vector_lane is True

        # Would previously raise psycopg.errors.InvalidColumnReference --
        # the old chunk_id-keyed table was never actually dropped/recreated
        # with the (content_hash, model_identity) primary key ON CONFLICT
        # depends on, because the contaminated detection query didn't see
        # a clean {"chunk_id"} match.
        migrated.cache_embedding("hash-scoped-test", "test/model", [0.1, 0.2, 0.3])
        result = migrated.get_cached_embedding("hash-scoped-test", "test/model")

        assert result is not None
        assert result[0] == pytest.approx(0.1, abs=1e-4)
    finally:
        conn = _psycopg.connect(_DB_URL, autocommit=True)
        conn.execute(f'DROP SCHEMA "{other_schema}" CASCADE')
        conn.execute(f'DROP SCHEMA "{target_schema}" CASCADE')
        conn.close()


def test_memory_chunk_embeddings_self_heals_a_dimension_change():
    # R2-GAP-015 regression: same reasoning as session/db.py's identical
    # message_embeddings check -- found via test-isolation work sharing one
    # schema across PostgresMemoryIndex instances constructed with
    # different embedding_dimensions, also a real production scenario if
    # embeddings.dimensions ever changes in config.json.
    PostgresMemoryIndex(_DB_URL, embedding_dimensions=5)

    healed = PostgresMemoryIndex(_DB_URL, embedding_dimensions=3)
    try:
        healed.cache_embedding("hash-dim-heal-test", "test/model", [0.1, 0.2, 0.3])  # must not raise
        result = healed.get_cached_embedding("hash-dim-heal-test", "test/model")
        assert result is not None
    finally:
        conn = _psycopg.connect(_DB_URL, autocommit=True)
        conn.execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = 'hash-dim-heal-test'"
        )
        conn.close()


def test_cache_embedding_is_a_no_op_without_a_vector_lane(index):
    index.cache_embedding("hash-a", "test/model", [0.1, 0.2, 0.3])  # must not raise
    assert index.get_cached_embedding("hash-a", "test/model") is None


def test_get_cached_embedding_returns_none_for_a_never_cached_hash(vector_index):
    assert vector_index.get_cached_embedding("hash-never-cached", "test/model") is None


def test_cache_embedding_round_trips(vector_index):
    content_hash = "hash-round-trip"
    try:
        vector_index.cache_embedding(content_hash, "test/model", [0.1, 0.2, 0.3])

        result = vector_index.get_cached_embedding(content_hash, "test/model")

        assert result is not None
        assert len(result) == 3
        assert result[0] == pytest.approx(0.1, abs=1e-4)
        assert result[1] == pytest.approx(0.2, abs=1e-4)
        assert result[2] == pytest.approx(0.3, abs=1e-4)
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = %s", (content_hash,)
        )


def test_get_cached_embedding_misses_on_wrong_model_identity(vector_index):
    content_hash = "hash-model-mismatch"
    try:
        vector_index.cache_embedding(content_hash, "test/model-a", [0.1, 0.2, 0.3])

        assert vector_index.get_cached_embedding(content_hash, "test/model-b") is None
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = %s", (content_hash,)
        )


def test_cache_embedding_replaces_a_previous_value_for_the_same_hash_and_model(vector_index):
    content_hash = "hash-replace"
    try:
        vector_index.cache_embedding(content_hash, "test/model", [0.1, 0.2, 0.3])
        vector_index.cache_embedding(content_hash, "test/model", [0.4, 0.5, 0.6])

        result = vector_index.get_cached_embedding(content_hash, "test/model")
        assert result[0] == pytest.approx(0.4, abs=1e-4)
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = %s", (content_hash,)
        )


def test_cache_embedding_is_shared_across_agents_for_identical_content(vector_index):
    # No agent_id in the cache key by design -- identical text embeds
    # identically regardless of which agent's note it came from.
    content_hash = "hash-shared"
    try:
        vector_index.cache_embedding(content_hash, "test/model", [0.7, 0.8, 0.9])

        result = vector_index.get_cached_embedding(content_hash, "test/model")
        assert result[0] == pytest.approx(0.7, abs=1e-4)
    finally:
        vector_index._conn().execute(
            "DELETE FROM memory_chunk_embeddings WHERE content_hash = %s", (content_hash,)
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


# ---------------------------------------------------------------------------
# Fusion helpers (Stage One Phase 4, slice C) — pure functions, no DB needed
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_sums_contributions_across_lanes():
    row_a = {"id": 1, "rel_path": "a.md"}
    row_b = {"id": 2, "rel_path": "b.md"}
    lane1 = [row_a, row_b]  # a ranked 1st, b ranked 2nd
    lane2 = [row_b, row_a]  # b ranked 1st, a ranked 2nd

    scores, rows = _reciprocal_rank_fusion([lane1, lane2])

    # Symmetric ranks across two lanes -> equal fused scores.
    assert scores[1] == pytest.approx(scores[2])
    assert rows[1] == row_a
    assert rows[2] == row_b


def test_reciprocal_rank_fusion_favors_a_chunk_ranked_highly_in_multiple_lanes():
    row_a = {"id": 1}
    row_b = {"id": 2}
    lane1 = [row_a, row_b]
    lane2 = [row_a, row_b]  # a is top in both lanes

    scores, _rows = _reciprocal_rank_fusion([lane1, lane2])

    assert scores[1] > scores[2]


def test_reciprocal_rank_fusion_handles_empty_lanes():
    scores, rows = _reciprocal_rank_fusion([[], []])
    assert scores == {}
    assert rows == {}


def test_reciprocal_rank_fusion_a_chunk_missing_from_a_lane_is_not_penalized():
    # A chunk absent from a lane contributes 0 from that lane, not a
    # negative adjustment -- it just doesn't collect that lane's score.
    row_a = {"id": 1}
    scores, _rows = _reciprocal_rank_fusion([[row_a], []])
    assert scores[1] > 0


def test_decay_factor_is_1_for_durable_content_regardless_of_path():
    assert _decay_factor("memory/2020-01-01.md", "durable") == 1.0


def test_decay_factor_is_1_for_import_content():
    assert _decay_factor("memory/2020-01-01.md", "import") == 1.0


def test_decay_factor_is_1_for_a_daily_note_dated_today():
    today_path = f"memory/{_date.today().isoformat()}.md"
    assert _decay_factor(today_path, "daily") == 1.0


def test_decay_factor_is_less_than_1_for_an_old_daily_note():
    old_path = "memory/2000-01-01.md"
    assert _decay_factor(old_path, "daily") < 1.0


def test_decay_factor_decreases_monotonically_with_age():
    recent = _decay_factor("memory/2026-07-01.md", "daily")
    older = _decay_factor("memory/2026-01-01.md", "daily")
    assert older < recent


def test_decay_factor_is_1_for_an_unparseable_daily_path():
    assert _decay_factor("memory/not-a-date.md", "daily") == 1.0


def test_cosine_similarity_of_identical_vectors_is_1():
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_0():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_1():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_with_a_zero_vector_is_0():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# hybrid_search (Stage One Phase 4, slice C)
# ---------------------------------------------------------------------------

def test_hybrid_search_finds_a_lexical_match_without_an_embedding_provider(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode in the editor.")

    results = index.hybrid_search(agent_id, "dark mode")

    assert any(r["rel_path"] == "MEMORY.md" for r in results)


def test_hybrid_search_path_lane_surfaces_exact_filename_match(index, agent_id):
    # The query term appears in the file's path/key but nowhere in its body.
    index.reindex_file(
        agent_id, "memory/topics/project-goals.md", "durable", "Ship it this quarter."
    )

    results = index.hybrid_search(agent_id, "project-goals")

    assert any(r["rel_path"] == "memory/topics/project-goals.md" for r in results)


def test_hybrid_search_always_includes_pinned_content_even_without_a_query_match(
    index, agent_id
):
    index.reindex_file(agent_id, "memory/topics/standing-rule.md", "durable", "Always be polite.")
    index.pin_file(agent_id, "memory/topics/standing-rule.md")

    results = index.hybrid_search(agent_id, "completely unrelated query xyz")

    assert any(r["rel_path"] == "memory/topics/standing-rule.md" for r in results)


def test_hybrid_search_respects_corpus_filter(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "shared keyword content")
    index.reindex_file(agent_id, "memory/imports/x.md", "import", "shared keyword content")

    durable_only = index.hybrid_search(agent_id, "shared keyword", corpus="durable")

    assert all(r["source_kind"] == "durable" for r in durable_only)


def test_hybrid_search_respects_max_results(index, agent_id):
    for i in range(10):
        index.reindex_file(agent_id, f"memory/topics/note-{i}.md", "durable", "shared keyword")

    results = index.hybrid_search(agent_id, "shared keyword", max_results=3)

    assert len(results) <= 3


def test_hybrid_search_returns_empty_for_an_agent_with_nothing_indexed(index, agent_id):
    # No recent-lane fallback content exists at all for this agent, so
    # there's nothing any lane could surface.
    results = index.hybrid_search(agent_id, "anything")
    assert results == []


def test_hybrid_search_recent_lane_still_surfaces_content_for_an_unrelated_query(index, agent_id):
    # The recent lane is explicitly "regardless of content match" (it's a
    # fallback so a fresh conversation still gets some injected context) --
    # an unrelated query is not expected to return nothing as long as the
    # agent has indexed content at all.
    index.reindex_file(agent_id, "MEMORY.md", "durable", "unrelated content")

    results = index.hybrid_search(agent_id, "xyzzy-nonexistent-term")

    assert any(r["rel_path"] == "MEMORY.md" for r in results)


def test_hybrid_search_only_searches_the_given_agent(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.reindex_file(other_agent, "MEMORY.md", "durable", "shared keyword content")

    try:
        results = index.hybrid_search(agent_id, "shared keyword")
        assert results == []
    finally:
        index.remove_file(other_agent, "MEMORY.md")


def test_hybrid_search_uses_the_vector_lane_when_a_provider_is_configured(
    embedding_index, mock_embedding_provider, agent_id
):
    embedding_index.reindex_file(agent_id, "MEMORY.md", "durable", "some indexed content")

    embedding_index.hybrid_search(agent_id, "a semantically related query")

    # The vector lane embeds the query itself (one call), on top of whatever
    # reindex_file's own indexing-time embedding call already made.
    assert mock_embedding_provider.embed.call_count >= 2


def test_hybrid_search_applies_temporal_decay_to_old_daily_notes(index, agent_id):
    # MEMORY.md indexed FIRST (older by indexed_at, disadvantaged in the
    # recent lane) and the daily note indexed LAST (favored by recency) --
    # if durable still outranks it despite that recency disadvantage,
    # decay is doing real work, not just recency coincidentally agreeing.
    index.reindex_file(agent_id, "MEMORY.md", "durable", "shared keyword content")
    index.reindex_file(agent_id, "memory/2000-01-01.md", "daily", "shared keyword content")

    results = index.hybrid_search(agent_id, "shared keyword")

    by_path = {r["rel_path"]: r["score"] for r in results}
    assert by_path["memory/2000-01-01.md"] < by_path["MEMORY.md"]


# ---------------------------------------------------------------------------
# hash_query (Stage One Phase 5, slice A) — pure function, no DB needed
# ---------------------------------------------------------------------------

def test_hash_query_is_deterministic():
    assert hash_query("Python facts") == hash_query("Python facts")


def test_hash_query_ignores_case():
    assert hash_query("Python Facts") == hash_query("python facts")


def test_hash_query_collapses_whitespace():
    assert hash_query("python   facts") == hash_query("python facts")
    assert hash_query("  python facts  ") == hash_query("python facts")


def test_hash_query_differs_for_different_queries():
    assert hash_query("python facts") != hash_query("coffee preferences")


# ---------------------------------------------------------------------------
# Recall telemetry (Stage One Phase 5, slice A)
# ---------------------------------------------------------------------------

def test_recall_stats_for_a_never_recalled_file(index, agent_id):
    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats == {
        "recall_count": 0, "unique_queries": 0, "injected_count": 0, "last_recalled_at": None,
    }


def test_record_recall_increments_recall_count(index, agent_id):
    index.record_recall(agent_id, "MEMORY.md", hash_query("first query"))
    index.record_recall(agent_id, "MEMORY.md", hash_query("second query"))

    stats = index.recall_stats(agent_id, "MEMORY.md")

    assert stats["recall_count"] == 2
    assert stats["injected_count"] == 0
    assert stats["last_recalled_at"] is not None


def test_record_recall_tracks_unique_queries(index, agent_id):
    index.record_recall(agent_id, "MEMORY.md", hash_query("same query"))
    index.record_recall(agent_id, "MEMORY.md", hash_query("same query"))
    index.record_recall(agent_id, "MEMORY.md", hash_query("different query"))

    stats = index.recall_stats(agent_id, "MEMORY.md")

    assert stats["recall_count"] == 3
    assert stats["unique_queries"] == 2


def test_mark_injected_flags_the_most_recent_matching_event(index, agent_id):
    query_hash = hash_query("some query")
    index.record_recall(agent_id, "MEMORY.md", query_hash)

    index.mark_injected(agent_id, ["MEMORY.md"], query_hash)

    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats["injected_count"] == 1


def test_mark_injected_does_not_affect_a_different_query_hash(index, agent_id):
    index.record_recall(agent_id, "MEMORY.md", hash_query("query one"))

    index.mark_injected(agent_id, ["MEMORY.md"], hash_query("query two"))

    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats["injected_count"] == 0


def test_mark_injected_is_a_no_op_for_a_file_never_recalled(index, agent_id):
    index.mark_injected(agent_id, ["never-recalled.md"], hash_query("q"))  # must not raise
    assert index.recall_stats(agent_id, "never-recalled.md")["recall_count"] == 0


def test_recall_events_do_not_leak_across_agents(index, agent_id):
    other_agent = f"test-{uuid.uuid4()}"
    index.record_recall(other_agent, "MEMORY.md", hash_query("q"))

    try:
        stats = index.recall_stats(agent_id, "MEMORY.md")
        assert stats["recall_count"] == 0
    finally:
        index._conn().execute(
            "DELETE FROM memory_recall_events WHERE agent_id = %s", (other_agent,)
        )


def test_plain_search_does_not_record_recall_telemetry(index, agent_id):
    # search() is now purely an internal lane candidate-generator for
    # hybrid_search() (nothing else calls it directly) -- recording
    # telemetry here too would double-count/over-count lane candidates
    # that fusion filters out, which were never actually "surfaced" to
    # anyone. Only hybrid_search()'s final, actually-returned results
    # should ever be recorded (Task 1: "results actually surfaced").
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode.")

    index.search(agent_id, "dark mode")

    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats["recall_count"] == 0


def test_hybrid_search_records_recall_telemetry(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode.")

    index.hybrid_search(agent_id, "dark mode")

    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats["recall_count"] >= 1


def test_hybrid_search_uses_the_same_query_hash_as_a_manual_mark_injected_call(index, agent_id):
    index.reindex_file(agent_id, "MEMORY.md", "durable", "User prefers dark mode.")

    index.hybrid_search(agent_id, "dark mode")
    index.mark_injected(agent_id, ["MEMORY.md"], hash_query("dark mode"))

    stats = index.recall_stats(agent_id, "MEMORY.md")
    assert stats["injected_count"] == 1


# ---------------------------------------------------------------------------
# Consolidation previews (Stage One Phase 5, slice C)
# ---------------------------------------------------------------------------

def test_record_consolidation_preview_returns_a_new_id_each_time(index, agent_id):
    first_id = index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Drafted content.", "New preference."
    )
    second_id = index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Different draft.", "Redrafted."
    )

    assert first_id != second_id


def test_list_consolidation_previews_returns_everything_for_the_agent(index, agent_id):
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft one.", "Reason one."
    )
    index.record_consolidation_preview(
        agent_id, 2, "revise_topic", "project-goals", "abc123", "Draft two.", "Reason two."
    )

    previews = index.list_consolidation_previews(agent_id)

    assert {p["proposal_id"] for p in previews} == {1, 2}


def test_list_consolidation_previews_restricts_to_one_proposal(index, agent_id):
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft one.", "Reason one."
    )
    index.record_consolidation_preview(
        agent_id, 2, "revise_topic", "project-goals", "abc123", "Draft two.", "Reason two."
    )

    previews = index.list_consolidation_previews(agent_id, proposal_id=1)

    assert [p["proposal_id"] for p in previews] == [1]


def test_list_consolidation_previews_returns_newest_first(index, agent_id):
    first_id = index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "First draft.", "First."
    )
    second_id = index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Second draft.", "Second."
    )

    previews = index.list_consolidation_previews(agent_id, proposal_id=1)

    assert [p["id"] for p in previews] == [second_id, first_id]


def test_list_consolidation_previews_is_empty_for_an_agent_with_none(index, agent_id):
    assert index.list_consolidation_previews(agent_id) == []


def test_consolidation_previews_do_not_leak_across_agents(index, agent_id):
    other_agent = f"other-{agent_id}"
    index.record_consolidation_preview(
        other_agent, 1, "new_topic", "dark-mode", "", "Draft.", "Reason."
    )
    try:
        assert index.list_consolidation_previews(agent_id) == []
    finally:
        index._conn().execute(
            "DELETE FROM memory_consolidation_previews WHERE agent_id = %s", (other_agent,)
        )


def test_get_consolidation_preview_returns_the_matching_row(index, agent_id):
    preview_id = index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft.", "New preference."
    )

    preview = index.get_consolidation_preview(preview_id)

    assert preview["proposal_id"] == 1
    assert preview["target_key"] == "dark-mode"
    assert preview["rationale"] == "New preference."


def test_get_consolidation_preview_returns_none_for_an_unknown_id(index):
    assert index.get_consolidation_preview(-1) is None


# ---------------------------------------------------------------------------
# remove_consolidation_previews_for_proposal (MEM-GAP-003)
# ---------------------------------------------------------------------------

def test_remove_consolidation_previews_for_proposal_deletes_matching_previews(index, agent_id):
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft.", "Reason."
    )
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Redraft.", "Reason two."
    )

    index.remove_consolidation_previews_for_proposal(agent_id, 1)

    assert index.list_consolidation_previews(agent_id, proposal_id=1) == []


def test_remove_consolidation_previews_for_proposal_leaves_other_proposals_alone(index, agent_id):
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft one.", "Reason one."
    )
    index.record_consolidation_preview(
        agent_id, 2, "revise_topic", "project-goals", "abc123", "Draft two.", "Reason two."
    )

    index.remove_consolidation_previews_for_proposal(agent_id, 1)

    remaining = index.list_consolidation_previews(agent_id)
    assert [p["proposal_id"] for p in remaining] == [2]


def test_remove_consolidation_previews_for_proposal_is_a_no_op_when_none_exist(index, agent_id):
    index.remove_consolidation_previews_for_proposal(agent_id, 999)  # must not raise


# ---------------------------------------------------------------------------
# prune_operational_tables (MEM-GAP-015)
# ---------------------------------------------------------------------------

def _backdate(index, table: str, column: str, agent_id: str, days_old: float) -> None:
    index._conn().execute(
        f"UPDATE {table} SET {column} = %s WHERE agent_id = %s",
        (time.time() - days_old * 86400, agent_id),
    )


def test_prune_operational_tables_deletes_old_recall_events(index, agent_id):
    index.record_recall(agent_id, "topics/x.md", "hash-1")
    _backdate(index, "memory_recall_events", "surfaced_at", agent_id, days_old=40)

    counts = index.prune_operational_tables(30)

    assert counts["recall_events"] == 1
    assert index._conn().execute(
        "SELECT count(*) FROM memory_recall_events WHERE agent_id = %s", (agent_id,)
    ).fetchone()[0] == 0


def test_prune_operational_tables_never_deletes_a_recent_recall_event(index, agent_id):
    index.record_recall(agent_id, "topics/x.md", "hash-1")

    counts = index.prune_operational_tables(30)

    assert counts["recall_events"] == 0
    assert index._conn().execute(
        "SELECT count(*) FROM memory_recall_events WHERE agent_id = %s", (agent_id,)
    ).fetchone()[0] == 1


def test_prune_operational_tables_dry_run_counts_without_deleting(index, agent_id):
    index.record_recall(agent_id, "topics/x.md", "hash-1")
    _backdate(index, "memory_recall_events", "surfaced_at", agent_id, days_old=40)

    counts = index.prune_operational_tables(30, dry_run=True)

    assert counts["recall_events"] == 1
    assert index._conn().execute(
        "SELECT count(*) FROM memory_recall_events WHERE agent_id = %s", (agent_id,)
    ).fetchone()[0] == 1  # still there — dry run must not delete


def test_prune_operational_tables_deletes_old_consolidation_previews_regardless_of_review_status(
    index, agent_id
):
    """A reviewed (approved/rejected) preview row is never deleted by the
    review flow itself (see remove_consolidation_previews_for_proposal's
    docstring) — this is the only cleanup path for an old one."""
    index.record_consolidation_preview(
        agent_id, 1, "new_topic", "dark-mode", "", "Draft.", "Reason."
    )
    _backdate(index, "memory_consolidation_previews", "created_at", agent_id, days_old=40)

    counts = index.prune_operational_tables(30)

    assert counts["consolidation_previews"] == 1


def test_prune_operational_tables_deletes_old_import_previews(index, agent_id):
    index.record_import_preview(
        agent_id, "import-1", "new_topic", "dark-mode", "", "Draft.", "Reason."
    )
    _backdate(index, "memory_import_previews", "created_at", agent_id, days_old=40)

    counts = index.prune_operational_tables(30)

    assert counts["import_previews"] == 1


def test_prune_operational_tables_never_touches_topic_revisions(index, agent_id):
    """memory_topic_revisions is a rollback undo-stack, not telemetry — see
    prune_operational_tables' own docstring for why it's deliberately excluded."""
    index.record_topic_revision(agent_id, "project-goals", 1, "Existing goal: ship v1.")
    _backdate(index, "memory_topic_revisions", "created_at", agent_id, days_old=365)

    index.prune_operational_tables(30)

    assert index.latest_topic_revision(agent_id, "project-goals") is not None


# ---------------------------------------------------------------------------
# Topic revisions (Stage One Phase 5, slice D)
# ---------------------------------------------------------------------------

def test_latest_topic_revision_is_none_when_never_applied(index, agent_id):
    assert index.latest_topic_revision(agent_id, "project-goals") is None


def test_record_and_fetch_a_topic_revision(index, agent_id):
    index.record_topic_revision(agent_id, "project-goals", 1, "Existing goal: ship v1.")

    revision = index.latest_topic_revision(agent_id, "project-goals")

    assert revision["proposal_id"] == 1
    assert revision["prior_content"] == "Existing goal: ship v1."


def test_record_a_topic_revision_with_no_proposal(index, agent_id):
    # MEM-GAP-020: an explicit MemoryService.remember()/delete() write has
    # no proposal to attribute the revision to.
    index.record_topic_revision(agent_id, "project-goals", None, "Prior content.")

    revision = index.latest_topic_revision(agent_id, "project-goals")

    assert revision["proposal_id"] is None
    assert revision["prior_content"] == "Prior content."


def test_latest_topic_revision_returns_the_most_recent_one(index, agent_id):
    index.record_topic_revision(agent_id, "project-goals", 1, "First prior content.")
    index.record_topic_revision(agent_id, "project-goals", 2, "Second prior content.")

    revision = index.latest_topic_revision(agent_id, "project-goals")

    assert revision["prior_content"] == "Second prior content."


def test_delete_topic_revision_removes_it_from_the_undo_stack(index, agent_id):
    index.record_topic_revision(agent_id, "project-goals", 1, "First prior content.")
    second_id = index.record_topic_revision(agent_id, "project-goals", 2, "Second prior content.")

    index.delete_topic_revision(second_id)

    revision = index.latest_topic_revision(agent_id, "project-goals")
    assert revision["prior_content"] == "First prior content."


def test_topic_revisions_do_not_leak_across_agents(index, agent_id):
    other_agent = f"other-{agent_id}"
    index.record_topic_revision(other_agent, "project-goals", 1, "Content.")
    try:
        assert index.latest_topic_revision(agent_id, "project-goals") is None
    finally:
        index._conn().execute(
            "DELETE FROM memory_topic_revisions WHERE agent_id = %s", (other_agent,)
        )


# ---------------------------------------------------------------------------
# Import review previews (Stage One Phase 7, slice E)
# ---------------------------------------------------------------------------

def test_record_import_preview_returns_a_new_id_each_time(index, agent_id):
    first_id = index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Drafted content.", "New pref."
    )
    second_id = index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Different draft.", "Redrafted."
    )

    assert first_id != second_id


def test_list_import_previews_returns_everything_for_the_agent(index, agent_id):
    index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Draft one.", "Reason one."
    )
    index.record_import_preview(
        agent_id, "other-import", "revise_topic", "project-goals", "abc123", "Draft two.", "Reason two."
    )

    previews = index.list_import_previews(agent_id)

    assert {p["import_key"] for p in previews} == {"_auto_extracted", "other-import"}


def test_list_import_previews_restricts_to_one_import_key(index, agent_id):
    index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Draft one.", "Reason one."
    )
    index.record_import_preview(
        agent_id, "other-import", "revise_topic", "project-goals", "abc123", "Draft two.", "Reason two."
    )

    previews = index.list_import_previews(agent_id, import_key="_auto_extracted")

    assert [p["import_key"] for p in previews] == ["_auto_extracted"]


def test_list_import_previews_returns_newest_first(index, agent_id):
    first_id = index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "First draft.", "First."
    )
    second_id = index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Second draft.", "Second."
    )

    previews = index.list_import_previews(agent_id, import_key="_auto_extracted")

    assert [p["id"] for p in previews] == [second_id, first_id]


def test_list_import_previews_is_empty_for_an_agent_with_none(index, agent_id):
    assert index.list_import_previews(agent_id) == []


def test_import_previews_do_not_leak_across_agents(index, agent_id):
    other_agent = f"other-{agent_id}"
    index.record_import_preview(
        other_agent, "_auto_extracted", "new_topic", "dark-mode", "", "Draft.", "Reason."
    )
    try:
        assert index.list_import_previews(agent_id) == []
    finally:
        index._conn().execute(
            "DELETE FROM memory_import_previews WHERE agent_id = %s", (other_agent,)
        )


def test_get_import_preview_returns_the_matching_row(index, agent_id):
    preview_id = index.record_import_preview(
        agent_id, "_auto_extracted", "new_topic", "dark-mode", "", "Draft.", "New preference."
    )

    preview = index.get_import_preview(preview_id)

    assert preview["import_key"] == "_auto_extracted"
    assert preview["target_key"] == "dark-mode"
    assert preview["rationale"] == "New preference."


def test_get_import_preview_returns_none_for_an_unknown_id(index):
    assert index.get_import_preview(-1) is None
