"""Tests for SessionStore."""

import os

from minion_assist.session.store import SessionInfo, SessionStore


def test_get_or_create_new_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    info = store.get_or_create("main")
    assert info.agent_id == "main"
    assert info.turn_count == 0
    assert info.created_at
    assert info.last_active


def test_get_or_create_returns_same_on_second_call(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    first = store.get_or_create("main")
    second = store.get_or_create("main")
    assert first.created_at == second.created_at


def test_touch_updates_last_active(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")
    info = store.touch("main")
    assert info.agent_id == "main"


def test_touch_increments_turns(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")
    store.touch("main", increment_turns=True)
    info = store.touch("main", increment_turns=True)
    assert info.turn_count == 2


def test_touch_no_increment(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")
    info = store.touch("main", increment_turns=False)
    assert info.turn_count == 0


def test_touch_nonexistent_creates_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    info = store.touch("researcher")
    assert info.agent_id == "researcher"


def test_list_sessions_empty(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    assert store.list_sessions() == []


def test_list_sessions_returns_all(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")
    store.get_or_create("researcher")
    sessions = store.list_sessions()
    agent_ids = {s.agent_id for s in sessions}
    assert agent_ids == {"main", "researcher"}


def test_session_info_is_dataclass():
    info = SessionInfo(agent_id="x", created_at="t1", last_active="t2", turn_count=3)
    assert info.agent_id == "x"
    assert info.turn_count == 3


def test_session_info_session_id_defaults_empty():
    info = SessionInfo(agent_id="x", created_at="t1", last_active="t2", turn_count=0)
    assert info.session_id == ""


def test_store_persists_to_file(tmp_path):
    path = tmp_path / "sessions.json"
    store = SessionStore(path)
    store.get_or_create("main")
    assert path.exists()

    store2 = SessionStore(path)
    sessions = store2.list_sessions()
    assert any(s.agent_id == "main" for s in sessions)


def test_load_corrupt_json_returns_empty_and_renames(tmp_path):
    """Corrupt sessions.json must not crash; file is renamed to .corrupt."""
    path = tmp_path / "sessions.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = SessionStore(path)

    result = store._load()

    assert result == {}
    assert not path.exists()
    assert (tmp_path / "sessions.corrupt").exists()


def test_save_is_atomic_via_temp_file(tmp_path, monkeypatch):
    """_save() must write through a .tmp file and replace atomically."""
    path = tmp_path / "sessions.json"
    store = SessionStore(path)

    replaced: list[tuple] = []
    real_replace = os.replace

    def _spy(src, dst):
        replaced.append((str(src), dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)

    store._save({"x": {}})

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src.endswith(".tmp")
    assert dst == path
    assert path.exists()


# ---------------------------------------------------------------------------
# IMP-14: In-memory cache
# ---------------------------------------------------------------------------


def test_cache_avoids_repeated_disk_reads(tmp_path):
    """After the first load, subsequent reads use the cache (no file I/O)."""
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")

    # Corrupt the file — if the cache is working, this won't affect the store.
    (tmp_path / "sessions.json").write_text("CORRUPTED", encoding="utf-8")

    info = store.touch("main", increment_turns=True)
    assert info.turn_count == 1


def test_cache_write_through_persists_to_disk(tmp_path):
    """After touch(), a new SessionStore instance sees the updated data."""
    store1 = SessionStore(tmp_path / "sessions.json")
    store1.get_or_create("main")
    store1.touch("main", increment_turns=True)
    store1.touch("main", increment_turns=True)

    # New instance reads fresh from disk.
    store2 = SessionStore(tmp_path / "sessions.json")
    assert store2.get_or_create("main").turn_count == 2


# ---------------------------------------------------------------------------
# NEW-04: parent_id (session fork lineage)
# ---------------------------------------------------------------------------


def test_get_or_create_with_parent_id_stores_lineage(tmp_path):
    """get_or_create with parent_id must record the fork lineage."""
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("main")
    child = store.get_or_create("child", parent_id="main")
    assert child.parent_id == "main"


def test_get_or_create_without_parent_id_defaults_none(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    info = store.get_or_create("main")
    assert info.parent_id is None


def test_parent_id_persists_across_reload(tmp_path):
    """parent_id must survive a serialize/deserialize round-trip."""
    path = tmp_path / "sessions.json"
    store1 = SessionStore(path)
    store1.get_or_create("parent")
    store1.get_or_create("child", parent_id="parent")

    store2 = SessionStore(path)
    records = {r.agent_id: r for r in store2.list_sessions()}
    assert records["child"].parent_id == "parent"
    assert records["parent"].parent_id is None


def test_existing_session_parent_id_not_overwritten(tmp_path):
    """Calling get_or_create again on an existing session must not change parent_id."""
    store = SessionStore(tmp_path / "sessions.json")
    store.get_or_create("child", parent_id="original-parent")
    # Second call with a different parent_id — must be ignored.
    info = store.get_or_create("child", parent_id="different-parent")
    assert info.parent_id == "original-parent"


def test_session_info_parent_id_defaults_none():
    info = SessionInfo(agent_id="x", created_at="t1", last_active="t2", turn_count=0)
    assert info.parent_id is None


# ---------------------------------------------------------------------------
# session_id — UUID tracking
# ---------------------------------------------------------------------------


def test_get_or_create_generates_session_id(tmp_path):
    """get_or_create must populate session_id with a UUID on first create."""
    store = SessionStore(tmp_path / "sessions.json")
    info = store.get_or_create("main")
    assert info.session_id
    assert len(info.session_id) == 36  # UUID format


def test_get_or_create_preserves_existing_session_id(tmp_path):
    """Calling get_or_create twice must not rotate the session_id."""
    store = SessionStore(tmp_path / "sessions.json")
    first = store.get_or_create("main")
    second = store.get_or_create("main")
    assert first.session_id == second.session_id


def test_new_session_rotates_uuid(tmp_path):
    """new_session must generate a different UUID from the current one."""
    store = SessionStore(tmp_path / "sessions.json")
    info = store.get_or_create("main")
    original_id = info.session_id
    new_id = store.new_session("main")
    assert new_id != original_id
    assert len(new_id) == 36


def test_new_session_persists_across_reload(tmp_path):
    """After new_session(), a fresh store instance reads the rotated session_id."""
    path = tmp_path / "sessions.json"
    store1 = SessionStore(path)
    store1.get_or_create("main")
    new_id = store1.new_session("main")

    store2 = SessionStore(path)
    info = store2.get_or_create("main")
    assert info.session_id == new_id


def test_new_session_creates_record_if_missing(tmp_path):
    """new_session on an unknown agent creates the record and returns a UUID."""
    store = SessionStore(tmp_path / "sessions.json")
    new_id = store.new_session("fresh-agent")
    assert new_id
    info = store.get_or_create("fresh-agent")
    assert info.session_id == new_id


def test_get_or_create_migrates_legacy_record_without_session_id(tmp_path):
    """A sessions.json written by older code (no session_id field) gets a UUID on first access."""
    import json
    path = tmp_path / "sessions.json"
    # Write a legacy record without session_id.
    path.write_text(json.dumps({"main": {
        "agent_id": "main",
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_active": "2026-01-01T00:00:00+00:00",
        "turn_count": 5,
        "parent_id": None,
    }}), encoding="utf-8")

    store = SessionStore(path)
    info = store.get_or_create("main")
    assert info.session_id  # must have been populated
    assert info.turn_count == 5  # legacy data preserved
