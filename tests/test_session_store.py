"""Tests for SessionStore."""

from mini_minion.session.store import SessionInfo, SessionStore


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


def test_store_persists_to_file(tmp_path):
    path = tmp_path / "sessions.json"
    store = SessionStore(path)
    store.get_or_create("main")
    assert path.exists()

    store2 = SessionStore(path)
    sessions = store2.list_sessions()
    assert any(s.agent_id == "main" for s in sessions)
