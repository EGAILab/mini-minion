"""Tests for ShortTermMemory — per-session JSONL storage."""

import json

from minion_assist.memory.short_term import ShortTermMemory

SID = "test-session-id"


def test_load_empty_returns_empty_list(tmp_path):
    mem = ShortTermMemory(tmp_path)
    assert mem.load("agent1", SID) == []


def test_save_and_load_roundtrip(tmp_path):
    mem = ShortTermMemory(tmp_path)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    mem.save("agent1", SID, messages)
    assert mem.load("agent1", SID) == messages


def test_append_single_message(tmp_path):
    mem = ShortTermMemory(tmp_path)
    msg = {"role": "user", "content": "test"}
    mem.append("agent1", SID, msg)
    assert mem.load("agent1", SID) == [msg]


def test_append_multiple_messages(tmp_path):
    mem = ShortTermMemory(tmp_path)
    m1 = {"role": "user", "content": "a"}
    m2 = {"role": "assistant", "content": "b"}
    mem.append("agent1", SID, m1)
    mem.append("agent1", SID, m2)
    assert mem.load("agent1", SID) == [m1, m2]


def test_save_overwrites_existing(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", SID, [{"role": "user", "content": "old"}])
    new_messages = [{"role": "user", "content": "new"}]
    mem.save("agent1", SID, new_messages)
    assert mem.load("agent1", SID) == new_messages


def test_clear_removes_file(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", SID, [{"role": "user", "content": "hi"}])
    mem.clear("agent1", SID)
    assert mem.load("agent1", SID) == []


def test_clear_nonexistent_key_is_noop(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.clear("nonexistent", SID)  # should not raise


def test_separate_agent_keys_are_independent(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("a", SID, [{"role": "user", "content": "for a"}])
    mem.save("b", SID, [{"role": "user", "content": "for b"}])
    assert mem.load("a", SID)[0]["content"] == "for a"
    assert mem.load("b", SID)[0]["content"] == "for b"


def test_separate_session_ids_are_independent(tmp_path):
    """Different session IDs for the same agent produce independent histories."""
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", "session-a", [{"role": "user", "content": "for a"}])
    mem.save("agent1", "session-b", [{"role": "user", "content": "for b"}])
    assert mem.load("agent1", "session-a")[0]["content"] == "for a"
    assert mem.load("agent1", "session-b")[0]["content"] == "for b"


def test_jsonl_file_format(tmp_path):
    mem = ShortTermMemory(tmp_path)
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    mem.save("agent1", SID, messages)
    lines = (tmp_path / "agent1" / f"{SID}.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == messages[0]
    assert json.loads(lines[1]) == messages[1]


def test_base_dir_created_if_missing(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    mem = ShortTermMemory(new_dir)
    assert new_dir.exists()


def test_agent_subdir_created_on_save(tmp_path):
    """Save creates the agent subdirectory automatically."""
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", SID, [{"role": "user", "content": "hi"}])
    assert (tmp_path / "agent1").is_dir()


def test_load_skips_corrupt_jsonl_lines(tmp_path):
    """A corrupt JSONL line must be silently skipped; valid lines are returned."""
    p = tmp_path / "agent1" / f"{SID}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    good_msg = {"role": "user", "content": "hello"}
    p.write_text(
        json.dumps(good_msg) + "\n" + "not valid json }{{\n",
        encoding="utf-8",
    )
    mem = ShortTermMemory(tmp_path)
    result = mem.load("agent1", SID)
    assert result == [good_msg]


def test_save_uses_atomic_write(tmp_path, monkeypatch):
    """save() must use os.replace() for crash-safe atomic writes."""
    import os as os_module
    replace_calls: list[tuple] = []
    original_replace = os_module.replace

    def spy_replace(src, dst):
        replace_calls.append((src, dst))
        return original_replace(src, dst)

    monkeypatch.setattr(os_module, "replace", spy_replace)
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", SID, [{"role": "user", "content": "test"}])
    assert replace_calls, "os.replace() was not called during save()"


# ---------------------------------------------------------------------------
# list_sessions / prune_sessions
# ---------------------------------------------------------------------------

def test_list_sessions_empty_when_no_history(tmp_path):
    mem = ShortTermMemory(tmp_path)
    assert mem.list_sessions("agent1") == []


def test_list_sessions_returns_saved_files(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", "s1", [{"role": "user", "content": "a"}])
    mem.save("agent1", "s2", [{"role": "user", "content": "b"}])
    files = mem.list_sessions("agent1")
    names = {f.stem for f in files}
    assert names == {"s1", "s2"}


def test_list_sessions_sorted_oldest_first(tmp_path):
    """list_sessions returns files sorted by mtime, oldest first."""
    import time
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", "old", [{"role": "user", "content": "old"}])
    time.sleep(0.05)
    mem.save("agent1", "new", [{"role": "user", "content": "new"}])
    files = mem.list_sessions("agent1")
    assert files[0].stem == "old"
    assert files[1].stem == "new"


def test_prune_sessions_keeps_n_most_recent(tmp_path):
    """prune_sessions deletes oldest files, keeping the N newest."""
    import time
    mem = ShortTermMemory(tmp_path)
    for i in range(5):
        mem.save("agent1", f"s{i}", [{"role": "user", "content": str(i)}])
        time.sleep(0.01)
    deleted = mem.prune_sessions("agent1", keep_n=3)
    assert deleted == 2
    remaining = {f.stem for f in mem.list_sessions("agent1")}
    assert remaining == {"s2", "s3", "s4"}


def test_prune_sessions_noop_when_under_limit(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", "s1", [])
    mem.save("agent1", "s2", [])
    deleted = mem.prune_sessions("agent1", keep_n=10)
    assert deleted == 0
    assert len(mem.list_sessions("agent1")) == 2


def test_prune_sessions_noop_on_missing_agent(tmp_path):
    mem = ShortTermMemory(tmp_path)
    deleted = mem.prune_sessions("nonexistent", keep_n=5)
    assert deleted == 0
