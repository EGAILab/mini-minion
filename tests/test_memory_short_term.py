"""Tests for ShortTermMemory."""

import json

from minion_assist.memory.short_term import ShortTermMemory


def test_load_empty_returns_empty_list(tmp_path):
    mem = ShortTermMemory(tmp_path)
    assert mem.load("agent1") == []


def test_save_and_load_roundtrip(tmp_path):
    mem = ShortTermMemory(tmp_path)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    mem.save("agent1", messages)
    assert mem.load("agent1") == messages


def test_append_single_message(tmp_path):
    mem = ShortTermMemory(tmp_path)
    msg = {"role": "user", "content": "test"}
    mem.append("agent1", msg)
    assert mem.load("agent1") == [msg]


def test_append_multiple_messages(tmp_path):
    mem = ShortTermMemory(tmp_path)
    m1 = {"role": "user", "content": "a"}
    m2 = {"role": "assistant", "content": "b"}
    mem.append("agent1", m1)
    mem.append("agent1", m2)
    assert mem.load("agent1") == [m1, m2]


def test_save_overwrites_existing(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", [{"role": "user", "content": "old"}])
    new_messages = [{"role": "user", "content": "new"}]
    mem.save("agent1", new_messages)
    assert mem.load("agent1") == new_messages


def test_clear_removes_file(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("agent1", [{"role": "user", "content": "hi"}])
    mem.clear("agent1")
    assert mem.load("agent1") == []


def test_clear_nonexistent_key_is_noop(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.clear("nonexistent")  # should not raise


def test_separate_keys_are_independent(tmp_path):
    mem = ShortTermMemory(tmp_path)
    mem.save("a", [{"role": "user", "content": "for a"}])
    mem.save("b", [{"role": "user", "content": "for b"}])
    assert mem.load("a")[0]["content"] == "for a"
    assert mem.load("b")[0]["content"] == "for b"


def test_jsonl_file_format(tmp_path):
    mem = ShortTermMemory(tmp_path)
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    mem.save("agent1", messages)
    lines = (tmp_path / "agent1.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == messages[0]
    assert json.loads(lines[1]) == messages[1]


def test_base_dir_created_if_missing(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    mem = ShortTermMemory(new_dir)
    assert new_dir.exists()


def test_load_skips_corrupt_jsonl_lines(tmp_path):
    """A corrupt JSONL line must be silently skipped; valid lines are returned."""
    p = tmp_path / "agent1.jsonl"
    good_msg = {"role": "user", "content": "hello"}
    p.write_text(
        json.dumps(good_msg) + "\n" + "not valid json }{{\n",
        encoding="utf-8",
    )
    mem = ShortTermMemory(tmp_path)
    result = mem.load("agent1")
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
    mem.save("agent1", [{"role": "user", "content": "test"}])
    assert replace_calls, "os.replace() was not called during save()"
