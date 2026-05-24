"""Tests for LongTermMemory."""

from mini_minion.memory.long_term import LongTermMemory


def test_save_and_load_roundtrip(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("my-note", "# Title\nsome content")
    assert mem.load("my-note") == "# Title\nsome content"


def test_load_nonexistent_returns_none(tmp_path):
    mem = LongTermMemory(tmp_path)
    assert mem.load("no-such-key") is None


def test_save_creates_md_file(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("test-key", "hello")
    assert (tmp_path / "test-key.md").exists()


def test_save_overwrites_existing(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("key", "old")
    mem.save("key", "new")
    assert mem.load("key") == "new"


def test_search_finds_matching_content(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("alpha", "This talks about Python")
    mem.save("beta", "This is about JavaScript")
    results = mem.search("Python")
    keys = [k for k, _ in results]
    assert "alpha" in keys
    assert "beta" not in keys


def test_search_case_insensitive(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("note", "Contains UPPERCASE word")
    results = mem.search("uppercase")
    assert len(results) == 1
    assert results[0][0] == "note"


def test_search_returns_key_and_content(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("my-note", "important info")
    results = mem.search("important")
    assert results[0] == ("my-note", "important info")


def test_search_no_match_returns_empty(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("note", "nothing relevant")
    assert mem.search("xyzzy") == []


def test_list_keys_empty(tmp_path):
    mem = LongTermMemory(tmp_path)
    assert mem.list_keys() == []


def test_list_keys_returns_all(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("a", "x")
    mem.save("b", "y")
    assert set(mem.list_keys()) == {"a", "b"}


def test_delete_existing_key(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("key", "content")
    result = mem.delete("key")
    assert result is True
    assert mem.load("key") is None


def test_delete_nonexistent_returns_false(tmp_path):
    mem = LongTermMemory(tmp_path)
    assert mem.delete("no-such-key") is False


def test_key_with_slash_sanitized(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("a/b/c", "content")
    assert mem.load("a/b/c") == "content"
    assert (tmp_path / "a_b_c.md").exists()


def test_base_dir_created_if_missing(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    mem = LongTermMemory(new_dir)
    assert new_dir.exists()
