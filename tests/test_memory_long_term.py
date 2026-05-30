"""Tests for LongTermMemory."""

from mini_minion.memory.long_term import _SEARCH_MAX_RESULTS, LongTermMemory


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


def test_search_multi_term_matches_any(tmp_path):
    """Any term in a space-separated query is sufficient for a match."""
    mem = LongTermMemory(tmp_path)
    mem.save("python-notes", "Python async patterns")
    mem.save("java-notes", "Java threading model")
    # "python" matches first note; "java" matches second; both returned
    results = mem.search("python java")
    keys = [k for k, _ in results]
    assert "python-notes" in keys
    assert "java-notes" in keys


def test_search_multi_term_no_partial_phrase_required(tmp_path):
    """Phrase 'daughter math' need not appear verbatim — either word suffices."""
    mem = LongTermMemory(tmp_path)
    mem.save("daughter-math-challenge", "# Parent Help\n10-year-old girl dislikes math")
    results = mem.search("daughter math")
    assert len(results) == 1
    assert results[0][0] == "daughter-math-challenge"


def test_search_matches_key_stem(tmp_path):
    """A query term that appears only in the file key (not content) still matches."""
    mem = LongTermMemory(tmp_path)
    # Content does not contain "daughter"; key does
    mem.save("daughter-profile", "age 10, enjoys art and music")
    results = mem.search("daughter")
    assert len(results) == 1
    assert results[0][0] == "daughter-profile"


def test_search_empty_query_returns_empty(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("note", "content")
    assert mem.search("") == []
    assert mem.search("   ") == []


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


def test_search_caps_at_max_results(tmp_path):
    """search() must return at most _SEARCH_MAX_RESULTS notes even when more match."""
    mem = LongTermMemory(tmp_path)
    for i in range(_SEARCH_MAX_RESULTS + 5):
        mem.save(f"note-{i}", "matching keyword here")
    results = mem.search("keyword")
    assert len(results) == _SEARCH_MAX_RESULTS


def test_per_agent_memory_is_isolated(tmp_path):
    """Two LongTermMemory instances at different paths don't share notes.

    This mirrors the minion.py setup where each agent gets its own subdirectory
    under workspace/memory/<agent_id>/.
    """
    main_mem = LongTermMemory(tmp_path / "main")
    researcher_mem = LongTermMemory(tmp_path / "researcher")

    main_mem.save("shared-key", "Ada's version")
    researcher_mem.save("shared-key", "Elizabeth's version")

    assert main_mem.load("shared-key") == "Ada's version"
    assert researcher_mem.load("shared-key") == "Elizabeth's version"

    # A note saved by one agent is invisible to the other.
    main_mem.save("ada-only", "private to Ada")
    assert researcher_mem.load("ada-only") is None
    assert "ada-only" not in researcher_mem.list_keys()
