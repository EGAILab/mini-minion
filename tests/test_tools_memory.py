"""Tests for SaveMemoryTool and SearchMemoryTool."""

from mini_minion.memory.long_term import LongTermMemory
from mini_minion.tools.memory import SaveMemoryTool, SearchMemoryTool


def test_save_memory_tool_schema():
    mem = LongTermMemory.__new__(LongTermMemory)
    tool = SaveMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "save_memory"
    assert schema.description
    assert "key" in schema.parameters["properties"]
    assert "content" in schema.parameters["properties"]


def test_search_memory_tool_schema():
    mem = LongTermMemory.__new__(LongTermMemory)
    tool = SearchMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "search_memory"
    assert "query" in schema.parameters["properties"]


def test_save_memory_saves_and_returns_confirmation(tmp_path):
    mem = LongTermMemory(tmp_path)
    tool = SaveMemoryTool(mem)
    result = tool.execute(key="my-note", content="# Hello\nworld")
    assert "my-note" in result
    assert mem.load("my-note") == "# Hello\nworld"


def test_search_memory_finds_saved_content(tmp_path):
    mem = LongTermMemory(tmp_path)
    mem.save("api-notes", "REST API best practices")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="REST")
    assert "api-notes" in result
    assert "REST API" in result


def test_search_memory_no_results_empty_memory(tmp_path):
    """Empty memory returns a 'memory is empty' message."""
    mem = LongTermMemory(tmp_path)
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="xyzzy")
    assert "xyzzy" in result
    assert "No memories" in result


def test_search_memory_no_results_lists_available_keys(tmp_path):
    """When notes exist but nothing matches, available keys are listed."""
    mem = LongTermMemory(tmp_path)
    mem.save("daughter-profile", "age 10, likes art")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="Isabella")
    assert "No memories" in result
    assert "daughter-profile" in result
    assert "broader" in result.lower() or "available" in result.lower()


def test_search_memory_multi_term_finds_note(tmp_path):
    """Multi-term query with any matching term returns the note."""
    mem = LongTermMemory(tmp_path)
    mem.save("daughter-math-challenge", "# Parent Help\n10-year-old girl dislikes math")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="daughter vibe coding")
    assert "daughter-math-challenge" in result
    assert "Parent Help" in result


def test_default_registry_with_long_term_has_memory_tools(tmp_path):
    from mini_minion.tools import default_registry

    mem = LongTermMemory(tmp_path)
    reg = default_registry(long_term=mem)
    names = {d["function"]["name"] for d in reg.definitions}
    assert "save_memory" in names
    assert "search_memory" in names


def test_default_registry_without_long_term_has_no_memory_tools():
    from mini_minion.tools import default_registry

    reg = default_registry()
    names = {d["function"]["name"] for d in reg.definitions}
    assert "save_memory" not in names
    assert "search_memory" not in names


def test_search_memory_cap_note_shown_when_limit_hit(tmp_path):
    """SearchMemoryTool output must include a cap hint when _SEARCH_MAX_RESULTS notes match."""
    from mini_minion.memory.long_term import _SEARCH_MAX_RESULTS

    mem = LongTermMemory(tmp_path)
    for i in range(_SEARCH_MAX_RESULTS + 5):
        mem.save(f"note-{i}", "keyword content")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="keyword")
    assert "capped" in result.lower()
