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


def test_search_memory_no_results(tmp_path):
    mem = LongTermMemory(tmp_path)
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="xyzzy")
    assert "xyzzy" in result
    assert "No memories" in result


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
