"""Tests for SaveMemoryTool and SearchMemoryTool."""

from unittest.mock import Mock

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.models import MemoryHit
from minion_assist.memory.service import MemoryService
from minion_assist.tools.memory import SaveMemoryTool, SearchMemoryTool


def _service(tmp_path) -> MemoryService:
    return MemoryService(MemoryFileRepository(tmp_path))


def test_save_memory_tool_schema():
    mem = MemoryService.__new__(MemoryService)
    tool = SaveMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "save_memory"
    assert schema.description
    assert "key" in schema.parameters["properties"]
    assert "content" in schema.parameters["properties"]


def test_search_memory_tool_schema():
    mem = MemoryService.__new__(MemoryService)
    tool = SearchMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "search_memory"
    assert "query" in schema.parameters["properties"]


def test_save_memory_saves_and_returns_confirmation(tmp_path):
    mem = _service(tmp_path)
    tool = SaveMemoryTool(mem)
    result = tool.execute(key="my-note", content="# Hello\nworld")
    assert "my-note" in result
    assert mem.load("my-note") == "# Hello\nworld"


def test_search_memory_finds_saved_content(tmp_path):
    mem = _service(tmp_path)
    mem.remember("api-notes", "REST API best practices")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="REST")
    assert "api-notes" in result
    assert "REST API" in result


def test_search_memory_shows_boundary_annotation_when_present():
    # Stage One Phase 6, slice A. Only an index-backed hit ever carries
    # hit.boundary -- a Mock stands in rather than a real Postgres index.
    mem = Mock()
    mem.search.return_value = [
        MemoryHit(
            key="deploy-note", content="Some content.", source="durable",
            boundary="[Boundary — advisory only, does not itself grant permission — Owner: main]",
        )
    ]
    tool = SearchMemoryTool(mem)

    result = tool.execute(query="deploy")

    assert "Boundary" in result
    assert "Owner: main" in result


def test_search_memory_omits_boundary_text_when_absent():
    mem = Mock()
    mem.search.return_value = [
        MemoryHit(key="plain-note", content="Some content.", source="durable")
    ]
    tool = SearchMemoryTool(mem)

    result = tool.execute(query="query")

    assert "Boundary" not in result


def test_search_memory_no_results_empty_memory(tmp_path):
    """Empty memory returns a 'memory is empty' message."""
    mem = _service(tmp_path)
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="xyzzy")
    assert "xyzzy" in result
    assert "No memories" in result


def test_search_memory_no_results_lists_available_keys(tmp_path):
    """When notes exist but nothing matches, available keys are listed."""
    mem = _service(tmp_path)
    mem.remember("daughter-profile", "age 10, likes art")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="Isabella")
    assert "No memories" in result
    assert "daughter-profile" in result
    assert "broader" in result.lower() or "available" in result.lower()


def test_search_memory_multi_term_finds_note(tmp_path):
    """Multi-term query with any matching term returns the note."""
    mem = _service(tmp_path)
    mem.remember("daughter-math-challenge", "# Parent Help\n10-year-old girl dislikes math")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="daughter vibe coding")
    assert "daughter-math-challenge" in result
    assert "Parent Help" in result


def test_default_registry_with_memory_has_memory_tools(tmp_path):
    from minion_assist.tools import default_registry

    mem = _service(tmp_path)
    reg = default_registry(memory=mem)
    names = {d["function"]["name"] for d in reg.definitions}
    assert "save_memory" in names
    assert "search_memory" in names
    assert "memory_get" in names
    assert "write_daily_memory" in names


def test_default_registry_without_memory_has_no_memory_tools():
    from minion_assist.tools import default_registry

    reg = default_registry()
    names = {d["function"]["name"] for d in reg.definitions}
    assert "save_memory" not in names
    assert "search_memory" not in names
    assert "memory_get" not in names
    assert "write_daily_memory" not in names


def test_default_registry_with_memory_but_no_db_has_no_pin_tool(tmp_path):
    # pin_memory needs a lexical index, which needs a database — not offered
    # to the LLM at all without one (Stage One Phase 4, slice B).
    from minion_assist.tools import default_registry

    mem = _service(tmp_path)
    reg = default_registry(memory=mem)
    names = {d["function"]["name"] for d in reg.definitions}
    assert "pin_memory" not in names


def test_default_registry_with_memory_and_db_has_pin_tool(tmp_path):
    from unittest.mock import Mock

    from minion_assist.tools import default_registry

    mem = _service(tmp_path)
    reg = default_registry(memory=mem, db=Mock())
    names = {d["function"]["name"] for d in reg.definitions}
    assert "pin_memory" in names


def test_search_memory_cap_note_shown_when_limit_hit(tmp_path):
    """SearchMemoryTool output must include a cap hint when _SEARCH_MAX_RESULTS notes match."""
    from minion_assist.memory.service import _SEARCH_MAX_RESULTS

    mem = _service(tmp_path)
    for i in range(_SEARCH_MAX_RESULTS + 5):
        mem.remember(f"note-{i}", "keyword content")
    tool = SearchMemoryTool(mem)
    result = tool.execute(query="keyword")
    assert "capped" in result.lower()
