"""Tests for SaveMemoryTool and NoteTool policy (read_only_mode) integration."""

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.tools.memory import NoteTool, SaveMemoryTool
from minion_assist.tools.policy import PermissionPolicy


def _service(tmp_path) -> MemoryService:
    return MemoryService(MemoryFileRepository(tmp_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(read_only: bool = False) -> PermissionPolicy:
    """Return a PermissionPolicy with read_only_mode set as requested."""
    return PermissionPolicy(read_only_mode=read_only)


# ---------------------------------------------------------------------------
# SaveMemoryTool policy tests
# ---------------------------------------------------------------------------

def test_save_memory_no_policy_writes(tmp_path):
    """When no policy is passed, SaveMemoryTool writes without restriction."""
    mem = _service(tmp_path)
    tool = SaveMemoryTool(mem)
    result = tool.execute(key="k", content="v")
    assert "k" in result
    assert mem.load("k") == "v"


def test_save_memory_policy_read_write_allowed(tmp_path):
    """When policy read_only_mode=False, writes proceed normally."""
    mem = _service(tmp_path)
    tool = SaveMemoryTool(mem, policy=_make_policy(read_only=False))
    result = tool.execute(key="goal", content="world domination")
    assert "goal" in result
    assert mem.load("goal") == "world domination"


def test_save_memory_policy_read_only_blocked(tmp_path):
    """When policy read_only_mode=True, SaveMemoryTool returns an error and does not write."""
    mem = _service(tmp_path)
    tool = SaveMemoryTool(mem, policy=_make_policy(read_only=True))
    result = tool.execute(key="secret", content="should not be saved")
    assert "Error" in result
    assert "read-only" in result.lower()
    # Nothing should be on disk.
    assert mem.load("secret") is None


def test_save_memory_read_only_error_suggests_auto(tmp_path):
    """The error message tells the user how to exit read-only mode."""
    mem = _service(tmp_path)
    tool = SaveMemoryTool(mem, policy=_make_policy(read_only=True))
    result = tool.execute(key="x", content="y")
    assert "/auto" in result


# ---------------------------------------------------------------------------
# NoteTool policy tests
# ---------------------------------------------------------------------------
# NoteTool writes quarantined content (memory/imports/), not curated topic
# notes — see docs/adr/0003-per-agent-memory-scope.md — so these tests check
# list_import_keys()/load_import() rather than list_keys()/load().

def test_note_tool_no_policy_appends(tmp_path):
    """When no policy is passed, NoteTool appends to today's daily log."""
    mem = _service(tmp_path)
    tool = NoteTool(mem)
    result = tool.execute(text="quick observation")
    assert "Note saved" in result
    # Verify something was actually stored with a _notes_ key.
    all_keys = mem.list_import_keys()
    assert any(k.startswith("_notes_") for k in all_keys)


def test_note_tool_policy_read_write_allowed(tmp_path):
    """When policy read_only_mode=False, NoteTool appends normally."""
    mem = _service(tmp_path)
    tool = NoteTool(mem, policy=_make_policy(read_only=False))
    result = tool.execute(text="hello")
    assert "Note saved" in result


def test_note_tool_policy_read_only_blocked(tmp_path):
    """When policy read_only_mode=True, NoteTool returns an error and does not write."""
    mem = _service(tmp_path)
    tool = NoteTool(mem, policy=_make_policy(read_only=True))
    result = tool.execute(text="should not appear")
    assert "Error" in result
    assert "read-only" in result.lower()
    # Nothing should be stored.
    assert mem.list_import_keys() == []


def test_note_tool_empty_text_rejected(tmp_path):
    """NoteTool rejects empty text even without a policy."""
    mem = _service(tmp_path)
    tool = NoteTool(mem)
    result = tool.execute(text="")
    assert "Error" in result


def test_note_tool_appends_multiple_entries(tmp_path):
    """Calling NoteTool twice appends both bullets to today's log."""
    mem = _service(tmp_path)
    tool = NoteTool(mem)
    tool.execute(text="first note")
    tool.execute(text="second note")
    all_keys = mem.list_import_keys()
    note_key = next(k for k in all_keys if k.startswith("_notes_"))
    content = mem.load_import(note_key)
    assert "first note" in content
    assert "second note" in content


def test_note_tool_schema():
    """NoteTool.schema has the expected name and required 'text' parameter."""
    mem = MemoryService.__new__(MemoryService)
    tool = NoteTool(mem)
    schema = tool.schema
    assert schema.name == "note"
    assert "text" in schema.parameters["properties"]
    assert "text" in schema.parameters.get("required", [])
