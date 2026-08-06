"""Tests for SaveMemoryTool and WriteDailyMemoryTool policy (read_only_mode) integration."""

from datetime import date

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.tools.memory import SaveMemoryTool
from minion_assist.tools.policy import PermissionPolicy
from minion_assist.tools.write_daily_memory import WriteDailyMemoryTool


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
# WriteDailyMemoryTool policy tests
# ---------------------------------------------------------------------------
# Since Phase 1 slice 4, WriteDailyMemoryTool absorbed the retired `note`
# tool's responsibility (and its read_only_mode check — write_daily_memory
# did not check policy at all before the merge).

def test_write_daily_memory_no_policy_appends(tmp_path):
    """When no policy is passed, WriteDailyMemoryTool appends without restriction."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem)
    result = tool.execute(content="quick observation")
    assert "Appended" in result


def test_write_daily_memory_policy_read_write_allowed(tmp_path):
    """When policy read_only_mode=False, WriteDailyMemoryTool appends normally."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem, policy=_make_policy(read_only=False))
    result = tool.execute(content="hello")
    assert "Appended" in result


def test_write_daily_memory_policy_read_only_blocked(tmp_path):
    """When policy read_only_mode=True, WriteDailyMemoryTool returns an error and does not write."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem, policy=_make_policy(read_only=True))
    result = tool.execute(content="should not appear")
    assert "Error" in result
    assert "read-only" in result.lower()
    assert mem.status().daily_count == 0


def test_write_daily_memory_empty_content_rejected(tmp_path):
    """WriteDailyMemoryTool rejects empty content even without a policy."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem)
    result = tool.execute(content="")
    assert "Empty" in result


def test_write_daily_memory_appends_multiple_entries(tmp_path):
    """Calling WriteDailyMemoryTool twice appends both entries to today's log."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem)
    tool.execute(content="first note")
    tool.execute(content="second note")
    path = tmp_path / "memory" / f"{date.today().isoformat()}.md"
    content = path.read_text(encoding="utf-8")
    assert "first note" in content
    assert "second note" in content


def test_write_daily_memory_schema(tmp_path):
    """WriteDailyMemoryTool.schema has the expected name and required 'content' parameter."""
    mem = _service(tmp_path)
    tool = WriteDailyMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "write_daily_memory"
    assert "content" in schema.parameters["properties"]
    assert "content" in schema.parameters.get("required", [])
