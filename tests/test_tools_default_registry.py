"""Tests for default_registry factory."""

from unittest.mock import Mock

from minion_assist.tools import default_registry


def test_default_registry_has_expected_tools():
    names = {d["function"]["name"] for d in default_registry().definitions}
    # Core tools always registered (Tier 1 additions: edit, grep, web_fetch)
    assert {"read", "write", "glob", "bash", "web_search", "edit", "grep", "web_fetch"}.issubset(names)


def test_default_registry_tool_names_are_unique():
    defs = default_registry().definitions
    names = [d["function"]["name"] for d in defs]
    assert len(names) == len(set(names))


def test_default_registry_definitions_are_valid():
    for d in default_registry().definitions:
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_default_registry_can_execute_read(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    reg = default_registry()
    result = reg.execute("read", {"path": str(f)})
    assert "hello" in result


def test_default_registry_can_execute_write(tmp_path):
    target = tmp_path / "out.txt"
    reg = default_registry()
    reg.execute("write", {"path": str(target), "content": "written"})
    assert target.read_text() == "written"


def test_default_registry_can_execute_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    reg = default_registry()
    result = reg.execute("glob", {"pattern": "*.py", "path": str(tmp_path)})
    assert "a.py" in result


# ---------------------------------------------------------------------------
# root parameter — verify it is threaded to all file tools
# ---------------------------------------------------------------------------


def test_default_registry_root_enforced_in_read(tmp_path):
    """Read outside the workspace root must be rejected when root is set."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("shh")
    reg = default_registry(root=root)
    result = reg.execute("read", {"path": str(outside)})
    assert "outside the workspace root" in result


def test_default_registry_root_enforced_in_write(tmp_path):
    """Write outside the workspace root must be rejected when root is set."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "evil.txt"
    reg = default_registry(root=root)
    result = reg.execute("write", {"path": str(outside), "content": "bad"})
    assert "outside the workspace root" in result
    assert not outside.exists()


def test_default_registry_root_enforced_in_glob(tmp_path):
    """Glob with explicit path outside the workspace root must be rejected."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    reg = default_registry(root=root)
    result = reg.execute("glob", {"pattern": "*", "path": str(outside)})
    assert "outside the workspace root" in result


# ---------------------------------------------------------------------------
# memory_root parameter — widens `read` (only) to the agent's own workspace
# directory, e.g. SOUL.md/USER.md/memory/YYYY-MM-DD.md that live outside `root`
# ---------------------------------------------------------------------------


def test_default_registry_memory_root_allows_read_outside_project_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    memory_root = tmp_path / "workspace" / "main"
    memory_root.mkdir(parents=True)
    soul = memory_root / "SOUL.md"
    soul.write_text("soul content")

    reg = default_registry(root=root, memory_root=memory_root)
    result = reg.execute("read", {"path": str(soul)})

    assert "soul content" in result


def test_default_registry_no_memory_root_still_rejects_workspace_dir(tmp_path):
    """Without memory_root, the workspace directory is treated like any other
    out-of-root path (regression guard for the widening above)."""
    root = tmp_path / "project"
    root.mkdir()
    memory_root = tmp_path / "workspace" / "main"
    memory_root.mkdir(parents=True)
    soul = memory_root / "SOUL.md"
    soul.write_text("soul content")

    reg = default_registry(root=root)
    result = reg.execute("read", {"path": str(soul)})

    assert "outside the workspace root" in result


def test_default_registry_memory_root_does_not_widen_write(tmp_path):
    """memory_root only widens `read` — write must still be rejected outside root."""
    root = tmp_path / "project"
    root.mkdir()
    memory_root = tmp_path / "workspace" / "main"
    memory_root.mkdir(parents=True)
    target = memory_root / "SOUL.md"

    reg = default_registry(root=root, memory_root=memory_root)
    result = reg.execute("write", {"path": str(target), "content": "hijacked"})

    assert "outside the workspace root" in result
    assert not target.exists()


# ---------------------------------------------------------------------------
# bash_confirm parameter — verify it is threaded to BashTool
# ---------------------------------------------------------------------------


def test_default_registry_bash_confirm_none_runs_without_prompt():
    """bash_confirm=None must run bash commands without calling any confirm callable."""
    reg = default_registry(bash_confirm=None)
    result = reg.execute("bash", {"command": "echo hello-from-test"})
    assert "hello-from-test" in result


def test_default_registry_bash_confirm_callable_cancel():
    """A confirm callable that returns False must cancel the command."""
    reg = default_registry(bash_confirm=lambda _: False)
    result = reg.execute("bash", {"command": "echo should-not-run"})
    assert "cancelled" in result.lower()


# ---------------------------------------------------------------------------
# session_search tool registration — agent scoping (MEM-GAP-002)
# ---------------------------------------------------------------------------


def test_session_search_not_registered_without_a_db():
    reg = default_registry(agent_id="main")
    assert "session_search" not in {d["function"]["name"] for d in reg.definitions}


def test_session_search_not_registered_without_an_agent_id():
    # A db without a known owning agent must not register the tool at all —
    # there's no safe unscoped fallback (MEM-GAP-002).
    reg = default_registry(db=Mock())
    assert "session_search" not in {d["function"]["name"] for d in reg.definitions}


def test_session_search_registered_when_db_and_agent_id_are_both_given():
    reg = default_registry(db=Mock(), agent_id="main")
    assert "session_search" in {d["function"]["name"] for d in reg.definitions}
