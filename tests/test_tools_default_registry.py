"""Tests for default_registry factory."""

from mini_minion.tools import default_registry


def test_default_registry_has_expected_tools():
    names = {d["function"]["name"] for d in default_registry().definitions}
    assert names == {"read", "write", "glob", "bash"}


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
