"""Tests for tool workspace-root sandboxing and bash confirmation."""

import builtins

from mini_minion.tools.bash import BashTool
from mini_minion.tools.glob import GlobTool
from mini_minion.tools.read import ReadTool
from mini_minion.tools.write import WriteTool

# ---------------------------------------------------------------------------
# ReadTool — path containment
# ---------------------------------------------------------------------------


def test_read_rejects_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secrets.txt"
    outside.write_text("secret", encoding="utf-8")

    result = ReadTool(root).execute(path=str(outside))

    assert "outside the workspace root" in result


def test_read_allows_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    inside = root / "hello.txt"
    inside.write_text("hi", encoding="utf-8")

    result = ReadTool(root).execute(path=str(inside))

    assert "1: hi" in result


def test_read_no_root_allows_any_path(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("open", encoding="utf-8")

    result = ReadTool(root=None).execute(path=str(f))

    assert "1: open" in result


def test_read_rejects_traversal_escape(tmp_path):
    """Path traversal via .. must not escape the root."""
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    traversal = root / ".." / "outside.txt"

    result = ReadTool(root).execute(path=str(traversal))

    assert "outside the workspace root" in result


# ---------------------------------------------------------------------------
# WriteTool — path containment
# ---------------------------------------------------------------------------


def test_write_rejects_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "evil.txt"

    result = WriteTool(root).execute(path=str(outside), content="bad")

    assert "outside the workspace root" in result
    assert not outside.exists()


def test_write_allows_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "out.txt"

    result = WriteTool(root).execute(path=str(target), content="hello")

    assert "Wrote" in result
    assert target.read_text(encoding="utf-8") == "hello"


# ---------------------------------------------------------------------------
# GlobTool — path containment and default root
# ---------------------------------------------------------------------------


def test_glob_rejects_explicit_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()

    result = GlobTool(root).execute(pattern="*", path=str(outside))

    assert "outside the workspace root" in result


def test_glob_allows_explicit_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("", encoding="utf-8")

    result = GlobTool(root).execute(pattern="*.py", path=str(root))

    assert "a.py" in result


def test_glob_defaults_to_root_when_no_path_given(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "found.py").write_text("", encoding="utf-8")

    result = GlobTool(root).execute(pattern="*.py")

    assert "found.py" in result


# ---------------------------------------------------------------------------
# BashTool — confirmation prompt
# ---------------------------------------------------------------------------


def test_bash_confirm_y_runs_command(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _: "y")
    tool = BashTool(confirm=True)

    result = tool.execute(command="echo hello")

    assert "hello" in result


def test_bash_confirm_n_cancels(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _: "n")
    tool = BashTool(confirm=True)

    result = tool.execute(command="echo should-not-run")

    assert "cancelled" in result.lower()
    assert "should-not-run" not in result


def test_bash_confirm_false_runs_without_prompt(monkeypatch):
    # If confirm=False, input() must never be called.
    monkeypatch.setattr(builtins, "input", lambda _: (_ for _ in ()).throw(AssertionError("input() called")))
    tool = BashTool(confirm=False)

    result = tool.execute(command="echo no-prompt")

    assert "no-prompt" in result


def test_bash_confirm_prints_command_before_prompt(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _: "n")
    BashTool(confirm=True).execute(command="echo marker")

    out = capsys.readouterr().out
    assert "echo marker" in out


def test_bash_tool_cwd_is_used(tmp_path):
    """BashTool cwd= starts the subprocess in the specified directory."""
    tool = BashTool(confirm=False, cwd=tmp_path)
    result = tool.execute(command='python -c "import os; print(os.getcwd())"')
    # Normalize separators for cross-platform comparison.
    assert str(tmp_path).lower().replace("\\", "/") in result.lower().replace("\\", "/")
