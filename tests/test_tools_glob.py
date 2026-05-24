"""Tests for GlobTool."""

from mini_minion.tools.glob import GlobTool


def test_glob_finds_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = GlobTool().execute(pattern="*.py", path=str(tmp_path))
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


def test_glob_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.py").write_text("")
    (sub / "nested.py").write_text("")
    result = GlobTool().execute(pattern="**/*.py", path=str(tmp_path))
    assert "root.py" in result
    assert "nested.py" in result


def test_glob_no_matches(tmp_path):
    result = GlobTool().execute(pattern="*.xyz", path=str(tmp_path))
    assert result == "(no matches)"


def test_glob_excludes_directories(tmp_path):
    (tmp_path / "mydir").mkdir()
    (tmp_path / "file.py").write_text("")
    result = GlobTool().execute(pattern="*", path=str(tmp_path))
    assert "file.py" in result
    assert "mydir" not in result


def test_glob_returns_one_path_per_line(tmp_path):
    for name in ("x.py", "y.py", "z.py"):
        (tmp_path / name).write_text("")
    result = GlobTool().execute(pattern="*.py", path=str(tmp_path))
    lines = result.strip().splitlines()
    assert len(lines) == 3
