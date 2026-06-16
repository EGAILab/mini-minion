"""Tests for GlobTool."""

from minion_assist.tools.glob import _MAX_RESULTS, GlobTool


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


def test_glob_skips_git_directory(tmp_path):
    """Files inside .git/ must be excluded from results."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("")
    (tmp_path / "real.py").write_text("")
    result = GlobTool().execute(pattern="**/*", path=str(tmp_path))
    assert "real.py" in result
    assert ".git" not in result


def test_glob_caps_results(tmp_path):
    """More than _MAX_RESULTS matches must be capped with a truncation hint."""
    for i in range(_MAX_RESULTS + 10):
        (tmp_path / f"f{i}.py").write_text("")
    result = GlobTool().execute(pattern="*.py", path=str(tmp_path))
    lines = [l for l in result.splitlines() if l and not l.startswith("(")]
    assert len(lines) == _MAX_RESULTS
    assert "capped" in result.lower()
