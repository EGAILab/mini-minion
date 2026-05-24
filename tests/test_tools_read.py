"""Tests for ReadTool."""

from mini_minion.tools.read import ReadTool


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line one\nline two\nline three\n")
    result = ReadTool().execute(path=str(f))
    assert "1: line one" in result
    assert "2: line two" in result
    assert "3: line three" in result


def test_read_file_with_offset_and_limit(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)))
    result = ReadTool().execute(path=str(f), offset=3, limit=3)
    assert "3: line 3" in result
    assert "4: line 4" in result
    assert "5: line 5" in result
    assert "2: line 2" not in result
    assert "6: line 6" not in result


def test_read_file_pagination_hint(tmp_path):
    f = tmp_path / "paged.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 21)))
    result = ReadTool().execute(path=str(f), limit=5)
    assert "offset=6" in result


def test_read_file_no_pagination_hint_at_eof(tmp_path):
    f = tmp_path / "short.txt"
    f.write_text("only one line")
    result = ReadTool().execute(path=str(f))
    assert "offset=" not in result


def test_read_file_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    result = ReadTool().execute(path=str(f))
    # No lines to show — result should be empty string or benign
    assert "offset=" not in result


def test_read_directory(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    result = ReadTool().execute(path=str(tmp_path))
    assert "a.py" in result
    assert "b.txt" in result
    assert "subdir/" in result


def test_read_empty_directory(tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    result = ReadTool().execute(path=str(empty))
    assert "empty" in result.lower()


def test_read_file_not_found(tmp_path):
    result = ReadTool().execute(path=str(tmp_path / "missing.txt"))
    assert "not found" in result.lower()


def test_read_offset_beyond_eof(tmp_path):
    f = tmp_path / "short.txt"
    f.write_text("line one\nline two\n")
    result = ReadTool().execute(path=str(f), offset=100)
    # Should not crash; returns empty or benign output
    assert isinstance(result, str)
