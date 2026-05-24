"""Tests for WriteTool."""

from mini_minion.tools.write import WriteTool


def test_write_creates_file(tmp_path):
    target = tmp_path / "out.txt"
    result = WriteTool().execute(path=str(target), content="hello world")
    assert target.read_text() == "hello world"
    assert str(target) in result


def test_write_result_is_status_not_content(tmp_path):
    target = tmp_path / "out.txt"
    result = WriteTool().execute(path=str(target), content="hello world")
    assert "hello world" not in result


def test_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.txt"
    WriteTool().execute(path=str(target), content="deep")
    assert target.read_text() == "deep"


def test_write_overwrites_existing(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("old content")
    WriteTool().execute(path=str(target), content="new content")
    assert target.read_text() == "new content"


def test_write_reports_char_count(tmp_path):
    target = tmp_path / "count.txt"
    result = WriteTool().execute(path=str(target), content="abc")
    assert "3" in result


def test_write_empty_content(tmp_path):
    target = tmp_path / "empty.txt"
    WriteTool().execute(path=str(target), content="")
    assert target.read_text() == ""


def test_write_unicode_content(tmp_path):
    target = tmp_path / "unicode.txt"
    content = "こんにちは 🌍"
    WriteTool().execute(path=str(target), content=content)
    assert target.read_text(encoding="utf-8") == content
