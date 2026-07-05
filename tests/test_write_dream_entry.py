"""Tests for WriteDreamEntryTool."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from minion_assist.tools.write_dream_entry import (
    WriteDreamEntryTool,
    _DIARY_END,
    _DIARY_START,
    _DREAMS_HEADER,
    _timestamp,
)


class TestTimestamp:
    def test_returns_string(self) -> None:
        ts = _timestamp(None)
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_with_timezone(self) -> None:
        ts = _timestamp("UTC")
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_invalid_timezone_falls_back(self) -> None:
        # Should not raise, fall back to local time
        ts = _timestamp("Invalid/Timezone")
        assert isinstance(ts, str)
        assert len(ts) > 0


class TestWriteDreamEntryTool:
    def test_schema_name(self) -> None:
        tool = WriteDreamEntryTool(Path("/tmp"))
        assert tool.schema.name == "write_dream_entry"

    def test_schema_requires_entry(self) -> None:
        tool = WriteDreamEntryTool(Path("/tmp"))
        assert "entry" in tool.schema.parameters["required"]

    def test_schema_is_not_read_only(self) -> None:
        tool = WriteDreamEntryTool(Path("/tmp"))
        assert tool.schema.is_read_only is False

    def test_empty_entry_returns_early(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        result = tool.execute(entry="")
        assert "nothing written" in result
        assert not (tmp_path / "DREAMS.md").exists()

    def test_whitespace_only_entry_returns_early(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        result = tool.execute(entry="   \n   ")
        assert "nothing written" in result

    def test_creates_file_from_scratch(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="The moon was a binary star tonight.")
        assert (tmp_path / "DREAMS.md").exists()

    def test_created_file_contains_markers(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="First entry.")
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        assert _DIARY_START in content
        assert _DIARY_END in content

    def test_created_file_contains_entry(self, tmp_path: Path) -> None:
        entry = "Rain drummed a steady recursion against the glass."
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry=entry)
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        assert entry in content

    def test_entry_is_between_markers(self, tmp_path: Path) -> None:
        entry = "Between markers check."
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry=entry)
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        start_idx = content.find(_DIARY_START)
        end_idx = content.find(_DIARY_END)
        entry_idx = content.find(entry)
        assert start_idx < entry_idx < end_idx

    def test_appends_second_entry_before_end_marker(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="First dream.")
        tool.execute(entry="Second dream.")
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        first_idx = content.find("First dream.")
        second_idx = content.find("Second dream.")
        end_idx = content.find(_DIARY_END)
        assert first_idx < second_idx < end_idx

    def test_both_entries_preserved_on_second_write(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="First dream.")
        tool.execute(entry="Second dream.")
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        assert "First dream." in content
        assert "Second dream." in content

    def test_does_not_duplicate_markers(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="Alpha.")
        tool.execute(entry="Beta.")
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        assert content.count(_DIARY_START) == 1
        assert content.count(_DIARY_END) == 1

    def test_repairs_missing_start_marker(self, tmp_path: Path) -> None:
        dreams_path = tmp_path / "DREAMS.md"
        # Write file with end marker only (corrupt state)
        dreams_path.write_text(f"# Dream Diary\n\n{_DIARY_END}\n", encoding="utf-8")
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="Repaired entry.")
        content = dreams_path.read_text(encoding="utf-8")
        assert _DIARY_START in content
        assert "Repaired entry." in content

    def test_repairs_missing_end_marker(self, tmp_path: Path) -> None:
        dreams_path = tmp_path / "DREAMS.md"
        dreams_path.write_text(f"# Dream Diary\n\n{_DIARY_START}\n", encoding="utf-8")
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="Repaired entry.")
        content = dreams_path.read_text(encoding="utf-8")
        assert _DIARY_END in content
        assert "Repaired entry." in content

    def test_return_message_mentions_dreams_md(self, tmp_path: Path) -> None:
        tool = WriteDreamEntryTool(tmp_path)
        result = tool.execute(entry="Something.")
        assert "DREAMS.md" in result

    def test_existing_dreams_md_with_markers_appends_correctly(self, tmp_path: Path) -> None:
        dreams_path = tmp_path / "DREAMS.md"
        dreams_path.write_text(
            f"# Dream Diary\n\n{_DIARY_START}\n\n"
            f"---\n\n*Old timestamp*\n\nOld entry.\n\n{_DIARY_END}\n",
            encoding="utf-8",
        )
        tool = WriteDreamEntryTool(tmp_path)
        tool.execute(entry="New entry.")
        content = dreams_path.read_text(encoding="utf-8")
        assert "Old entry." in content
        assert "New entry." in content
        old_idx = content.find("Old entry.")
        new_idx = content.find("New entry.")
        end_idx = content.find(_DIARY_END)
        assert old_idx < new_idx < end_idx

    def test_timezone_used_in_timestamp(self, tmp_path: Path) -> None:
        # Ensure the tool uses the passed timezone without raising.
        tool = WriteDreamEntryTool(tmp_path, timezone="UTC")
        result = tool.execute(entry="UTC dream.")
        assert "DREAMS.md" in result
        content = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
        assert "UTC dream." in content
