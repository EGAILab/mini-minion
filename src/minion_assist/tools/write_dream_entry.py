"""WriteDreamEntryTool — append a diary entry to DREAMS.md.

DREAMS.md is written exclusively during nightly isolated dream sessions;
it is never registered in the default tool registry and must not be callable
during normal conversation.  The dreaming scheduler injects this tool as an
``extra_tool`` for the duration of the dream turn only.

File format
-----------
The diary section is delimited by HTML comment markers so the content is
portable and parseable without a custom format::

    # Dream Diary

    <!-- minion-assist:dreaming:diary:start -->
    ---

    *July 5, 2026 at 3:00 AM AEST*

    Prose entry...

    <!-- minion-assist:dreaming:diary:end -->

The markers are created automatically on first use.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from .base import Tool, ToolSchema

_DIARY_START = "<!-- minion-assist:dreaming:diary:start -->"
_DIARY_END = "<!-- minion-assist:dreaming:diary:end -->"

_DREAMS_HEADER = """\
# Dream Diary

<!-- minion-assist:dreaming:diary:start -->
<!-- minion-assist:dreaming:diary:end -->
"""


def _timestamp(timezone: str | None) -> str:
    """Format the current wall-clock time as a diary timestamp.

    Uses the provided IANA timezone name when possible; falls back to local
    time if zoneinfo is unavailable or the timezone name is unrecognised.

    Args:
        timezone: IANA timezone name, e.g. ``"Australia/Sydney"``.

    Returns:
        Human-readable timestamp string, e.g.
        ``"July 5, 2026 at 3:00 AM AEST"``.
    """
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: PLC0415
        tz = ZoneInfo(timezone) if timezone else None
        now = datetime.now(tz) if tz else datetime.now()
    except (ImportError, Exception):
        now = datetime.now()
    return now.strftime("%-d %B %Y at %-I:%M %p %Z").strip() if os.name != "nt" else now.strftime(
        "%#d %B %Y at %#I:%M %p %Z"
    ).strip()


class WriteDreamEntryTool(Tool):
    """Append a diary entry to ``DREAMS.md`` in the workspace.

    This tool is injected only during nightly dream sessions.  Calling it
    outside that context is an error — register it via ``extra_tools`` in
    the dreaming scheduler, not in the default registry.

    Args:
        workspace_dir: The agent's workspace directory.  ``DREAMS.md`` is
            written directly inside this directory (alongside ``MEMORY.md``).
        timezone: Optional IANA timezone name used to format the timestamp,
            e.g. ``"Australia/Sydney"``.  Defaults to local system time.
    """

    def __init__(self, workspace_dir: Path, timezone: str | None = None) -> None:
        self._workspace_dir = workspace_dir
        self._timezone = timezone

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_dream_entry",
            description=(
                "Append your dream diary entry to DREAMS.md. "
                "Call exactly once per dream session with the finished prose. "
                "Do not include markdown headers or bullet points — flowing "
                "prose only, 80–180 words."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "The diary entry prose (80-180 words, no markdown formatting).",
                    },
                },
                "required": ["entry"],
            },
            is_read_only=False,
        )

    def execute(self, **kwargs: object) -> str:
        """Append *entry* to ``DREAMS.md`` between the diary markers.

        Args:
            entry (str): Diary prose to append.

        Returns:
            str: Confirmation with the path written.
        """
        entry = str(kwargs.get("entry", "")).strip()
        if not entry:
            return "[write_dream_entry] Empty entry — nothing written."

        dreams_path = self._workspace_dir / "DREAMS.md"

        # Bootstrap the file with markers if it doesn't exist yet.
        if not dreams_path.exists():
            dreams_path.write_text(_DREAMS_HEADER, encoding="utf-8")

        content = dreams_path.read_text(encoding="utf-8")

        # Locate the diary markers; create them if absent (file was hand-edited).
        start_idx = content.find(_DIARY_START)
        end_idx = content.find(_DIARY_END)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            content = content.rstrip() + f"\n\n{_DIARY_START}\n{_DIARY_END}\n"
            start_idx = content.find(_DIARY_START)
            end_idx = content.find(_DIARY_END)

        ts = _timestamp(self._timezone)
        diary_entry = f"\n---\n\n*{ts}*\n\n{entry}\n\n"

        # Insert just before the closing marker.
        new_content = content[:end_idx] + diary_entry + content[end_idx:]
        dreams_path.write_text(new_content, encoding="utf-8")

        rel = os.path.relpath(dreams_path, self._workspace_dir)
        return f"[write_dream_entry] Dream diary entry written to {rel}."
