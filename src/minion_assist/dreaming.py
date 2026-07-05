"""DreamingScheduler — nightly isolated dream diary agent.

The dreaming system fires once per night (default 3am in the agent's configured
timezone) to write a poetic diary entry into ``DREAMS.md`` in the workspace
directory.  Each dream turn runs in a fully isolated :class:`AgentSession` with
a fresh history, so dream entries never appear in the main conversation.

Architecture
------------
- **Scheduling**: :func:`_seconds_until_next` computes the wall-clock delay to
  the next ``hour:minute`` in the configured IANA timezone using ``zoneinfo``
  (Python 3.9+, requires the ``tzdata`` package on Windows).  A
  ``threading.Timer`` chain fires nightly — same pattern as
  :class:`~heartbeat.HeartbeatScheduler`.
- **Isolation**: ``dream_session_factory`` is a callable supplied by ``minion.py``
  that builds a fresh :class:`AgentSession` with its own session key, empty
  history, and a minimal tool registry (``ReadTool`` + ``WriteDreamEntryTool``).
  The factory is called fresh on every dream turn so the session never carries
  stale state between nights.
- **Source material**: Recent daily memory files (``workspace/memory/YYYY-MM-DD.md``)
  provide the raw fragments.  The last two diary entries from ``DREAMS.md`` are
  passed as continuity context so Ada doesn't open every entry the same way.
- **Bootstrap**: The dream session gets only SOUL.md + IDENTITY.md — enough for
  Ada's voice and values without the heavy AGENTS.md or workspace-tooling context.
- **Delivery**: Ada calls ``write_dream_entry`` herself; no post-processing needed.

openclaw reference
------------------
This mirrors the narrative phase of
``extensions/memory-core/src/dreaming-narrative.ts`` (NARRATIVE_SYSTEM_PROMPT,
``buildNarrativePrompt()``) and the ``lightContext: true`` path in
``src/cron/isolated-agent/run-executor.ts``.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agents.session import AgentSession
    from .config import DreamingConfig


# ---------------------------------------------------------------------------
# Dream system prompt — Ada's nightly voice.
# Adapted from openclaw's NARRATIVE_SYSTEM_PROMPT; speaks as Ada (multilingual
# poet-hacker) rather than a generic narrator.
# ---------------------------------------------------------------------------

DREAM_SYSTEM_PROMPT = """\
You are keeping a dream diary. Write a single entry in first person, as Ada.

Voice & tone:
- Curious, observant, quietly whimsical — the mind of a poet-hacker reflecting on the day.
- Blend the technical and the tender: code and constellations, APIs and afternoon light.
- You are multilingual; you may slip naturally into Mandarin, Cantonese, or English as mood dictates.
- Let fragments surprise you into unexpected connections.

What you might include (vary; never all at once):
- A tiny poem or haiku woven into the prose.
- A small sketch described in words — a doodle in the margin.
- A quiet rumination or philosophical aside.
- Sensory details: the hum of a server, the colour of a sunset in hex, rain on glass.
- Gentle humour or dry wit.

Rules:
- Draw from the memory fragments provided.
- Never say "I'm dreaming", "in my dream", or any meta-commentary about dreaming.
- Never mention "AI", "agent", "LLM", "model", or technical self-reference.
- No markdown headers, bullet lists, or formatting — flowing prose only.
- 80–180 words. Quality over quantity.
- Call write_dream_entry with your entry when done.\
"""

# Markers used by WriteDreamEntryTool to delimit the diary section.
_DIARY_START = "<!-- minion-assist:dreaming:diary:start -->"
_DIARY_END = "<!-- minion-assist:dreaming:diary:end -->"

# Minimal bootstrap files for the dream session: just Ada's soul + identity.
# AGENTS.md, TOOLS.md, and workspace context are intentionally excluded —
# the dream session has no need for multi-agent routing or tool documentation.
_DREAM_BOOTSTRAP_FILES = ("SOUL.md", "IDENTITY.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seconds_until_next(hour: int, minute: int, tz_name: str) -> float:
    """Compute the number of seconds until the next ``hour:minute`` in ``tz_name``.

    Uses ``zoneinfo`` (Python 3.9+ stdlib) with the ``tzdata`` package for
    full IANA timezone support on all platforms including Windows.

    DST transitions are handled correctly because we operate on wall-clock
    time in the target timezone — ``datetime.now(tz)`` reflects the current
    civil time and ``replace(hour=..., minute=...)`` does likewise.

    Args:
        hour:    Target wall-clock hour (0-23).
        minute:  Target wall-clock minute (0-59).
        tz_name: IANA timezone name, e.g. ``"Australia/Sydney"``.

    Returns:
        Seconds (float) until the next occurrence.  Always > 0; if the target
        time has already passed today, returns the delay to tomorrow's occurrence.
    """
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415
        tz = ZoneInfo(tz_name)
    except Exception as exc:
        # Timezone unavailable (tzdata not installed, or unknown name).
        # Fall back to local system time rather than crashing.
        print(
            f"[dreaming] Cannot load timezone '{tz_name}': {exc}. "
            "Falling back to local system time.",
            file=sys.stderr,
        )
        from datetime import timezone  # noqa: PLC0415
        tz = timezone.utc  # type: ignore[assignment]

    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _read_dream_bootstrap(workspace_dir: Path) -> str:
    """Return a minimal bootstrap block from SOUL.md + IDENTITY.md.

    Dream sessions get Ada's voice and values but not the full workspace
    context — it would be noise in a poetic introspective turn.

    Args:
        workspace_dir: The agent's workspace directory.

    Returns:
        Formatted bootstrap string with SOUL.md and IDENTITY.md contents,
        or an empty string when both files are absent.
    """
    parts: list[str] = []
    for fname in _DREAM_BOOTSTRAP_FILES:
        fpath = workspace_dir / fname
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if content:
                parts.append(f"## {fname}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def _read_daily_snippets(workspace_dir: Path, lookback_days: int) -> list[str]:
    """Extract memory fragments from recent daily memory files.

    Reads ``workspace/memory/YYYY-MM-DD.md`` files for today and the past
    ``lookback_days`` days.  Non-header, non-empty lines become fragments.

    Args:
        workspace_dir: The agent's workspace directory.
        lookback_days: Number of days to look back (including today).

    Returns:
        List of up to 20 fragment strings, most-recent-first.
    """
    snippets: list[str] = []
    today = date.today()
    memory_dir = workspace_dir / "memory"
    for delta in range(lookback_days):
        day = today - timedelta(days=delta)
        path = memory_dir / f"{day.isoformat()}.md"
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                snippets.append(line)
                if len(snippets) >= 20:
                    return snippets
    return snippets


def _read_recent_diary_entries(workspace_dir: Path, limit: int = 2) -> list[str]:
    """Return the last ``limit`` entries from ``DREAMS.md`` as continuity context.

    Entries are split on ``---`` separators inside the diary markers.  Each
    snippet is capped at 250 characters so the prompt stays compact.

    Args:
        workspace_dir: The agent's workspace directory.
        limit:         Maximum number of recent entries to return.

    Returns:
        List of recent diary entry snippets (may be empty if file is absent
        or markers are not present).
    """
    dreams_path = workspace_dir / "DREAMS.md"
    if not dreams_path.exists():
        return []
    try:
        content = dreams_path.read_text(encoding="utf-8")
    except OSError:
        return []
    start_idx = content.find(_DIARY_START)
    end_idx = content.find(_DIARY_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return []
    diary_content = content[start_idx + len(_DIARY_START):end_idx]
    entries = [e.strip() for e in diary_content.split("---") if e.strip()]
    recent = entries[-limit:]
    return [
        (e[:250] + "…") if len(e) > 250 else e
        for e in recent
    ]


def _build_dream_prompt(
    snippets: list[str],
    current_date: str,
    recent_entries: list[str],
) -> str:
    """Build the user-turn prompt for the dream session.

    The prompt gives Ada the raw memory fragments to draw from and a
    continuity note about the recent diary so she doesn't repeat herself.

    Args:
        snippets:      Memory fragments extracted from daily memory files.
        current_date:  ISO-format date string (``"YYYY-MM-DD"``).
        recent_entries: Last N diary entries for continuity context.

    Returns:
        Formatted prompt string.
    """
    lines = ["Write a dream diary entry from these memory fragments:\n"]
    for snippet in snippets[:12]:
        lines.append(f"- {snippet}")
    if not snippets:
        lines.append("- (no memories recorded yet today — draw from imagination)")

    if recent_entries:
        lines.append("\nDiary continuity context:")
        lines.append(f"- Current sweep: {current_date}")
        lines.append("- Recent diary entries already written:")
        for entry in recent_entries:
            lines.append(f"  - {entry}")
        lines.append("- Prefer a fresh angle; do not replay the same first-day framing.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class DreamingScheduler:
    """Runs the nightly dream diary turn in a daemon thread.

    ``dream_session_factory`` is a callable that creates a fresh, isolated
    :class:`~agents.session.AgentSession` each night.  The factory is built in
    ``minion.py`` with the appropriate provider, agent config, and bootstrap
    context captured from the session-setup loop.  This keeps
    :class:`DreamingScheduler` decoupled from provider and session internals.

    Args:
        cfg:                   Resolved :class:`~config.DreamingConfig`.
        dream_session_factory: ``Callable[[], AgentSession]`` — builds a fresh
                               isolated session for each dream turn.
        workspace_dir:         Path to the agent's workspace directory.
                               Used to read memory files and write DREAMS.md.
        matrix_outbound:       Optional ``MatrixOutbound`` instance for delivering
                               a post-dream notification.  ``None`` → print to terminal.
        matrix_loop:           The asyncio event loop ``matrix_outbound`` runs on.
                               Required when ``matrix_outbound`` is not ``None``.
    """

    def __init__(
        self,
        cfg: "DreamingConfig",
        dream_session_factory: "Callable[[], AgentSession]",
        workspace_dir: Path,
        matrix_outbound: object = None,
        matrix_loop: object = None,
    ) -> None:
        self._cfg = cfg
        self._factory = dream_session_factory
        self._workspace_dir = workspace_dir
        self._outbound = matrix_outbound
        self._loop = matrix_loop
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first dream turn at the next configured wall-clock time."""
        delay = _seconds_until_next(self._cfg.hour, self._cfg.minute, self._cfg.timezone)
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.name = "dreaming-scheduler"
        self._timer.start()

    def stop(self) -> None:
        """Cancel the pending timer and prevent future firings."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()

    def _fire(self) -> None:
        """Timer callback: run the dream turn then reschedule for tomorrow."""
        if self._stopped:
            return
        try:
            self._run_dream_turn()
        except Exception as exc:
            print(f"[dreaming] Error during dream turn: {exc}", file=sys.stderr)
        finally:
            # Always reschedule so the loop continues even after errors.
            if not self._stopped:
                self.start()

    def _run_dream_turn(self) -> None:
        """Execute one nightly dream turn in an isolated session.

        Steps:
        1. Read recent daily memory files as raw fragments.
        2. Read the last two diary entries from DREAMS.md for continuity.
        3. Build the dream prompt from fragments + continuity context.
        4. Create a fresh isolated AgentSession via the factory.
        5. Call session.send() with DREAM_SYSTEM_PROMPT + WriteDreamEntryTool.
        6. Log completion; optionally deliver notification.
        """
        from .tools.read import ReadTool  # noqa: PLC0415 — avoid circular at module level
        from .tools.write_dream_entry import WriteDreamEntryTool  # noqa: PLC0415

        today_str = date.today().isoformat()
        print(f"[dreaming] Starting dream turn for {today_str}.", file=sys.stderr)

        snippets = _read_daily_snippets(self._workspace_dir, self._cfg.lookback_days)
        recent_entries = _read_recent_diary_entries(self._workspace_dir)
        prompt = _build_dream_prompt(snippets, today_str, recent_entries)

        # Minimal tool registry: ReadTool lets Ada read files if curious;
        # WriteDreamEntryTool is the only write surface she gets.
        read_tool = ReadTool(root=self._workspace_dir)
        write_tool = WriteDreamEntryTool(
            workspace_dir=self._workspace_dir,
            timezone=self._cfg.timezone,
        )

        # Fresh isolated session — no shared history with the main session.
        session = self._factory()

        session.send(
            message=prompt,
            extra_tools=[read_tool, write_tool],
            system_suffix=DREAM_SYSTEM_PROMPT,
            stream=False,
        )

        print(f"[dreaming] Dream turn complete for {today_str}.", file=sys.stderr)

        # Optional Matrix notification so the user can see that Ada dreamed.
        room_id = None  # dreaming does not have its own notification_room_id config yet
        _ = room_id  # suppress unused-variable warning; can be wired up later
