"""Agent/session switcher sidebar (TUI Phase 3).

Purely a display + selection surface — it never mutates session state
itself. Selecting a row posts AgentSelected/SessionSelected, which
tui/app.py turns into the exact same /switch and /session command text the
composer already accepts, so the actual switching logic lives in exactly
one place: commands.py's dispatch_command(), shared with the REPL.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, ListItem, ListView, Static


class AgentSelected(Message):
    """Posted when a row in the Agents list is selected."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__()


class SessionSelected(Message):
    """Posted when a row in the Sessions list is selected.

    index is 1-based, matching the numbering /session <N> already expects
    (and that the sidebar's own labels display) — not a 0-based list index.
    """

    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__()


class _AgentListItem(ListItem):
    def __init__(self, agent_id: str, label: str) -> None:
        super().__init__(Label(label))
        self.agent_id = agent_id


class _SessionListItem(ListItem):
    def __init__(self, index: int, label: str) -> None:
        super().__init__(Label(label))
        self.session_index = index


class Sidebar(Vertical):
    """Agent list (top) + active agent's session list (bottom)."""

    DEFAULT_CSS = """
    Sidebar {
        width: 32;
        border-left: solid $accent;
        padding: 0 1;
    }
    Sidebar ListView {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 — matches Textual's own `id` param name
        super().__init__(id=id)
        self._agent_list: ListView = ListView()
        self._session_list: ListView = ListView()

    def compose(self) -> ComposeResult:
        yield Static(Text("Agents", style="bold"))
        yield self._agent_list
        yield Static(Text("Sessions", style="bold"))
        yield self._session_list

    def update_agents(self, agents: list[tuple[str, str]], active_agent_id: str) -> None:
        """Replace the Agents list.

        Args:
            agents: (agent_id, display_label) pairs, e.g.
                ("main", "Ada (42 turns)").
            active_agent_id: Marked with a leading arrow.
        """
        self._agent_list.clear()
        for agent_id, label in agents:
            marker = "> " if agent_id == active_agent_id else "  "
            self._agent_list.append(_AgentListItem(agent_id, marker + label))

    def update_sessions(self, sessions: list[tuple[int, str]], current_index: int | None) -> None:
        """Replace the Sessions list.

        Args:
            sessions: (1-based index, display_label) pairs, in the same
                order /session's own bare listing uses (most recent first).
            current_index: The active session's 1-based index, or None if
                unknown — marked with a leading arrow.
        """
        self._session_list.clear()
        for index, label in sessions:
            marker = "> " if index == current_index else "  "
            self._session_list.append(_SessionListItem(index, marker + label))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        item = event.item
        if isinstance(item, _AgentListItem):
            self.post_message(AgentSelected(item.agent_id))
        elif isinstance(item, _SessionListItem):
            self.post_message(SessionSelected(item.session_index))
