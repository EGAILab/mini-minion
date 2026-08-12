"""Modal screens for tui/app.py.

Two categories:

1. Console-callback replacements — the plain REPL's ``_console_approve``/
   ``_console_confirm``/``_console_ask_user``/``_console_approve_codex``
   (in ``minion.py``) use ``print()``/``input()``. Those don't work inside
   a Textual app — Textual puts the terminal in raw mode and owns all
   keyboard input itself, so a blocking ``input()`` call would never see a
   keystroke. These ``ModalScreen`` subclasses are the TUI-native
   equivalent of the same four prompts, with identical decision semantics
   (same options, same defaults) — ``tui/app.py`` wires them in via
   ``push_screen_wait`` instead of the console versions.

2. ``AttachFilePickerModal`` (Phase 2) — an interactive file browser for
   ``/attach`` used with no path argument, wrapping Textual's built-in
   ``DirectoryTree`` widget instead of requiring the user to type a full
   path.

All dynamic text shown in these modals (shell commands, tool call arguments,
questions) comes from the LLM or the user's own input and is rendered via
``rich.text.Text`` rather than an f-string interpolated into a Rich markup
string — ``Text`` never parses its content as markup, so arbitrary text
(which may contain literal ``[`` / ``]`` characters, e.g. a Python list
literal in a shell command) can never raise a ``MarkupError`` or be
misinterpreted as a style tag.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static

from ..tools.audit import ApprovalDecision

_MODAL_CSS = """
Vertical {
    width: auto;
    height: auto;
    max-width: 80%;
    border: round $accent;
    background: $panel;
    padding: 1 2;
}
Static {
    margin-bottom: 1;
}
Button {
    margin-right: 1;
}
"""


class ApprovalModal(ModalScreen[ApprovalDecision]):
    """Bash command approval — mirrors minion.py's ``_console_approve``.

    Same four options and the same default (Enter = allow once).
    """

    CSS = _MODAL_CSS
    BINDINGS = [
        ("1", "choose('allow_once')", "Allow once"),
        ("2", "choose('allow_session')", "Allow session"),
        ("3", "choose('deny')", "Deny"),
        ("4", "choose('always_deny')", "Always deny"),
        ("enter", "choose('allow_once')", "Allow once"),
        ("escape", "choose('deny')", "Deny"),
    ]

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Bash command approval")
            yield Static(Text(self._command))
            yield Button("[1] Allow once", id="allow_once", variant="primary")
            yield Button("[2] Allow session", id="allow_session")
            yield Button("[3] Deny", id="deny", variant="error")
            yield Button("[4] Always deny", id="always_deny", variant="error")

    def action_choose(self, decision: str) -> None:
        self.dismiss(ApprovalDecision(decision))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.dismiss(ApprovalDecision(event.button.id))


class ConfirmModal(ModalScreen[bool]):
    """Git-commit y/n confirmation — mirrors minion.py's ``_console_confirm``."""

    CSS = _MODAL_CSS
    BINDINGS = [
        ("y", "choose(True)", "Yes"),
        ("n", "choose(False)", "No"),
        ("enter", "choose(False)", "No"),
        ("escape", "choose(False)", "No"),
    ]

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Run this git command?")
            yield Static(Text(self._command))
            yield Button("Yes", id="yes", variant="success")
            yield Button("No", id="no", variant="error")

    def action_choose(self, value: bool) -> None:
        self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class AskUserModal(ModalScreen[str]):
    """Free-text prompt for AskUserTool — mirrors minion.py's ``_console_ask_user``."""

    CSS = _MODAL_CSS

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(self._question))
            yield Input(id="ask-user-input")

    def on_mount(self) -> None:
        self.query_one("#ask-user-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Stop propagation: MinionApp.on_input_submitted only acts on its own
        # "composer" Input, but stopping here is the explicit, correct thing
        # to do rather than relying on that id check as the only guard.
        event.stop()
        self.dismiss(event.value)


class CodexApprovalModal(ModalScreen[str]):
    """Codex built-in tool permission y/n — mirrors ``_console_approve_codex``.

    Returns ``"approve"`` or ``"deny"`` (matching the string contract
    ``CodexProvider`` expects — unlike the other three modals, this one is
    not backed by ``ApprovalDecision``, matching the console version's own
    return type exactly).
    """

    CSS = _MODAL_CSS
    BINDINGS = [
        ("y", "choose('approve')", "Approve"),
        ("n", "choose('deny')", "Deny"),
        ("enter", "choose('approve')", "Approve"),
        ("escape", "choose('deny')", "Deny"),
    ]

    def __init__(self, method: str, summary: str) -> None:
        super().__init__()
        self._method = method
        self._summary = summary

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(f"Codex permission request: {self._method}"))
            if self._summary:
                yield Static(Text(self._summary))
            yield Button("Approve", id="approve", variant="success")
            yield Button("Deny", id="deny", variant="error")

    def action_choose(self, value: str) -> None:
        self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.dismiss(event.button.id)


class AttachFilePickerModal(ModalScreen["Path | None"]):
    """Interactive file browser for ``/attach`` used with no path argument.

    Enter (or double-click) on a file selects it and dismisses with its
    path; Escape cancels and dismisses with ``None``. Directories expand/
    collapse as normal DirectoryTree navigation — only file selection
    dismisses this screen.
    """

    CSS = _MODAL_CSS + """
    AttachFilePickerModal DirectoryTree {
        width: 80;
        height: 20;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self._start_path = start_path or Path.cwd()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select a file to attach (Esc to cancel)")
            yield DirectoryTree(str(self._start_path))

    def on_mount(self) -> None:
        self.query_one(DirectoryTree).focus()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        self.dismiss(event.path)

    def action_cancel(self) -> None:
        self.dismiss(None)
