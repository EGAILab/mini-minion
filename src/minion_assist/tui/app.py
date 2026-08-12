"""MinionApp — the Textual full-screen TUI (Phase 1: core chat shell).

Started via ``minion-assist --tui``. ``minion.py`` builds this app *before*
constructing any ``AgentSession``s (mirroring how it builds ``sessions``
before branching into ``--voice`` mode), because two of this app's bound
methods (``approve_bash``, ``confirm_git``, ``ask_user``, ``approve_codex``)
must already exist to be wired into tool/session construction as
replacements for the REPL's ``print()``/``input()``-based console callbacks
of the same name — those primitives cannot be used once a Textual app has
put the terminal in raw mode. ``configure()`` is then called once, after
every subsystem (sessions, MCP manager, workers, etc.) is built, exactly the
data ``main()``'s REPL loop already closes over.

Threading model
----------------
``AgentSession.send()`` is a blocking synchronous call, so it must not run on
Textual's own asyncio event loop (the UI would freeze for the whole turn).
``run_turn`` is a ``@work(thread=True)`` worker — Textual runs it in a plain
OS thread. Everything that worker thread needs to do to the UI (append a
chat message, show an approval modal and wait for the result) goes through
``App.call_from_thread()``, the only sanctioned way to touch widgets or
await screen results from a non-Textual thread.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.suggester import SuggestFromList
from textual.widgets import Input, Static

from ..agents import AGENTS, resolve
from ..agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    MaxRoundsReached,
    MemoryFlushed,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
)
from ..commands import CommandContext, dispatch_command, parse_command
from ..media import MediaAttachment, describe_attachment, stage_attachment
from ..tools.audit import ApprovalDecision
from .widgets import ApprovalModal, AskUserModal, CodexApprovalModal, ConfirmModal

if TYPE_CHECKING:
    from collections.abc import Sequence

_WELCOME = "Mini-Minion ready. Type /help for commands, /quit to quit."


class MinionApp(App[None]):
    """Full-screen chat TUI. See module docstring for the two-phase build order."""

    CSS = """
    #chat-log {
        height: 1fr;
        padding: 0 1;
    }
    #composer {
        dock: bottom;
    }
    """
    TITLE = "minion-assist"

    def __init__(self) -> None:
        super().__init__()
        # --- Wired in by configure(), called once before run() ---
        self._sessions: dict = {}
        self._agents_cfg: dict = {}
        self._session_store: object = None
        self._mcp_manager: object = None
        self._skills: dict | None = None
        self._short_term: object = None
        self._db: object = None
        self._worker_health: dict | None = None
        self._media_dir: Path | None = None
        self._active_agent_id: str = ""
        self._use_streaming: bool = False
        self._pending_attachments: dict[str, list[MediaAttachment]] = {}
        self._completion_words: list[str] = []
        # --- Per-turn streaming state (single turn in flight at a time — the
        # composer is disabled for the duration of run_turn, so there is
        # never more than one "current" streamed widget to track). ---
        self._streaming_widget: Static | None = None
        self._streaming_text: str = ""

    def configure(
        self,
        *,
        sessions: dict,
        agents_cfg: dict,
        session_store: object,
        mcp_manager: object,
        skills: dict | None,
        short_term: object,
        db: object,
        worker_health: dict | None,
        media_dir: Path,
        active_agent_id: str,
        use_streaming: bool,
        completion_items: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Wire in the real subsystems built by minion.py's main().

        Called exactly once, after every other subsystem (sessions, MCP
        manager, memory workers, etc.) has been constructed — the same
        point at which ``--voice`` mode builds its ``VoiceSession``. Must
        be called before ``run()`` — ``compose()`` reads ``completion_items``
        to build the composer's slash-command suggester.
        """
        self._sessions = sessions
        self._agents_cfg = agents_cfg
        self._session_store = session_store
        self._mcp_manager = mcp_manager
        self._skills = skills
        self._short_term = short_term
        self._db = db
        self._worker_health = worker_health
        self._media_dir = media_dir
        self._active_agent_id = active_agent_id
        self._use_streaming = use_streaming
        self._pending_attachments = {aid: [] for aid in sessions}
        self._completion_words = [value for value, _meta in completion_items]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Input(
            placeholder="Message... (/help for commands)",
            id="composer",
            suggester=SuggestFromList(self._completion_words, case_sensitive=False),
        )

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(Text(_WELCOME, style="bold")))
        for agent_id, cfg_entry in self._agents_cfg.items():
            if cfg_entry.route_prefix:
                agent_name = AGENTS[agent_id].name
                log.mount(Static(Text(f"  {cfg_entry.route_prefix} <message>  -> {agent_name}")))
        self.query_one("#composer", Input).focus()

    # ------------------------------------------------------------------
    # Input handling — mirrors minion.py's REPL loop body (attach commands,
    # routing, slash-command dispatch, then normal send()), adapted to post
    # to widgets instead of print().
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "composer":
            return
        text = event.value
        event.input.value = ""
        self._process_input(text.strip())

    def _process_input(self, user_input: str) -> None:
        if not user_input:
            return

        if user_input.lower() in ("exit", "quit"):
            self.exit()
            return

        # --- Attachment commands — same three as the REPL, same _media_dir. ---
        if user_input.lower().startswith("/attach "):
            paths_str = user_input[len("/attach "):].strip()
            for p in paths_str.split():
                try:
                    att = stage_attachment(Path(p), self._media_dir)
                    self._pending_attachments[self._active_agent_id].append(att)
                    self._log_system(f"Attached: {describe_attachment(att)}")
                except (ValueError, FileNotFoundError) as exc:
                    self._log_system(f"Error: {exc}", error=True)
            return

        if user_input.strip().lower() == "/attachments":
            atts = self._pending_attachments.get(self._active_agent_id, [])
            if atts:
                lines = [f"Pending attachments for {self._active_agent_id}:"]
                lines += [f"  [{i}] {describe_attachment(a)}" for i, a in enumerate(atts, 1)]
                self._log_system("\n".join(lines))
            else:
                self._log_system("No pending attachments.")
            return

        if user_input.strip().lower() == "/clear-attachments":
            self._pending_attachments[self._active_agent_id] = []
            self._log_system("Cleared pending attachments.")
            return

        # --- Routing + slash-command dispatch (same order as the REPL). ---
        agent_id, message = resolve(user_input)
        route_matched = message != user_input
        cmd_text = message if route_matched else user_input

        parsed = parse_command(cmd_text)
        if parsed is not None:
            cmd_token, cmd_args = parsed
            is_lone_route_prefix = any(
                user_input.strip() == cfg.route_prefix
                for cfg in self._agents_cfg.values()
                if cfg.route_prefix
            )
            if not is_lone_route_prefix:
                ctx = CommandContext(
                    raw=user_input,
                    command=cmd_token,
                    args=cmd_args,
                    target_agent_id=agent_id,
                    sessions=self._sessions,
                    agents_cfg=self._agents_cfg,
                    session_store=self._session_store,
                    mcp_manager=self._mcp_manager,
                    skills=self._skills,
                    short_term=self._short_term,
                    db=self._db,
                    worker_health=self._worker_health,
                )
                result = dispatch_command(ctx)
                if result.handled:
                    if result.message:
                        self._log_system(result.message)
                    if result.activate_agent_id:
                        self._active_agent_id = result.activate_agent_id
                    if result.should_exit:
                        self.exit()
                    return
                self._log_system(
                    f"Unknown command '{cmd_token}'. Type /help for commands.", error=True
                )
                return

        # --- Normal agent turn. ---
        self._active_agent_id = agent_id
        atts = self._pending_attachments.get(agent_id, [])
        self._pending_attachments[agent_id] = []

        self._log_user(message)
        self.query_one("#composer", Input).disabled = True
        self.run_turn(agent_id, message, atts or None)

    # ------------------------------------------------------------------
    # Agent turn execution (background thread) + chat log rendering (main
    # thread, always reached via call_from_thread from the worker).
    # ------------------------------------------------------------------

    @work(thread=True)
    def run_turn(
        self, agent_id: str, message: str, attachments: Sequence[MediaAttachment] | None
    ) -> None:
        try:
            self._sessions[agent_id].send(
                message,
                attachments=list(attachments) if attachments else None,
                on_event=self.on_agent_event,
                stream=self._use_streaming,
                channel="tui",
            )
        except Exception as exc:
            agent_name = AGENTS[agent_id].name
            self.call_from_thread(
                self._log_system, f"[Error] {agent_name} failed to respond: {exc}", True
            )
        finally:
            self.call_from_thread(self._finish_turn)

    def _finish_turn(self) -> None:
        composer = self.query_one("#composer", Input)
        composer.disabled = False
        composer.focus()

    def on_agent_event(self, event: object) -> None:
        """Called synchronously, on the worker thread, by session.send()'s
        runner loop — bridges to the main thread for every event."""
        self.call_from_thread(self._apply_event, event)

    def _apply_event(self, event: object) -> None:
        """Render one agent runtime event into the chat log (main thread only).

        Mirrors minion.py's _on_event REPL renderer event-for-event, adapted
        from print() to mounted Static widgets.
        """
        log = self.query_one("#chat-log", VerticalScroll)

        if isinstance(event, StreamingStarted):
            self._streaming_text = ""
            self._streaming_widget = Static("", markup=False)
            log.mount(Static(Text(f"{event.agent_name}:", style="bold magenta")))
            log.mount(self._streaming_widget)
            log.scroll_end(animate=False)
            return

        if isinstance(event, TokenStreamed):
            self._streaming_text += event.token
            if self._streaming_widget is not None:
                self._streaming_widget.update(self._streaming_text)
                log.scroll_end(animate=False)
            return

        was_streaming = self._streaming_widget is not None
        if was_streaming and self._streaming_text:
            # Upgrade the raw streamed text to rendered Markdown now that
            # the turn is complete — matches the REPL's own behaviour of
            # only ever showing raw tokens while they arrive.
            self._streaming_widget.update(RichMarkdown(self._streaming_text))
        self._streaming_widget = None
        self._streaming_text = ""

        if isinstance(event, ThoughtEmitted):
            log.mount(Static(Text(f"{event.agent_name}:", style="bold magenta")))
            log.mount(Static(RichMarkdown(event.text)))
        elif isinstance(event, FinalAnswer):
            if event.text and not was_streaming:
                log.mount(Static(Text(f"{event.agent_name}:", style="bold magenta")))
                log.mount(Static(RichMarkdown(event.text)))
        elif isinstance(event, ToolCalled):
            log.mount(Static(Text(f"  [tool: {event.name}({event.args})]", style="dim")))
        elif isinstance(event, MaxRoundsReached):
            log.mount(Static(Text(f"{event.agent_name}:", style="bold magenta")))
            log.mount(Static(Text(event.message)))
        elif isinstance(event, CompactionStarted):
            log.mount(Static(Text("Compacting session history...", style="dim")))
        elif isinstance(event, CompactionFailed):
            log.mount(Static(Text(f"[Warning] Compaction failed: {event.error}", style="red")))
        elif isinstance(event, MemoryFlushed) and event.status == "failed":
            log.mount(Static(Text(
                f"[Warning] Pre-compaction memory flush failed: {event.detail}", style="red"
            )))

        log.scroll_end(animate=False)

    def _log_user(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(Text("You:", style="bold cyan")))
        log.mount(Static(Text(text)))
        log.scroll_end(animate=False)

    def _log_system(self, text: str, error: bool = False) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(Text(text, style="red" if error else "dim")))
        log.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # Console-callback replacements — one pair (sync entry point called
    # from any thread + async implementation run on the main thread) per
    # minion.py console callback. Each sync method is passed directly as
    # the corresponding tool/provider constructor argument in minion.py,
    # in place of _console_approve/_console_confirm/_console_ask_user/
    # _console_approve_codex.
    # ------------------------------------------------------------------

    def approve_bash(self, command: str) -> ApprovalDecision:
        return self.call_from_thread(self._approve_bash_async, command)

    async def _approve_bash_async(self, command: str) -> ApprovalDecision:
        return await self.push_screen_wait(ApprovalModal(command))

    def confirm_git(self, command: str) -> bool:
        return self.call_from_thread(self._confirm_git_async, command)

    async def _confirm_git_async(self, command: str) -> bool:
        return await self.push_screen_wait(ConfirmModal(command))

    def ask_user(self, question: str) -> str:
        return self.call_from_thread(self._ask_user_async, question)

    async def _ask_user_async(self, question: str) -> str:
        return await self.push_screen_wait(AskUserModal(question))

    def approve_codex(self, method: str, params: dict) -> str:
        # Same summary-extraction logic as minion.py's _console_approve_codex.
        cmd = (
            params.get("command")
            or params.get("cmd")
            or (params.get("arguments") or {}).get("command")
            or ""
        )
        if cmd:
            summary = f"Command: {cmd[:200]}"
        elif params:
            summary = f"Params: {json.dumps(params, ensure_ascii=False)[:200]}"
        else:
            summary = ""
        return self.call_from_thread(self._approve_codex_async, method, summary)

    async def _approve_codex_async(self, method: str, summary: str) -> str:
        return await self.push_screen_wait(CodexApprovalModal(method, summary))
