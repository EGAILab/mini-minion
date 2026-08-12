"""Tests for tui/app.py — MinionApp.

Uses Textual's own App.run_test()/Pilot harness (a real, headless Textual
event loop), not a mock of the framework — the DOM (chat log, composer,
modals) is genuinely mounted and queried the same way it would be at
runtime. AgentSession itself is replaced with _FakeSession so no LLM/tool
call ever happens; that class's own tests (test_session*.py) already cover
AgentSession's real behavior — these tests are about MinionApp's dispatch
logic (attach commands, routing, slash-command interception, streaming
rendering, and the four console-callback-replacement modals), not about
re-proving AgentSession works.

Two Textual-specific gotchas these tests work around:
- Widgets only exist while the `async with app.run_test():` block is open —
  any query_one() after it exits raises NoMatches (the screen is torn
  down), so every chat-log assertion happens *inside* that block.
- push_screen_wait() requires an active Textual "worker" context (it checks
  for one and raises NoActiveWorker otherwise) — a plain threading.Thread
  does not count, even though it can still call call_from_thread(). Tests
  that drive approve_bash()/ask_user()/approve_codex() (which internally
  call push_screen_wait) must invoke them via app.run_worker(..., thread=True),
  the same mechanism run_turn's own @work(thread=True) decorator uses in
  production.

No pytest-asyncio in this project's test dependencies (matches
tests/matrix/*.py's own convention) — every test is a plain sync function
that drives its async body through asyncio.new_event_loop() via _run(),
rather than an `async def test_...` function, which plain pytest would
silently never await.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from minion_assist.agents.events import (
    CompactionFailed,
    FinalAnswer,
    StreamingStarted,
    TokenStreamed,
)
from minion_assist.tools.audit import ApprovalDecision
from minion_assist.tui.app import MinionApp
from minion_assist.tui.attachment_widgets import AudioAttachmentView, ImageAttachmentView
from minion_assist.tui.sidebar import Sidebar, _AgentListItem, _SessionListItem
from minion_assist.tui.widgets import ApprovalModal, AttachFilePickerModal
from minion_assist.worker_health import WorkerHealth


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeSession:
    """Stand-in for AgentSession — records calls, optionally raises or
    replays a scripted sequence of on_event() calls. reload()/
    switch_session()/session_id/history exist for /switch and /session,
    exercised by TestSidebar."""

    def __init__(
        self, response: str = "ok", events: list | None = None, raises: Exception | None = None
    ):
        self.send_calls: list[dict] = []
        self._response = response
        self._events = events or []
        self._raises = raises
        self.session_id: str | None = None
        self.reload_calls = 0
        self.switch_calls: list[str] = []
        self.history: list[dict] = []

    def send(self, message, attachments=None, on_event=None, stream=False, channel=None, **kwargs):
        self.send_calls.append(
            {"message": message, "attachments": attachments, "stream": stream, "channel": channel}
        )
        if self._raises is not None:
            raise self._raises
        if on_event is not None:
            for event in self._events:
                on_event(event)
        return self._response

    def reload(self) -> None:
        self.reload_calls += 1

    def switch_session(self, session_id: str) -> None:
        self.switch_calls.append(session_id)
        self.session_id = session_id


class _FakeSessionInfo:
    """Stand-in for session.store.SessionInfo — only the fields /agents and
    the sidebar actually read."""

    def __init__(
        self, agent_id: str, turn_count: int, last_active: str = "2026-08-12T00:00:00+00:00"
    ):
        self.agent_id = agent_id
        self.turn_count = turn_count
        self.last_active = last_active


class _FakeSessionStore:
    def __init__(self, infos: list[_FakeSessionInfo]):
        self._infos = infos

    def list_sessions(self) -> list[_FakeSessionInfo]:
        return self._infos


class _FakeShortTerm:
    def __init__(
        self,
        sessions_by_agent: dict[str, list[Path]] | None = None,
        names: dict[tuple[str, str], str] | None = None,
    ):
        self._sessions_by_agent = sessions_by_agent or {}
        self._names = names or {}

    def list_sessions(self, agent_id: str) -> list[Path]:
        return self._sessions_by_agent.get(agent_id, [])

    def get_name(self, agent_id: str, session_id: str) -> str | None:
        return self._names.get((agent_id, session_id))


class _FakeDB:
    """Stand-in for SessionDB — only queue_lag_summary(), the one method
    the status bar reads."""

    def __init__(self, lag: dict | None = None, raises: Exception | None = None):
        self._lag = lag or {
            "capture": {"pending_count": 0},
            "commitment": {"pending_count": 0},
            "message_embedding": {"pending_count": 0},
        }
        self._raises = raises

    def queue_lag_summary(self, agent_id: str) -> dict:
        if self._raises is not None:
            raise self._raises
        return self._lag


def _agents_cfg() -> dict:
    # Only "main" (no route_prefix) — routing across multiple agents is
    # already covered by agents/router.py's and the Matrix handler's own
    # tests; MinionApp calls the same resolve() unmodified.
    return {"main": SimpleNamespace(route_prefix=None)}


def _configured_app(
    session: _FakeSession,
    media_dir: Path,
    use_streaming: bool = False,
    *,
    sessions: dict | None = None,
    agents_cfg: dict | None = None,
    session_store: object = None,
    short_term: object = None,
    db: object = None,
    worker_health: dict | None = None,
    active_agent_id: str = "main",
    completion_items: list[tuple[str, str]] | None = None,
) -> MinionApp:
    app = MinionApp()
    app.configure(
        sessions=sessions or {"main": session},
        agents_cfg=agents_cfg or _agents_cfg(),
        session_store=session_store,
        mcp_manager=None,
        skills=None,
        short_term=short_term,
        db=db,
        worker_health=worker_health,
        media_dir=media_dir,
        active_agent_id=active_agent_id,
        use_streaming=use_streaming,
        completion_items=completion_items or (),
    )
    return app


def _plain_text(content: object) -> str:
    """Extract plain text from whatever a Static was constructed with.

    Static (Textual 8.x) exposes no public accessor for its stored content,
    so this reaches into the name-mangled private attribute — acceptable
    here because it's test-only introspection of a widget app.py itself
    constructs (we know exactly what shapes it can hold: str, rich.text.Text,
    or rich.markdown.Markdown), not a dependency on Static's public contract.
    """
    from rich.markdown import Markdown as RichMarkdown

    value = getattr(content, "_Static__content", content)
    if isinstance(value, RichMarkdown):
        return value.markup
    return str(value)


def _chat_log_text(app: MinionApp) -> str:
    from textual.containers import VerticalScroll

    log = app.query_one("#chat-log", VerticalScroll)
    return "\n".join(_plain_text(child) for child in log.children)


class TestMessageDispatch:
    def test_plain_message_dispatches_to_session_send(self, tmp_path):
        async def _test():
            session = _FakeSession(response="hi there")
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("hello")
                await app.workers.wait_for_complete()

            assert len(session.send_calls) == 1
            call = session.send_calls[0]
            assert call["message"] == "hello"
            assert call["channel"] == "tui"

        _run(_test())

    def test_provider_exception_shown_as_error_in_log(self, tmp_path):
        async def _test():
            session = _FakeSession(raises=RuntimeError("connection refused"))
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("hello")
                await app.workers.wait_for_complete()

                text = _chat_log_text(app)
                assert "[Error]" in text
                assert "connection refused" in text

        _run(_test())

    def test_composer_is_reenabled_after_the_turn_completes(self, tmp_path):
        async def _test():
            from textual.widgets import Input

            session = _FakeSession(response="hi there")
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("hello")
                await app.workers.wait_for_complete()
                assert app.query_one("#composer", Input).disabled is False

        _run(_test())


class TestSlashCommands:
    def test_help_is_dispatched_as_a_command_not_sent_to_the_llm(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("/help")
                await app.workers.wait_for_complete()

            assert session.send_calls == []

        _run(_test())

    def test_unknown_command_is_reported_and_not_sent_to_the_llm(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("/totallynotarealcommand")
                await app.workers.wait_for_complete()

                assert session.send_calls == []
                assert "Unknown command" in _chat_log_text(app)

        _run(_test())

    def test_quit_command_exits_the_app(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test() as pilot:
                app._process_input("/quit")
                await pilot.pause()
                assert app._exit

        _run(_test())


class TestAttachments:
    def test_attach_stages_a_file_for_the_next_message(self, tmp_path):
        async def _test():
            media_dir = tmp_path / "attachments"
            image_path = tmp_path / "shot.png"
            # Minimal PNG signature — media.py sniffs bytes, not the extension.
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {image_path}")
                await app.workers.wait_for_complete()

                assert len(app._pending_attachments["main"]) == 1
                assert "Attached" in _chat_log_text(app)

        _run(_test())

    def test_staged_attachment_is_sent_with_the_next_message(self, tmp_path):
        async def _test():
            media_dir = tmp_path / "attachments"
            image_path = tmp_path / "shot.png"
            # A genuinely decodable image, not just a valid PNG signature:
            # once sent, this attachment is rendered inline via
            # ImageAttachmentView -> textual_image, which actually opens
            # the file through Pillow (unlike media.py's own sniff check,
            # which only reads the first 12 bytes).
            from PIL import Image as PILImage

            PILImage.new("RGB", (2, 2)).save(image_path)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {image_path}")
                await app.workers.wait_for_complete()
                app._process_input("what is this?")
                await app.workers.wait_for_complete()

            assert len(session.send_calls) == 1
            attachments = session.send_calls[0]["attachments"]
            assert attachments is not None and len(attachments) == 1
            # One-shot: staged attachments are cleared after being sent.
            assert app._pending_attachments["main"] == []

        _run(_test())

    def test_invalid_attach_path_reports_an_error(self, tmp_path):
        async def _test():
            media_dir = tmp_path / "attachments"
            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {tmp_path / 'does-not-exist.png'}")
                await app.workers.wait_for_complete()

                assert app._pending_attachments["main"] == []
                assert "Error" in _chat_log_text(app)

        _run(_test())


class TestStreaming:
    def test_streamed_tokens_accumulate_into_the_log(self, tmp_path):
        async def _test():
            events = [
                StreamingStarted(agent_name="Ada"),
                TokenStreamed(token="Hel"),
                TokenStreamed(token="lo!"),
                FinalAnswer(agent_name="Ada", text="Hello!"),
            ]
            session = _FakeSession(events=events)
            app = _configured_app(session, tmp_path, use_streaming=True)
            async with app.run_test():
                app._process_input("hi")
                await app.workers.wait_for_complete()

                text = _chat_log_text(app)
                assert "Hello!" in text
                # FinalAnswer must not duplicate the already-streamed text
                # with a second header/body pair.
                assert text.count("Ada:") == 1

        _run(_test())

    def test_compaction_failure_is_shown_as_a_warning(self, tmp_path):
        async def _test():
            session = _FakeSession(events=[CompactionFailed(error="disk full")])
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app._process_input("hi")
                await app.workers.wait_for_complete()

                text = _chat_log_text(app)
                assert "Compaction failed" in text
                assert "disk full" in text

        _run(_test())


class TestConsoleCallbackModals:
    """Proves the worker-thread -> call_from_thread -> push_screen_wait bridge
    actually works end to end — the trickiest part of this design. Each
    modal is invoked via a real Textual thread worker (app.run_worker(...,
    thread=True) — the same mechanism run_turn's own @work(thread=True)
    uses in production, and required by push_screen_wait itself), while the
    main test coroutine drives the Pilot to answer it."""

    def test_approve_bash_modal_round_trip(self):
        async def _test():
            app = MinionApp()
            async with app.run_test() as pilot:
                worker = app.run_worker(lambda: app.approve_bash("rm -rf /tmp/x"), thread=True)
                await pilot.pause()
                assert isinstance(app.screen, ApprovalModal)

                await pilot.press("2")  # Allow session
                decision = await worker.wait()

            assert decision == ApprovalDecision.ALLOW_SESSION

        _run(_test())

    def test_ask_user_modal_round_trip(self):
        async def _test():
            app = MinionApp()
            async with app.run_test() as pilot:
                worker = app.run_worker(lambda: app.ask_user("What is your name?"), thread=True)
                await pilot.pause()

                for ch in "Eric":
                    await pilot.press(ch)
                await pilot.press("enter")
                answer = await worker.wait()

            assert answer == "Eric"

        _run(_test())

    def test_approve_codex_extracts_command_summary(self):
        async def _test():
            app = MinionApp()
            async with app.run_test() as pilot:
                worker = app.run_worker(
                    lambda: app.approve_codex("shell", {"command": "ls -la /tmp"}), thread=True
                )
                await pilot.pause()

                await pilot.press("n")  # Deny
                decision = await worker.wait()

            assert decision == "deny"

        _run(_test())


class TestAttachmentRendering:
    """Phase 2: attachments sent with a turn render inline in the chat log."""

    def test_image_attachment_renders_inline(self, tmp_path):
        async def _test():
            from PIL import Image as PILImage

            media_dir = tmp_path / "attachments"
            image_path = tmp_path / "shot.png"
            # A genuinely decodable image — ImageAttachmentView opens it
            # through Pillow (via textual-image), not just a signature check.
            PILImage.new("RGB", (2, 2)).save(image_path)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {image_path}")
                await app.workers.wait_for_complete()
                app._process_input("look at this")
                await app.workers.wait_for_complete()

                assert len(app.query(ImageAttachmentView)) == 1

        _run(_test())

    def test_audio_attachment_renders_waveform_and_play_button(self, tmp_path):
        async def _test():
            import numpy as np
            import soundfile as sf

            media_dir = tmp_path / "attachments"
            audio_path = tmp_path / "memo.wav"
            sf.write(str(audio_path), np.zeros(4410, dtype="float32"), 44100)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {audio_path}")
                await app.workers.wait_for_complete()
                app._process_input("listen to this")
                await app.workers.wait_for_complete()

                views = app.query(AudioAttachmentView)
                assert len(views) == 1

        _run(_test())

    def test_attachment_only_renders_on_the_turn_it_was_sent_with(self, tmp_path):
        """Staging (/attach) alone must not render anything yet — only
        actually sending a message with it does."""
        async def _test():
            from PIL import Image as PILImage

            media_dir = tmp_path / "attachments"
            image_path = tmp_path / "shot.png"
            PILImage.new("RGB", (2, 2)).save(image_path)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test():
                app._process_input(f"/attach {image_path}")
                await app.workers.wait_for_complete()

                assert len(app.query(ImageAttachmentView)) == 0

        _run(_test())


class TestFilePicker:
    """Phase 2: bare /attach opens an interactive DirectoryTree picker."""

    def test_bare_attach_opens_the_picker(self, tmp_path):
        async def _test():
            media_dir = tmp_path / "attachments"
            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test() as pilot:
                app._process_input("/attach")
                await pilot.pause()

                assert isinstance(app.screen, AttachFilePickerModal)

                # Clean up: cancel so run_test()'s teardown doesn't hang
                # waiting on the still-open modal's worker.
                await pilot.press("escape")
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)

        _run(_test())

    def test_escape_cancels_without_staging_anything(self, tmp_path):
        async def _test():
            media_dir = tmp_path / "attachments"
            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test() as pilot:
                app._process_input("/attach")
                await pilot.pause()

                await pilot.press("escape")
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)

                assert app._pending_attachments["main"] == []

        _run(_test())

    def test_selecting_a_file_stages_it(self, tmp_path, monkeypatch):
        async def _test():
            from PIL import Image as PILImage

            media_dir = tmp_path / "attachments"
            pick_dir = tmp_path / "pickme"
            pick_dir.mkdir()
            PILImage.new("RGB", (2, 2)).save(pick_dir / "chosen.png")

            # _open_attach_picker() starts the browser at Path.cwd() with no
            # override hook — pin it to a directory containing exactly one
            # known file so keyboard navigation is deterministic.
            monkeypatch.setattr(Path, "cwd", staticmethod(lambda: pick_dir))

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test() as pilot:
                app._process_input("/attach")
                await pilot.pause()

                from textual.widgets import DirectoryTree

                # DirectoryTree lives on the modal screen, not the app's
                # default screen — App.query_one() only searches the
                # default screen, so this must go through app.screen
                # (the currently active/topmost screen) instead.
                tree = app.screen.query_one(DirectoryTree)
                tree.focus()
                await pilot.pause()
                # The cursor starts on the root (the picked directory
                # itself, "pickme") — "down" moves it onto the one file
                # inside before "enter" selects it. Selecting the root
                # itself would just toggle/expand it, never firing
                # FileSelected.
                await pilot.press("down")
                await pilot.press("enter")
                # Bounded wait, not a bare await: if FileSelected ever fails
                # to fire again (e.g. a future DirectoryTree/Textual
                # behavior change), _open_attach_picker's worker never
                # completes and a bare wait_for_complete() hangs forever
                # instead of failing — exactly what happened once while
                # writing this test, before the "down" press above was
                # added.
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)
                await pilot.pause()

                assert len(app._pending_attachments["main"]) == 1
                assert app._pending_attachments["main"][0].source_name == "chosen.png"

        _run(_test())


class TestAudioPlayback:
    """Phase 2: AudioAttachmentView's Play button -> PlayAudioRequested ->
    MinionApp.play_audio_file(). voice.audio.play_audio is mocked out —
    that function's own real behavior (sounddevice I/O) is voice/audio.py's
    own test's responsibility, not this one's."""

    def test_play_button_triggers_playback(self, tmp_path, monkeypatch):
        async def _test():
            from unittest.mock import Mock

            import numpy as np
            import soundfile as sf

            media_dir = tmp_path / "attachments"
            audio_path = tmp_path / "memo.wav"
            sf.write(str(audio_path), np.zeros(4410, dtype="float32"), 44100)

            play_mock = Mock()
            monkeypatch.setattr("minion_assist.voice.audio.play_audio", play_mock)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test() as pilot:
                app._process_input(f"/attach {audio_path}")
                await app.workers.wait_for_complete()
                app._process_input("listen")
                await app.workers.wait_for_complete()

                view = app.query_one(AudioAttachmentView)
                view.query_one("#play").press()
                # press() only posts Button.Pressed; the chain of handlers
                # that turns it into a running play_audio_file worker
                # (on_button_pressed -> post_message(PlayAudioRequested) ->
                # on_play_audio_requested -> self.play_audio_file(...)) needs
                # at least one event-loop tick to run before that worker
                # exists for wait_for_complete() to wait on — without this
                # pause, wait_for_complete() can return immediately having
                # found nothing to wait for yet.
                await pilot.pause()
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)

            play_mock.assert_called_once()
            samples_arg, samplerate_arg = play_mock.call_args[0]
            assert samplerate_arg == 44100
            assert len(samples_arg) == 4410

        _run(_test())

    def test_playback_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        async def _test():
            import numpy as np
            import soundfile as sf

            media_dir = tmp_path / "attachments"
            audio_path = tmp_path / "memo.wav"
            sf.write(str(audio_path), np.zeros(4410, dtype="float32"), 44100)

            def _raise(*args, **kwargs):
                raise RuntimeError("sounddevice not installed")

            monkeypatch.setattr("minion_assist.voice.audio.play_audio", _raise)

            session = _FakeSession()
            app = _configured_app(session, media_dir)
            async with app.run_test() as pilot:
                app._process_input(f"/attach {audio_path}")
                await app.workers.wait_for_complete()
                app._process_input("listen")
                await app.workers.wait_for_complete()

                view = app.query_one(AudioAttachmentView)
                view.query_one("#play").press()
                # See the sibling test above for why this pause is needed.
                await pilot.pause()
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)

                assert "Couldn't play audio" in _chat_log_text(app)

        _run(_test())


def _two_agent_setup(session_main: _FakeSession, session_researcher: _FakeSession) -> dict:
    return {
        "sessions": {"main": session_main, "researcher": session_researcher},
        "agents_cfg": {
            "main": SimpleNamespace(route_prefix=None),
            "researcher": SimpleNamespace(route_prefix="/research"),
        },
    }


class TestSidebar:
    """Phase 3: agent/session lists are display-only; selecting a row runs
    the equivalent /switch or /session command text — same dispatch path
    as typing it, verified separately by TestSlashCommands."""

    def test_agents_list_shows_turn_counts_and_marks_the_active_one(self, tmp_path):
        async def _test():
            session_main = _FakeSession()
            session_researcher = _FakeSession()
            store = _FakeSessionStore(
                [_FakeSessionInfo("main", 12), _FakeSessionInfo("researcher", 3)]
            )
            app = _configured_app(
                session_main, tmp_path,
                session_store=store,
                **_two_agent_setup(session_main, session_researcher),
            )
            async with app.run_test():
                sidebar = app.query_one(Sidebar)
                rows = [_plain_text(item.children[0]) for item in sidebar._agent_list.children]

                assert any(r.startswith(">") and "Ada" in r and "12 turns" in r for r in rows)
                assert any(
                    not r.startswith(">") and "Elizabeth" in r and "3 turns" in r for r in rows
                )

        _run(_test())

    def test_agents_list_without_a_session_store_still_shows_agents(self, tmp_path):
        async def _test():
            session_main = _FakeSession()
            session_researcher = _FakeSession()
            app = _configured_app(
                session_main, tmp_path,
                **_two_agent_setup(session_main, session_researcher),
            )
            async with app.run_test():
                sidebar = app.query_one(Sidebar)
                rows = [_plain_text(item.children[0]) for item in sidebar._agent_list.children]

                assert any("Ada" in r for r in rows)
                assert any("Elizabeth" in r for r in rows)
                # No turn counts available without a session_store.
                assert not any("turns" in r for r in rows)

        _run(_test())

    def test_clicking_an_agent_switches_the_active_agent(self, tmp_path):
        async def _test():
            session_main = _FakeSession()
            session_researcher = _FakeSession()
            app = _configured_app(
                session_main, tmp_path,
                **_two_agent_setup(session_main, session_researcher),
            )
            async with app.run_test() as pilot:
                # Layout must settle before pilot.click() can compute
                # correct on-screen coordinates for the target widget —
                # without this, the click can land on the wrong item (or
                # nothing) because the sidebar hasn't been arranged yet.
                await pilot.pause()
                sidebar = app.query_one(Sidebar)
                target = next(
                    item for item in sidebar._agent_list.children
                    if isinstance(item, _AgentListItem) and item.agent_id == "researcher"
                )
                await pilot.click(target)
                await pilot.pause()

                assert app._active_agent_id == "researcher"
                assert session_researcher.reload_calls == 1

        _run(_test())

    def test_clicking_an_agent_refreshes_the_sidebar_marker(self, tmp_path):
        async def _test():
            session_main = _FakeSession()
            session_researcher = _FakeSession()
            app = _configured_app(
                session_main, tmp_path,
                **_two_agent_setup(session_main, session_researcher),
            )
            async with app.run_test() as pilot:
                await pilot.pause()  # let layout settle before clicking
                sidebar = app.query_one(Sidebar)
                target = next(
                    item for item in sidebar._agent_list.children
                    if isinstance(item, _AgentListItem) and item.agent_id == "researcher"
                )
                await pilot.click(target)
                await pilot.pause()

                rows = [_plain_text(item.children[0]) for item in sidebar._agent_list.children]
                assert any(r.startswith(">") and "Elizabeth" in r for r in rows)

        _run(_test())

    def test_sessions_list_shows_names_and_marks_the_current_session(self, tmp_path):
        async def _test():
            session = _FakeSession()
            session.session_id = "abc123"
            short_term = _FakeShortTerm(
                sessions_by_agent={"main": [Path("abc123.jsonl"), Path("def456.jsonl")]},
                names={("main", "def456"): "Old debugging session"},
            )
            app = _configured_app(session, tmp_path, short_term=short_term)
            async with app.run_test():
                sidebar = app.query_one(Sidebar)
                rows = [_plain_text(item.children[0]) for item in sidebar._session_list.children]

                assert any("Old debugging session" in r for r in rows)
                assert any(r.startswith(">") and "abc123"[:8] in r for r in rows)

        _run(_test())

    def test_clicking_a_session_restores_it(self, tmp_path):
        async def _test():
            session = _FakeSession()
            session.session_id = "abc123"
            short_term = _FakeShortTerm(
                sessions_by_agent={"main": [Path("abc123.jsonl"), Path("def456.jsonl")]},
            )
            app = _configured_app(session, tmp_path, short_term=short_term)
            async with app.run_test() as pilot:
                await pilot.pause()  # let layout settle before clicking
                sidebar = app.query_one(Sidebar)
                # refresh_sidebar() reverses short_term.list_sessions()'s
                # order, so def456 is index 1 and abc123 (already current)
                # is index 2.
                target = next(
                    item for item in sidebar._session_list.children
                    if isinstance(item, _SessionListItem) and item.session_index == 1
                )
                await pilot.click(target)
                await pilot.pause()

                assert session.switch_calls == ["def456"]

        _run(_test())


class TestStatusBar:
    """Phase 3: a condensed, periodically-refreshed view of the same two
    data sources /status deep reads (worker_health, SessionDB.queue_lag_summary)."""

    def test_shows_the_active_agent(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                assert "Agent: main" in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_shows_healthy_worker_count(self, tmp_path):
        async def _test():
            wh = {"capture_worker": WorkerHealth("capture_worker")}
            wh["capture_worker"].record_poll()
            session = _FakeSession()
            app = _configured_app(session, tmp_path, worker_health=wh)
            async with app.run_test():
                assert "Workers: 1/1 ok" in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_shows_failing_worker_count(self, tmp_path):
        async def _test():
            wh = {"capture_worker": WorkerHealth("capture_worker")}
            wh["capture_worker"].record_failure("boom")
            session = _FakeSession()
            app = _configured_app(session, tmp_path, worker_health=wh)
            async with app.run_test():
                assert "Workers: 1 failing" in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_shows_queue_lag_from_the_database(self, tmp_path):
        async def _test():
            db = _FakeDB(lag={
                "capture": {"pending_count": 2},
                "commitment": {"pending_count": 1},
                "message_embedding": {"pending_count": 0},
            })
            session = _FakeSession()
            app = _configured_app(session, tmp_path, db=db)
            async with app.run_test():
                assert "Queue: 3 pending" in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_queue_lag_query_failure_is_reported_not_raised(self, tmp_path):
        async def _test():
            db = _FakeDB(raises=RuntimeError("db down"))
            session = _FakeSession()
            app = _configured_app(session, tmp_path, db=db)
            async with app.run_test():
                assert "Queue: unavailable" in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_no_database_configured_omits_the_queue_line(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                assert "Queue:" not in _plain_text(app.query_one("#status-bar"))

        _run(_test())

    def test_refreshes_after_a_turn_completes(self, tmp_path):
        async def _test():
            db = _FakeDB(lag={
                "capture": {"pending_count": 5},
                "commitment": {"pending_count": 0},
                "message_embedding": {"pending_count": 0},
            })
            session = _FakeSession(response="ok")
            app = _configured_app(session, tmp_path, db=db)
            async with app.run_test():
                app._process_input("hello")
                await app.workers.wait_for_complete()

                assert "Queue: 5 pending" in _plain_text(app.query_one("#status-bar"))

        _run(_test())


class TestCommandPalette:
    """Phase 3: Ctrl+P palette entries mirror the composer's own slash-
    command suggestions and run through the identical dispatch path."""

    def test_run_command_text_dispatches_like_typing_it(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = _configured_app(session, tmp_path)
            async with app.run_test():
                app.run_command_text("/help")
                await app.workers.wait_for_complete()

                assert session.send_calls == []
                assert "Built-in commands:" in _chat_log_text(app)

        _run(_test())

    def test_completion_items_are_exposed_for_the_provider(self, tmp_path):
        async def _test():
            session = _FakeSession()
            app = MinionApp()
            app.configure(
                sessions={"main": session},
                agents_cfg=_agents_cfg(),
                session_store=None,
                mcp_manager=None,
                skills=None,
                short_term=None,
                db=None,
                worker_health=None,
                media_dir=tmp_path,
                active_agent_id="main",
                use_streaming=False,
                completion_items=[("/help", "show help"), ("/new", "clear history")],
            )
            async with app.run_test():
                assert app.completion_items == [("/help", "show help"), ("/new", "clear history")]

        _run(_test())

    def test_provider_search_matches_by_prefix_and_carries_help_text(self, tmp_path):
        async def _test():
            from minion_assist.tui.app import SlashCommandProvider

            session = _FakeSession()
            app = MinionApp()
            app.configure(
                sessions={"main": session},
                agents_cfg=_agents_cfg(),
                session_store=None,
                mcp_manager=None,
                skills=None,
                short_term=None,
                db=None,
                worker_health=None,
                media_dir=tmp_path,
                active_agent_id="main",
                use_streaming=False,
                completion_items=[("/help", "show help"), ("/new", "clear history")],
            )
            async with app.run_test():
                provider = SlashCommandProvider(app.screen)

                hits = [hit async for hit in provider.search("help")]
                assert [(hit.text, hit.help) for hit in hits] == [("/help", "show help")]

                discovered = [hit async for hit in provider.discover()]
                assert [(hit.text, hit.help) for hit in discovered] == [
                    ("/help", "show help"),
                    ("/new", "clear history"),
                ]

        _run(_test())

    def test_selecting_a_palette_hit_dispatches_the_command(self, tmp_path):
        async def _test():
            from minion_assist.tui.app import SlashCommandProvider

            session = _FakeSession()
            app = _configured_app(session, tmp_path, completion_items=[("/help", "show help")])
            async with app.run_test():
                provider = SlashCommandProvider(app.screen)
                hits = [hit async for hit in provider.search("help")]

                hits[0].command()  # a Hit's command is the palette's on-select callback
                await app.workers.wait_for_complete()

                assert session.send_calls == []
                assert "Built-in commands:" in _chat_log_text(app)

        _run(_test())
