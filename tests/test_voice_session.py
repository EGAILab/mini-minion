"""Tests for voice/session.py — VoiceSession orchestrator.

All dependencies are mocked: STTAdapter, TTSAdapter, VadCapture, and
AgentSession.  Tests verify the orchestration logic, not the ML inference.

Coverage:
- run() calls stt.load() and tts.load() before starting
- run() starts vad_capture with the utterance queue
- _loop() transcribes utterance → calls agent.send() → calls _speak_streaming()
- Empty transcription is skipped (no agent call)
- Empty agent response is skipped (no TTS call)
- STT errors are caught and logged, loop continues
- Agent errors are caught and logged, loop continues
- TTS errors are caught and logged, loop continues
- stop() enqueues sentinel None to unblock the queue
- build_voice_session() creates a VoiceSession with the right components
- _split_sentences() splits on . ! ? boundaries
- _speak_streaming() pipelines sentence synthesis and playback
- Barge-in during playback stops TTS and returns early
"""
from __future__ import annotations

import queue
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from minion_assist.voice.session import (
    VoiceSession,
    _split_sentences,
    _strip_markdown,
    build_voice_session,
)
from minion_assist.voice.stt import STTAdapter
from minion_assist.voice.tts import TTSAdapter
from minion_assist.voice.vad import VadCapture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(
    transcript: str = "hello",
    agent_response: str = "hi there",
    utterances: "list[np.ndarray | None] | None" = None,
) -> tuple[VoiceSession, dict]:
    """Build a VoiceSession with fully mocked components.

    ``_speak_streaming`` is replaced with a ``MagicMock`` so that ``_loop()``
    tests do not need sounddevice or real TTS inference.  Tests that exercise
    the real ``_speak_streaming`` implementation should delete the instance
    attribute to reveal the class method.

    Args:
        transcript: What the mock STT returns.
        agent_response: What the mock agent returns.
        utterances: Audio chunks to put in the queue; None = one default chunk.

    Returns:
        (session, mocks) where mocks has keys:
        stt, tts, vad, agent, queue, speak_streaming.
    """
    mock_stt = MagicMock(spec=STTAdapter)
    mock_stt.transcribe.return_value = transcript

    mock_tts = MagicMock(spec=TTSAdapter)
    # Default return for synthesise() so _speak_streaming tests have valid audio.
    mock_tts.synthesise.return_value = (np.zeros(512, dtype=np.float32), 24_000)
    # Disable streaming by default so sentence-split tests use synthesise().
    mock_tts.supports_streaming = False

    mock_vad = MagicMock(spec=VadCapture)
    # speech_started.is_set() must return False by default (no barge-in).
    mock_vad.speech_started = MagicMock()
    mock_vad.speech_started.is_set.return_value = False
    # tts_playing is set/cleared by _speak_streaming to tag barge-in utterances.
    mock_vad.tts_playing = MagicMock()

    mock_agent = MagicMock()
    mock_agent.send.return_value = agent_response

    utterance_q: queue.Queue = queue.Queue()
    if utterances is None:
        utterances = [np.zeros(512, dtype=np.float32)]
    for item in utterances:
        # Queue holds (audio, during_tts) tuples or None sentinel.
        # Wrap plain arrays as normal (non-TTS) utterances.
        utterance_q.put((item, False) if item is not None else None)
    # Always put a sentinel so the loop exits after processing all utterances.
    utterance_q.put(None)

    session = VoiceSession(
        agent_session=mock_agent,
        stt=mock_stt,
        tts=mock_tts,
        vad_capture=mock_vad,
    )
    # Inject the pre-loaded queue so _loop() doesn't create a new one.
    session._utterance_queue = utterance_q
    # Mock _speak_streaming so _loop() tests don't need sounddevice.
    session._speak_streaming = MagicMock()

    return session, {
        "stt": mock_stt,
        "tts": mock_tts,
        "vad": mock_vad,
        "agent": mock_agent,
        "queue": utterance_q,
        "speak_streaming": session._speak_streaming,
    }


# ---------------------------------------------------------------------------
# run() — startup sequence
# ---------------------------------------------------------------------------

def test_run_loads_stt_before_capture():
    """run() must call stt.load() before starting vad_capture."""
    session, mocks = _make_session()
    call_order = []
    mocks["stt"].load.side_effect = lambda: call_order.append("stt.load")
    mocks["vad"].start.side_effect = lambda q: call_order.append("vad.start")

    session.run()

    assert call_order.index("stt.load") < call_order.index("vad.start")


def test_run_loads_tts_before_capture():
    """run() must call tts.load() before starting vad_capture."""
    session, mocks = _make_session()
    call_order = []
    mocks["tts"].load.side_effect = lambda: call_order.append("tts.load")
    mocks["vad"].start.side_effect = lambda q: call_order.append("vad.start")

    session.run()

    assert call_order.index("tts.load") < call_order.index("vad.start")


def test_run_starts_vad_capture():
    """run() must call vad_capture.start() with the utterance queue."""
    session, mocks = _make_session()
    session.run()
    mocks["vad"].start.assert_called_once()
    q_arg = mocks["vad"].start.call_args.args[0]
    assert isinstance(q_arg, queue.Queue)


def test_run_stops_vad_capture_on_exit():
    """run() must call vad_capture.stop() even if the loop exits normally."""
    session, mocks = _make_session()
    session.run()
    mocks["vad"].stop.assert_called_once()


def test_run_stops_vad_capture_on_exception():
    """vad_capture.stop() must be called even when the loop raises."""
    session, mocks = _make_session()
    mocks["stt"].transcribe.side_effect = RuntimeError("forced")
    session.run()  # errors are caught inside _loop
    mocks["vad"].stop.assert_called_once()


# ---------------------------------------------------------------------------
# _loop() — normal flow
# ---------------------------------------------------------------------------

def test_loop_calls_stt_transcribe_with_audio():
    """_loop() must pass the audio array from the queue to stt.transcribe()."""
    audio = np.ones(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio])

    session._running = True
    session._loop()

    mocks["stt"].transcribe.assert_called_once()
    arg = mocks["stt"].transcribe.call_args.args[0]
    np.testing.assert_array_equal(arg, audio)


def test_loop_calls_agent_send_with_transcript():
    """_loop() must include the transcript text in the message sent to agent.send()."""
    session, mocks = _make_session(transcript="what time is it")

    session._running = True
    session._loop()

    mocks["agent"].send.assert_called_once()
    text_arg = mocks["agent"].send.call_args.args[0]
    assert "what time is it" in text_arg


def test_loop_calls_speak_streaming_with_response():
    """_loop() must pass the full agent response to _speak_streaming()."""
    session, mocks = _make_session(agent_response="it is noon")

    session._running = True
    session._loop()

    mocks["speak_streaming"].assert_called_once_with("it is noon")


def test_loop_processes_multiple_utterances():
    """_loop() must handle multiple consecutive utterances."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio, audio])

    session._running = True
    session._loop()

    assert mocks["stt"].transcribe.call_count == 3
    assert mocks["agent"].send.call_count == 3
    assert mocks["speak_streaming"].call_count == 3


# ---------------------------------------------------------------------------
# _loop() — skipping empty results
# ---------------------------------------------------------------------------

def test_loop_skips_empty_transcript():
    """_loop() must not call agent.send() when STT returns an empty string."""
    session, mocks = _make_session(transcript="   ")

    session._running = True
    session._loop()

    mocks["agent"].send.assert_not_called()


def test_loop_skips_empty_agent_response():
    """_loop() must not call _speak_streaming() when agent returns empty string."""
    session, mocks = _make_session(agent_response="")

    session._running = True
    session._loop()

    mocks["speak_streaming"].assert_not_called()


def test_loop_skips_whitespace_agent_response():
    session, mocks = _make_session(agent_response="   ")

    session._running = True
    session._loop()

    mocks["speak_streaming"].assert_not_called()


# ---------------------------------------------------------------------------
# _loop() — error handling
# ---------------------------------------------------------------------------

def test_loop_continues_after_stt_error(capsys):
    """An STT error must not crash the loop; a second utterance still processes."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio])

    # First call raises, second returns text
    mocks["stt"].transcribe.side_effect = [RuntimeError("STT failed"), "ok"]

    session._running = True
    session._loop()

    # Second utterance was still processed
    assert mocks["agent"].send.call_count == 1


def test_loop_continues_after_agent_error(capsys):
    """An agent error must not crash the loop; a subsequent utterance is processed."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio])

    mocks["agent"].send.side_effect = [RuntimeError("agent down"), "ok"]

    session._running = True
    session._loop()

    # _speak_streaming called once for the second utterance
    assert mocks["speak_streaming"].call_count == 1


def test_loop_continues_after_tts_error(capsys):
    """A TTS error from _speak_streaming must not crash the loop."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio])

    mocks["speak_streaming"].side_effect = [RuntimeError("TTS failed"), None]

    session._running = True
    session._loop()

    assert mocks["speak_streaming"].call_count == 2


def test_loop_reraises_keyboard_interrupt():
    """KeyboardInterrupt from agent.send() must propagate out of _loop()."""
    session, mocks = _make_session()
    mocks["agent"].send.side_effect = KeyboardInterrupt()

    session._running = True
    with pytest.raises(KeyboardInterrupt):
        session._loop()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

def test_stop_sets_running_false():
    session, _ = _make_session()
    session._running = True
    session.stop()
    assert session._running is False


def test_stop_enqueues_sentinel():
    """stop() must put None into the queue to unblock queue.get()."""
    session, _ = _make_session()
    # Drain the existing contents first
    while not session._utterance_queue.empty():
        session._utterance_queue.get_nowait()

    session.stop()
    sentinel = session._utterance_queue.get_nowait()
    assert sentinel is None


# ---------------------------------------------------------------------------
# on_event forwarding
# ---------------------------------------------------------------------------

def test_loop_passes_on_event_to_agent():
    """The on_event callback must be forwarded to agent.send()."""
    my_handler = MagicMock()
    session, mocks = _make_session()
    session._on_event = my_handler

    session._running = True
    session._loop()

    kwargs = mocks["agent"].send.call_args.kwargs
    assert kwargs.get("on_event") is my_handler


# ---------------------------------------------------------------------------
# Language injection
# ---------------------------------------------------------------------------

def test_loop_prepends_language_note_when_set():
    """When _language is set, send() receives the constraint prefix before the transcript."""
    session, mocks = _make_session(transcript="what time is it")
    session._language = "en"

    session._running = True
    session._loop()

    sent_text = mocks["agent"].send.call_args.args[0]
    assert sent_text.startswith("[Voice mode: reply in en only.]")
    assert "what time is it" in sent_text


def test_loop_no_language_note_when_language_empty():
    """When _language is empty, send() receives the raw transcript unchanged."""
    session, mocks = _make_session(transcript="hello")
    session._language = ""

    session._running = True
    session._loop()

    sent_text = mocks["agent"].send.call_args.args[0]
    assert sent_text == "hello"


def test_build_voice_session_passes_language():
    """build_voice_session() must forward voice_config.language to VoiceSession._language."""
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._language == "en"


# ---------------------------------------------------------------------------
# _split_sentences()
# ---------------------------------------------------------------------------

def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_single_no_punctuation():
    """A sentence with no terminating punctuation returns as a single element."""
    assert _split_sentences("hello world") == ["hello world"]


def test_split_sentences_two_sentences():
    result = _split_sentences("Hello world. How are you?")
    assert result == ["Hello world.", "How are you?"]


def test_split_sentences_three_sentences():
    result = _split_sentences("One. Two! Three?")
    assert result == ["One.", "Two!", "Three?"]


def test_split_sentences_preserves_content():
    """Each sentence must contain its original words."""
    result = _split_sentences("The cat sat. The dog ran.")
    assert "cat sat" in result[0]
    assert "dog ran" in result[1]


# ---------------------------------------------------------------------------
# _speak_streaming() — mocked sounddevice
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_sounddevice(monkeypatch):
    """Inject a mock sounddevice whose OutputStream is a no-op context manager.

    ``stream.write()`` and ``stream.abort()`` are no-ops; ``stream.latency``
    is 0 so the drain wait exits immediately in tests.
    """
    mock_sd = MagicMock()
    mock_stream = MagicMock()
    mock_stream.latency = 0
    # Make the context manager yield the same mock_stream object.
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_sd.OutputStream.return_value = mock_stream
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    return mock_sd


def test_speak_streaming_synthesises_each_sentence(mock_sounddevice):
    """_speak_streaming() must call tts.synthesise() once per sentence."""
    session, mocks = _make_session()
    del session._speak_streaming  # use real implementation from class

    session._running = True
    session._speak_streaming("Hello there. How are you?")

    assert mocks["tts"].synthesise.call_count == 2


def test_speak_streaming_single_sentence(mock_sounddevice):
    """Single sentence (no punctuation boundary) synthesises exactly once."""
    session, mocks = _make_session()
    del session._speak_streaming

    session._running = True
    session._speak_streaming("Just one sentence")

    mocks["tts"].synthesise.assert_called_once()
    args = mocks["tts"].synthesise.call_args.args
    assert "Just one sentence" in args[0]


def test_speak_streaming_empty_text_is_noop(mock_sounddevice):
    """_speak_streaming() with empty text must not call synthesise."""
    session, mocks = _make_session()
    del session._speak_streaming

    session._running = True
    session._speak_streaming("   ")

    mocks["tts"].synthesise.assert_not_called()
    mock_sounddevice.OutputStream.assert_not_called()


def test_speak_streaming_plays_audio(mock_sounddevice):
    """_speak_streaming() must write float32 audio to the output stream."""
    session, mocks = _make_session()
    del session._speak_streaming

    session._running = True
    session._speak_streaming("Say something.")

    mock_stream = mock_sounddevice.OutputStream.return_value
    assert mock_stream.write.called
    audio_chunk = mock_stream.write.call_args_list[0].args[0]
    assert audio_chunk.dtype == np.float32


def test_speak_streaming_returns_early_on_barge_in(mock_sounddevice):
    """Barge-in via speech_started event stops playback and returns early.

    When VadCapture fires speech_started during TTS, _speak_streaming must
    abort the stream and return so _loop() can pick up the new utterance.
    """
    session, mocks = _make_session()
    del session._speak_streaming

    utterance_q = session._utterance_queue
    while not utterance_q.empty():
        utterance_q.get_nowait()

    # Pre-load a barge-in utterance and arm the speech_started flag.
    utterance_q.put(np.zeros(512, dtype=np.float32))
    mocks["vad"].speech_started.is_set.return_value = True

    session._running = True
    session._speak_streaming("First sentence. Second sentence.")

    # Must have returned early; utterance still in queue for _loop().
    assert not utterance_q.empty()


def test_speak_streaming_falls_back_without_sounddevice():
    """When sounddevice is missing, _speak_streaming falls back to tts.speak()."""
    session, mocks = _make_session()
    del session._speak_streaming

    session._running = True
    with patch.dict(sys.modules, {"sounddevice": None}):
        session._speak_streaming("Hello world.")

    mocks["tts"].speak.assert_called_once()
    assert mocks["tts"].speak.call_args.args[0] == "Hello world."


# ---------------------------------------------------------------------------
# build_voice_session() factory
# ---------------------------------------------------------------------------

class _FakeVoiceConfig:
    language = "en"
    max_history_turns = 4
    skip_bootstrap = True
    class vad:
        threshold = 0.5
        silence_ms = 700
    class stt:
        model = "parakeet"
        parakeet_model_id = "nvidia/parakeet-tdt-0.6b-v3"
        whisper_model_id = "distil-whisper/distil-large-v3"
        chunk_duration_s = 20
        device = "cpu"
    class tts:
        model = "qwen3"
        qwen3_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        qwen3_precision = "bf16"
        qwen3_speaker = "Vivian"
        kokoro_voice = "af_heart"
        device = "cpu"
        voice_ref_audio = None
        voice_ref_text = None
        piper_model_path = ""
    class audio:
        input_device = None
        output_device = "test-speakers"
        sample_rate = 16_000


def test_build_voice_session_returns_voice_session():
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert isinstance(result, VoiceSession)


def test_build_voice_session_sets_output_device():
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._output_device == "test-speakers"


def test_build_voice_session_wires_agent_session():
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._agent_session is mock_agent


def test_build_voice_session_with_on_event():
    mock_agent = MagicMock()
    handler = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig, on_event=handler)
    assert result._on_event is handler


# ---------------------------------------------------------------------------
# _speak_streaming() — streaming TTS path (supports_streaming=True)
# ---------------------------------------------------------------------------

def test_speak_streaming_uses_synthesise_stream_when_supported(mock_sounddevice):
    """When tts.supports_streaming is True, synthesise_stream() is called instead of synthesise()."""
    session, mocks = _make_session()
    del session._speak_streaming

    audio = np.ones(512, dtype=np.float32) * 0.5
    mocks["tts"].supports_streaming = True
    mocks["tts"].synthesise_stream.return_value = iter([(audio, 24_000)])

    session._running = True
    session._speak_streaming("Hello world.")

    mocks["tts"].synthesise_stream.assert_called_once_with("Hello world.")
    mocks["tts"].synthesise.assert_not_called()


def test_speak_streaming_stream_passes_full_text(mock_sounddevice):
    """synthesise_stream() receives the full response text, not individual sentences."""
    session, mocks = _make_session()
    del session._speak_streaming

    audio = np.ones(512, dtype=np.float32) * 0.3
    mocks["tts"].supports_streaming = True
    mocks["tts"].synthesise_stream.return_value = iter([(audio, 24_000), (audio, 24_000)])

    session._running = True
    session._speak_streaming("First sentence. Second sentence.")

    call_arg = mocks["tts"].synthesise_stream.call_args.args[0]
    assert call_arg == "First sentence. Second sentence."


def test_speak_streaming_stream_plays_all_chunks(mock_sounddevice):
    """All chunks from synthesise_stream() are written to the output stream."""
    session, mocks = _make_session()
    del session._speak_streaming

    audio = np.ones(512, dtype=np.float32) * 0.4
    mocks["tts"].supports_streaming = True
    mocks["tts"].synthesise_stream.return_value = iter([(audio, 24_000), (audio, 24_000)])

    session._running = True
    session._speak_streaming("Hello. World.")

    mock_stream = mock_sounddevice.OutputStream.return_value
    assert mock_stream.write.called


def test_speak_streaming_strips_markdown_before_synthesis(mock_sounddevice):
    """_speak_streaming() must strip markdown before passing text to the synth."""
    session, mocks = _make_session()
    del session._speak_streaming

    session._running = True
    session._speak_streaming("**Hello** world.")

    args = mocks["tts"].synthesise.call_args.args
    assert "**" not in args[0]
    assert "Hello" in args[0]


# ---------------------------------------------------------------------------
# _strip_markdown()
# ---------------------------------------------------------------------------

def test_strip_markdown_bold_stars():
    assert _strip_markdown("**bold**") == "bold"


def test_strip_markdown_italic_stars():
    assert _strip_markdown("*italic*") == "italic"


def test_strip_markdown_bold_underscores():
    assert _strip_markdown("__bold__") == "bold"


def test_strip_markdown_italic_underscores():
    assert _strip_markdown("_italic_") == "italic"


def test_strip_markdown_bold_italic():
    assert _strip_markdown("***bold italic***") == "bold italic"


def test_strip_markdown_heading():
    assert _strip_markdown("# Title") == "Title"


def test_strip_markdown_heading_level3():
    assert _strip_markdown("### Sub-heading") == "Sub-heading"


def test_strip_markdown_inline_code():
    assert _strip_markdown("`code`") == "code"


def test_strip_markdown_link():
    assert _strip_markdown("[click here](https://example.com)") == "click here"


def test_strip_markdown_bullet_dash():
    assert _strip_markdown("- item one") == "item one"


def test_strip_markdown_bullet_star():
    assert _strip_markdown("* item two") == "item two"


def test_strip_markdown_horizontal_rule():
    result = _strip_markdown("before\n---\nafter")
    assert "---" not in result
    assert "before" in result
    assert "after" in result


def test_strip_markdown_numbered_list_unchanged():
    """Numbered lists like '1. item' must not be altered."""
    assert _strip_markdown("1. First item") == "1. First item"


def test_strip_markdown_em_dash_unchanged():
    """Em dashes are valid speech; they must not be removed."""
    assert "—" in _strip_markdown("Children of Time — Adrian Tchaikovsky")


def test_strip_markdown_mixed_real_example():
    """Reproduce the exact format from the reported issue."""
    text = "1. **Children of Time** — Adrian Tchaikovsky\n2. **A Fire Upon the Deep** — Vernor Vinge"
    result = _strip_markdown(text)
    assert "**" not in result
    assert "Children of Time" in result
    assert "Adrian Tchaikovsky" in result
    assert "—" in result


def test_strip_markdown_plain_text_unchanged():
    """Plain prose with no markdown must pass through unmodified."""
    plain = "The weather today is nice."
    assert _strip_markdown(plain) == plain


# ---------------------------------------------------------------------------
# Voice responsiveness — max_history_turns and skip_bootstrap forwarding
# ---------------------------------------------------------------------------

def test_loop_passes_max_history_turns_to_agent():
    """_loop() must forward max_history_turns to agent.send()."""
    session, mocks = _make_session()
    session._max_history_turns = 3

    session._running = True
    session._loop()

    kwargs = mocks["agent"].send.call_args.kwargs
    assert kwargs.get("max_history_turns") == 3


def test_loop_passes_skip_bootstrap_to_agent():
    """_loop() must forward skip_bootstrap to agent.send()."""
    session, mocks = _make_session()
    session._skip_bootstrap = True

    session._running = True
    session._loop()

    kwargs = mocks["agent"].send.call_args.kwargs
    assert kwargs.get("skip_bootstrap") is True


def test_loop_skip_bootstrap_false_forwarded():
    """skip_bootstrap=False must be forwarded faithfully (regression guard)."""
    session, mocks = _make_session()
    session._skip_bootstrap = False

    session._running = True
    session._loop()

    kwargs = mocks["agent"].send.call_args.kwargs
    assert kwargs.get("skip_bootstrap") is False


def test_loop_max_history_turns_none_forwarded():
    """max_history_turns=None (full history) must be forwarded faithfully."""
    session, mocks = _make_session()
    session._max_history_turns = None

    session._running = True
    session._loop()

    kwargs = mocks["agent"].send.call_args.kwargs
    assert kwargs.get("max_history_turns") is None


def test_loop_discards_barge_in_utterance(capsys):
    """_loop() must discard utterances where during_tts=True and not call STT."""
    audio = np.zeros(512, dtype=np.float32)
    # Put a contaminated utterance (during_tts=True) then a sentinel.
    utterance_q: queue.Queue = queue.Queue()
    utterance_q.put((audio, True))
    utterance_q.put(None)

    session, mocks = _make_session()
    session._utterance_queue = utterance_q

    session._running = True
    session._loop()

    mocks["stt"].transcribe.assert_not_called()
    mocks["agent"].send.assert_not_called()
    out = capsys.readouterr().out
    assert "discarded" in out


def test_build_voice_session_passes_max_history_turns():
    """build_voice_session() must forward voice_config.max_history_turns to VoiceSession."""
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._max_history_turns == 4


def test_build_voice_session_passes_skip_bootstrap():
    """build_voice_session() must forward voice_config.skip_bootstrap to VoiceSession."""
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._skip_bootstrap is True


def test_loop_transcript_continues_on_post_tts_prompt(capsys):
    """When post-TTS prompt is on screen, transcript continues on the same line (no extra label)."""
    session, mocks = _make_session(transcript="hello", agent_response="hello back")
    # Simulate state where post-TTS prompt is already on screen.
    session._prompt_shown = True
    session._user_label = "you"

    session._running = True
    session._loop()

    out = capsys.readouterr().out
    # "hello" appears (the transcript), but NOT prefixed with a new "[you]" label
    # (the label is already on screen from the post-TTS prompt).
    assert "hello" in out
    # The first occurrence of "[you]" in the output is NOT "[you] hello"
    # — the transcript text is printed alone, continuing the existing prompt line.
    first_you = out.index("[you]")
    segment = out[first_you: first_you + 20]
    assert not segment.startswith("[you] hello")


def test_loop_prints_post_tts_prompt_after_tts(capsys):
    """After TTS, _loop() prints '[label] ' on a new line and sets _prompt_shown."""
    session, mocks = _make_session(transcript="hello", agent_response="hello back")

    session._running = True
    session._loop()

    out = capsys.readouterr().out
    # Post-TTS prompt appears (possibly followed by more output).
    assert "[you] " in out
    # _prompt_shown is True after TTS (next transcript will continue on same line).
    assert session._prompt_shown is True


# ---------------------------------------------------------------------------
# user_label — personalised [name] prompt
# ---------------------------------------------------------------------------

def test_loop_uses_user_label_in_transcript_print(capsys):
    """_loop() prints [<user_label>] instead of [you] when user_label is set."""
    session, _ = _make_session(transcript="hello")
    session._user_label = "Alice"

    session._running = True
    session._loop()

    out = capsys.readouterr().out
    assert "[Alice] hello" in out
    assert "[you]" not in out


def test_loop_uses_user_label_with_transcript(capsys):
    """[label] with transcript text appears in the output."""
    session, _ = _make_session(transcript="hello", agent_response="hi there")
    session._user_label = "Bob"

    session._running = True
    session._loop()

    out = capsys.readouterr().out
    # Either as a fresh label+text (no prior TTS) or post-TTS prompt+text on same line.
    assert "hello" in out
    assert "[Bob]" in out


def test_loop_defaults_to_you_label(capsys):
    """_loop() falls back to '[you]' when user_label is the default."""
    session, _ = _make_session()
    assert session._user_label == "you"

    session._running = True
    session._loop()

    out = capsys.readouterr().out
    assert "[you]" in out


def test_build_voice_session_reads_name_from_user_md(tmp_path):
    """build_voice_session() sets user_label from USER.md Name: field."""
    (tmp_path / "USER.md").write_text("Name: Carol\n", encoding="utf-8")

    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig, bootstrap_root=tmp_path)
    assert result._user_label == "Carol"


def test_build_voice_session_defaults_to_you_without_user_md(tmp_path):
    """build_voice_session() defaults user_label to 'you' when USER.md absent."""
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig, bootstrap_root=tmp_path)
    assert result._user_label == "you"


def test_build_voice_session_defaults_to_you_without_bootstrap_root():
    """build_voice_session() defaults user_label to 'you' when bootstrap_root is None."""
    mock_agent = MagicMock()
    result = build_voice_session(mock_agent, _FakeVoiceConfig)
    assert result._user_label == "you"
