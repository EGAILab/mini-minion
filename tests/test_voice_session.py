"""Tests for voice/session.py — VoiceSession orchestrator.

All dependencies are mocked: STTAdapter, TTSAdapter, VadCapture, and
AgentSession.  Tests verify the orchestration logic, not the ML inference.

Coverage:
- run() calls stt.load() and tts.load() before starting
- run() starts vad_capture with the utterance queue
- _loop() transcribes utterance → calls agent.send() → calls tts.speak()
- Empty transcription is skipped (no agent call)
- Empty agent response is skipped (no TTS call)
- STT errors are caught and logged, loop continues
- Agent errors are caught and logged, loop continues
- TTS errors are caught and logged, loop continues
- stop() enqueues sentinel None to unblock the queue
- build_voice_session() creates a VoiceSession with the right components
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from minion_assist.voice.session import VoiceSession, build_voice_session
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

    Args:
        transcript: What the mock STT returns.
        agent_response: What the mock agent returns.
        utterances: Audio chunks to put in the queue; None = one default chunk.

    Returns:
        (session, mocks) where mocks has keys: stt, tts, vad, agent, queue.
    """
    mock_stt = MagicMock(spec=STTAdapter)
    mock_stt.transcribe.return_value = transcript

    mock_tts = MagicMock(spec=TTSAdapter)

    mock_vad = MagicMock(spec=VadCapture)

    mock_agent = MagicMock()
    mock_agent.send.return_value = agent_response

    utterance_q: queue.Queue = queue.Queue()
    if utterances is None:
        utterances = [np.zeros(512, dtype=np.float32)]
    for item in utterances:
        utterance_q.put(item)
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

    return session, {
        "stt": mock_stt,
        "tts": mock_tts,
        "vad": mock_vad,
        "agent": mock_agent,
        "queue": utterance_q,
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
    """_loop() must forward the transcript text to agent.send()."""
    session, mocks = _make_session(transcript="what time is it")

    session._running = True
    session._loop()

    mocks["agent"].send.assert_called_once()
    text_arg = mocks["agent"].send.call_args.args[0]
    assert text_arg == "what time is it"


def test_loop_calls_tts_speak_with_response():
    """_loop() must pass the agent response to tts.speak()."""
    session, mocks = _make_session(agent_response="it is noon")

    session._running = True
    session._loop()

    mocks["tts"].speak.assert_called_once()
    text_arg = mocks["tts"].speak.call_args.args[0]
    assert text_arg == "it is noon"


def test_loop_passes_output_device_to_speak():
    """tts.speak() must receive the configured output_device."""
    session, mocks = _make_session()
    session._output_device = "my-speakers"

    session._running = True
    session._loop()

    kwargs = mocks["tts"].speak.call_args.kwargs
    assert kwargs.get("device") == "my-speakers"


def test_loop_processes_multiple_utterances():
    """_loop() must handle multiple consecutive utterances."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio, audio])

    session._running = True
    session._loop()

    assert mocks["stt"].transcribe.call_count == 3
    assert mocks["agent"].send.call_count == 3
    assert mocks["tts"].speak.call_count == 3


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
    """_loop() must not call tts.speak() when agent returns empty string."""
    session, mocks = _make_session(agent_response="")

    session._running = True
    session._loop()

    mocks["tts"].speak.assert_not_called()


def test_loop_skips_whitespace_agent_response():
    session, mocks = _make_session(agent_response="   ")

    session._running = True
    session._loop()

    mocks["tts"].speak.assert_not_called()


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

    # TTS called once for the second utterance
    assert mocks["tts"].speak.call_count == 1


def test_loop_continues_after_tts_error(capsys):
    """A TTS error must not crash the loop; next utterance is processed."""
    audio = np.zeros(512, dtype=np.float32)
    session, mocks = _make_session(utterances=[audio, audio])

    mocks["tts"].speak.side_effect = [RuntimeError("TTS failed"), None]

    session._running = True
    session._loop()

    assert mocks["tts"].speak.call_count == 2


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
# build_voice_session() factory
# ---------------------------------------------------------------------------

class _FakeVoiceConfig:
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
        qwen3_model_id = "Qwen/Qwen3-TTS-1.7B"
        qwen3_precision = "fp16"
        kokoro_voice = "af_heart"
        device = "cpu"
        voice_ref_audio = None
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
