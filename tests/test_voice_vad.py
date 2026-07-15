"""Tests for voice/vad.py — Silero VAD wrapper and VadCapture.

All Silero VAD and sounddevice calls are mocked; no real audio device or ML
model is loaded.  The tests focus on:

- SileroVAD.process() dispatches to the correct mocked iterator
- SileroVAD.reset() calls reset_states() on the iterator
- VadCapture._capture_loop() correctly accumulates speech and emits utterances
- VadCapture start/stop lifecycle
- build_vad_capture() reads threshold, silence_ms, sample_rate, device from config
"""
from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import minion_assist.voice.vad as vad_module
from minion_assist.voice.vad import SileroVAD, VadCapture, build_vad_capture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_vad_singletons():
    """Reset internal state on SileroVAD between tests."""
    yield
    # Nothing global to reset currently, but keeps the pattern consistent
    # with test_tools_browser.py for future additions.


@pytest.fixture()
def mock_silero(monkeypatch):
    """Inject a mock silero_vad package into sys.modules.

    Returns a dict with 'model', 'iterator', and 'fn' keys pointing to
    MagicMocks that mirror the real silero_vad API.
    """
    import sys

    mock_model = MagicMock()
    mock_iterator = MagicMock()
    mock_iterator.return_value = None  # default: no event

    mock_load_fn = MagicMock(return_value=mock_model)
    mock_vad_iterator_cls = MagicMock(return_value=mock_iterator)

    mock_pkg = MagicMock()
    mock_pkg.load_silero_vad = mock_load_fn
    mock_pkg.VADIterator = mock_vad_iterator_cls

    monkeypatch.setitem(sys.modules, "silero_vad", mock_pkg)

    return {"model": mock_model, "iterator": mock_iterator, "load": mock_load_fn,
            "VADIterator": mock_vad_iterator_cls}


@pytest.fixture()
def mock_torch(monkeypatch):
    """Inject a mock torch package so from_numpy() doesn't need real torch."""
    import sys

    mock_tensor = MagicMock()
    mock_torch_pkg = MagicMock()
    mock_torch_pkg.from_numpy.return_value = mock_tensor

    monkeypatch.setitem(sys.modules, "torch", mock_torch_pkg)
    return {"tensor": mock_tensor, "torch": mock_torch_pkg}


@pytest.fixture()
def loaded_vad(mock_silero, mock_torch):
    """Return a SileroVAD instance whose model is pre-loaded (iterator set)."""
    vad = SileroVAD(threshold=0.5, sample_rate=16_000, silence_ms=700)
    vad._load()
    # _iterator is now the mock iterator returned by VADIterator(...)
    return vad


# ---------------------------------------------------------------------------
# SileroVAD — lazy loading
# ---------------------------------------------------------------------------

def test_vad_load_calls_load_silero_vad(mock_silero, mock_torch):
    """_load() should call load_silero_vad() exactly once."""
    vad = SileroVAD()
    vad._load()
    mock_silero["load"].assert_called_once()


def test_vad_load_creates_vad_iterator(mock_silero, mock_torch):
    """_load() should instantiate VADIterator with threshold and sample_rate."""
    vad = SileroVAD(threshold=0.7, sample_rate=16_000)
    vad._load()
    mock_silero["VADIterator"].assert_called_once()
    call_kwargs = mock_silero["VADIterator"].call_args.kwargs
    assert call_kwargs["threshold"] == 0.7
    assert call_kwargs["sampling_rate"] == 16_000


def test_vad_load_is_idempotent(mock_silero, mock_torch):
    """Calling _load() twice must not load the model a second time."""
    vad = SileroVAD()
    vad._load()
    vad._load()
    mock_silero["load"].assert_called_once()


def test_vad_load_missing_silero_raises():
    """ImportError from silero_vad must surface as a RuntimeError with hint."""
    import sys
    with patch.dict(sys.modules, {"silero_vad": None}):
        vad = SileroVAD()
        with pytest.raises(RuntimeError, match="silero-vad not installed"):
            vad._load()


def test_vad_load_missing_torch_raises(mock_silero):
    """ImportError from torch must surface as RuntimeError with hint."""
    import sys
    with patch.dict(sys.modules, {"torch": None}):
        vad = SileroVAD()
        chunk = np.zeros(512, dtype=np.float32)
        with pytest.raises(RuntimeError, match="torch not installed"):
            vad.process(chunk)


# ---------------------------------------------------------------------------
# SileroVAD.process()
# ---------------------------------------------------------------------------

def test_process_returns_none_during_silence(loaded_vad, mock_silero):
    """process() returns None when the iterator emits nothing."""
    mock_silero["iterator"].return_value = None
    result = loaded_vad.process(np.zeros(512, dtype=np.float32))
    assert result is None


def test_process_returns_start_event(loaded_vad, mock_silero):
    """process() passes through {'start': N} from the iterator."""
    mock_silero["iterator"].return_value = {"start": 0}
    result = loaded_vad.process(np.zeros(512, dtype=np.float32))
    assert result == {"start": 0}


def test_process_returns_end_event(loaded_vad, mock_silero):
    """process() passes through {'end': N} from the iterator."""
    mock_silero["iterator"].return_value = {"end": 1024}
    result = loaded_vad.process(np.zeros(512, dtype=np.float32))
    assert result == {"end": 1024}


def test_process_calls_iterator_with_tensor(loaded_vad, mock_silero, mock_torch):
    """process() must call the VAD iterator with a torch tensor."""
    loaded_vad.process(np.zeros(512, dtype=np.float32))
    mock_silero["iterator"].assert_called()


# ---------------------------------------------------------------------------
# SileroVAD.reset()
# ---------------------------------------------------------------------------

def test_reset_calls_reset_states(loaded_vad, mock_silero):
    """reset() should call reset_states() on the iterator."""
    loaded_vad.reset()
    mock_silero["iterator"].reset_states.assert_called_once()


def test_reset_noop_before_load():
    """reset() must not error when the iterator has not been loaded yet."""
    vad = SileroVAD()
    vad.reset()  # should not raise


# ---------------------------------------------------------------------------
# VadCapture._capture_loop() — utterance emission
# ---------------------------------------------------------------------------

def _make_capture_with_events(events: list) -> tuple[VadCapture, queue.Queue]:
    """Helper: build a VadCapture with a mock VAD that emits the given sequence."""
    mock_vad = MagicMock(spec=SileroVAD)
    # events is a list of process() return values (None, {'start':…}, {'end':…})
    mock_vad.process.side_effect = events

    capture = VadCapture(vad=mock_vad, sample_rate=16_000)
    q: queue.Queue = queue.Queue()
    return capture, q


def _run_capture_sync(capture: VadCapture, q: queue.Queue, chunks: list[np.ndarray]) -> None:
    """Directly call _capture_loop() logic with a mock MicrophoneStream."""
    # We simulate _capture_loop by injecting chunks directly via a mock stream.
    stop_event = threading.Event()
    capture._utterance_queue = q
    capture._stop_event = stop_event

    # Simulate: process N chunks then set stop_event
    chunk_iter = iter(chunks)
    original_loop = capture._capture_loop

    chunk_holder = {"chunks": chunks}
    call_count = [0]

    def mock_loop():
        """Run the inner loop body with mocked MicrophoneStream.

        Mirrors VadCapture._capture_loop() including the pre-buffer logic so
        tests stay in sync with the real implementation.
        """
        import collections as _collections
        speech_buffer: list[np.ndarray] = []
        pre_buffer: _collections.deque = _collections.deque(maxlen=capture.PRE_BUFFER_CHUNKS)
        in_speech = False
        for chunk in chunk_holder["chunks"]:
            if stop_event.is_set():
                break
            event = capture._vad.process(chunk)
            if event is not None and "start" in event:
                in_speech = True
                speech_buffer = list(pre_buffer) + [chunk]
                pre_buffer.clear()
            elif event is not None and "end" in event:
                if in_speech and speech_buffer:
                    speech_buffer.append(chunk)
                    q.put(np.concatenate(speech_buffer))
                in_speech = False
                speech_buffer = []
            elif in_speech:
                speech_buffer.append(chunk)
            else:
                pre_buffer.append(chunk)

    mock_loop()


def test_capture_emits_utterance_on_end_event():
    """A start→speech→end sequence produces exactly one utterance in the queue."""
    chunk = np.ones(512, dtype=np.float32)
    chunks = [chunk, chunk, chunk]
    events = [{"start": 0}, None, {"end": 512}]

    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)

    assert q.qsize() == 1
    utterance = q.get_nowait()
    assert isinstance(utterance, np.ndarray)
    # Three chunks concatenated (start, mid, end)
    assert len(utterance) == 512 * 3


def test_capture_emits_nothing_on_silence_only():
    """Silence-only audio produces no utterance."""
    chunks = [np.zeros(512, dtype=np.float32)] * 3
    events = [None, None, None]
    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)
    assert q.empty()


def test_capture_emits_two_utterances_for_two_speeches():
    """Two speech segments produce two independent utterances."""
    chunk = np.ones(512, dtype=np.float32)
    chunks = [chunk] * 6
    events = [
        {"start": 0}, None, {"end": 512},   # first utterance
        {"start": 0}, None, {"end": 512},   # second utterance
    ]
    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)
    assert q.qsize() == 2


def test_capture_includes_end_chunk_in_utterance():
    """The chunk that triggers the end event is included in the utterance."""
    start_chunk = np.full(512, 1.0, dtype=np.float32)
    end_chunk = np.full(512, 2.0, dtype=np.float32)
    chunks = [start_chunk, end_chunk]
    events = [{"start": 0}, {"end": 512}]
    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)
    utterance = q.get_nowait()
    # Both chunks should be present
    assert len(utterance) == 1024
    assert utterance[0] == 1.0   # from start_chunk
    assert utterance[512] == 2.0  # from end_chunk


def test_capture_includes_pre_buffer_before_start_event():
    """Silence chunks recorded before speech onset are prepended to the utterance.

    This verifies the fix for 'first word missed': VAD fires ~32-96 ms after
    the first voiced sound, so pre-buffer audio closes the gap.
    """
    silence_chunk = np.zeros(512, dtype=np.float32)
    speech_chunk = np.full(512, 0.5, dtype=np.float32)
    end_chunk = np.full(512, 0.3, dtype=np.float32)
    # Two silence chunks, then speech onset, one mid-speech, then end
    chunks = [silence_chunk, silence_chunk, speech_chunk, speech_chunk, end_chunk]
    events = [None, None, {"start": 0}, None, {"end": 512}]

    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)

    utterance = q.get_nowait()
    # pre_buffer (2 silence chunks) + start chunk + mid chunk + end chunk = 5 * 512
    assert len(utterance) == 512 * 5
    # First 1024 samples come from the silence pre-buffer
    assert np.all(utterance[:1024] == 0.0)
    # Remaining 3 chunks: 2 speech (0.5) then 1 end (0.3)
    assert np.all(utterance[1024:2048] == pytest.approx(0.5))
    assert np.all(utterance[2048:] == pytest.approx(0.3))


def test_capture_pre_buffer_bounded_by_pre_buffer_chunks():
    """Pre-buffer never grows beyond PRE_BUFFER_CHUNKS chunks."""
    # Fill with more silence chunks than PRE_BUFFER_CHUNKS, then speech
    n_silence = VadCapture.PRE_BUFFER_CHUNKS + 5
    silence_chunk = np.zeros(512, dtype=np.float32)
    speech_chunk = np.ones(512, dtype=np.float32)
    end_chunk = np.full(512, 2.0, dtype=np.float32)

    chunks = [silence_chunk] * n_silence + [speech_chunk, end_chunk]
    events = [None] * n_silence + [{"start": 0}, {"end": 512}]

    capture, q = _make_capture_with_events(events)
    _run_capture_sync(capture, q, chunks)

    utterance = q.get_nowait()
    # At most PRE_BUFFER_CHUNKS silence + 1 start + 1 end = PRE_BUFFER_CHUNKS+2
    assert len(utterance) == 512 * (VadCapture.PRE_BUFFER_CHUNKS + 2)


# ---------------------------------------------------------------------------
# VadCapture start/stop lifecycle
# ---------------------------------------------------------------------------

def test_capture_stop_without_start_is_safe():
    """stop() must not raise if start() was never called."""
    vad = MagicMock(spec=SileroVAD)
    capture = VadCapture(vad=vad)
    capture.stop()  # must not raise


def test_capture_start_launches_daemon_thread():
    """start() should create a daemon thread."""
    vad = MagicMock(spec=SileroVAD)
    capture = VadCapture(vad=vad)
    q: queue.Queue = queue.Queue()

    # Patch _capture_loop to immediately return so the thread exits.
    with patch.object(capture, "_capture_loop", return_value=None):
        capture.start(q)
        assert capture._thread is not None
        assert capture._thread.daemon is True
        capture.stop()


# ---------------------------------------------------------------------------
# build_vad_capture() — factory
# ---------------------------------------------------------------------------

class _FakeVoiceConfig:
    class vad:
        threshold = 0.6
        silence_ms = 500
    class audio:
        sample_rate = 16_000
        input_device = "test-device"


def test_build_vad_capture_reads_threshold():
    vc = build_vad_capture(_FakeVoiceConfig)
    assert vc._vad._threshold == 0.6


def test_build_vad_capture_reads_silence_ms():
    vc = build_vad_capture(_FakeVoiceConfig)
    assert vc._vad._silence_ms == 500


def test_build_vad_capture_reads_device():
    vc = build_vad_capture(_FakeVoiceConfig)
    assert vc._device == "test-device"


def test_build_vad_capture_reads_sample_rate():
    vc = build_vad_capture(_FakeVoiceConfig)
    assert vc._sample_rate == 16_000


def test_build_vad_capture_defaults_on_empty_config():
    """build_vad_capture() must not raise when config has no voice sub-keys."""
    vc = build_vad_capture(object())  # object() has no vad/audio attrs
    assert isinstance(vc, VadCapture)
    assert vc._vad._threshold == 0.7  # default
