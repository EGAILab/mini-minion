"""Tests for voice/stt.py — STT adapters and factory.

No real NeMo or transformers models are loaded.  ML packages are injected via
sys.modules monkeypatching, following the same pattern as test_tools_browser.py.

Test coverage:
- ParakeetSTT.load() initialises the NeMo model correctly
- ParakeetSTT.transcribe() calls model.transcribe() and unpacks the result
- WhisperSTT.load() initialises the transformers pipeline
- WhisperSTT.transcribe() feeds the audio dict and extracts "text"
- Both adapters handle lazy loading (load() auto-called by transcribe())
- Both adapters raise RuntimeError with install hints when dependencies absent
- build_stt() returns the correct adapter type based on config
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minion_assist.voice.stt import ParakeetSTT, STTAdapter, WhisperSTT, build_stt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_nemo(monkeypatch):
    """Inject a mock nemo.collections.asr module.

    ParakeetSTT calls ``nemo_asr.models.ASRModel.from_pretrained()``, so the
    mock must expose the ``.models.ASRModel`` chain on the asr package mock.
    """
    mock_model_instance = MagicMock()
    # transcribe() returns list of strings by default.
    mock_model_instance.transcribe.return_value = ["hello world"]

    # ``import nemo.collections.asr as nemo_asr`` resolves to this object.
    # The code then calls ``nemo_asr.models.ASRModel.from_pretrained()``.
    mock_asr_pkg = MagicMock()
    mock_asr_pkg.models.ASRModel.from_pretrained.return_value = mock_model_instance

    mock_nemo_collections = MagicMock()
    mock_nemo_collections.asr = mock_asr_pkg

    mock_nemo_pkg = MagicMock()
    mock_nemo_pkg.collections = mock_nemo_collections

    monkeypatch.setitem(sys.modules, "nemo", mock_nemo_pkg)
    monkeypatch.setitem(sys.modules, "nemo.collections", mock_nemo_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", mock_asr_pkg)

    return {"model": mock_model_instance, "asr": mock_asr_pkg}


@pytest.fixture()
def mock_transformers_stt(monkeypatch):
    """Inject a mock transformers pipeline for STT."""
    mock_result = {"text": " hello world"}
    mock_pipe = MagicMock(return_value=mock_result)
    mock_pipeline_fn = MagicMock(return_value=mock_pipe)

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_dtype"
    mock_torch.float32 = "float32_dtype"

    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline_fn

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    return {"pipe": mock_pipe, "pipeline_fn": mock_pipeline_fn, "torch": mock_torch}


# ---------------------------------------------------------------------------
# STTAdapter abstract base
# ---------------------------------------------------------------------------

def test_stt_adapter_is_abstract():
    """STTAdapter cannot be instantiated directly (transcribe is abstract)."""
    with pytest.raises(TypeError):
        STTAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ParakeetSTT.load()
# ---------------------------------------------------------------------------

def test_parakeet_load_calls_from_pretrained(mock_nemo):
    """load() should call models.ASRModel.from_pretrained with the model ID."""
    stt = ParakeetSTT(model_id="nvidia/parakeet-tdt-0.6b-v3", device="cpu")
    stt.load()
    # Code path: nemo_asr.models.ASRModel.from_pretrained(model_id)
    mock_nemo["asr"].models.ASRModel.from_pretrained.assert_called_once_with(
        "nvidia/parakeet-tdt-0.6b-v3"
    )


def test_parakeet_load_moves_model_to_device(mock_nemo):
    """load() should call .to(device) on the returned model instance."""
    stt = ParakeetSTT(device="cpu")
    stt.load()
    mock_nemo["model"].to.assert_called_once_with("cpu")


def test_parakeet_load_is_idempotent(mock_nemo):
    """Calling load() twice must not call from_pretrained a second time."""
    stt = ParakeetSTT()
    stt.load()
    stt.load()
    mock_nemo["asr"].models.ASRModel.from_pretrained.assert_called_once()


def test_parakeet_load_missing_nemo_raises():
    """ImportError from nemo must raise RuntimeError with install hint."""
    with patch.dict(sys.modules, {"nemo": None, "nemo.collections": None, "nemo.collections.asr": None}):
        stt = ParakeetSTT()
        with pytest.raises(RuntimeError, match="NeMo ASR not installed"):
            stt.load()


# ---------------------------------------------------------------------------
# ParakeetSTT.transcribe()
# ---------------------------------------------------------------------------

def test_parakeet_transcribe_calls_model_transcribe(mock_nemo):
    """transcribe() should call model.transcribe() with a list containing the audio.

    load() fires a warm-up call, so model.transcribe is called at least twice:
    once during warm-up and once for the real audio.
    """
    stt = ParakeetSTT()
    audio = np.zeros(16_000, dtype=np.float32)
    stt.transcribe(audio)
    # Warm-up fires during load(), then the real transcription fires here.
    assert mock_nemo["model"].transcribe.call_count >= 2
    # The last call must carry the real audio.
    args = mock_nemo["model"].transcribe.call_args.args
    assert isinstance(args[0], list)
    assert len(args[0]) == 1
    assert isinstance(args[0][0], np.ndarray)


def test_parakeet_transcribe_returns_string(mock_nemo):
    """transcribe() should return a stripped string from list[str] results."""
    mock_nemo["model"].transcribe.return_value = ["  Hello world  "]
    stt = ParakeetSTT()
    # Ensure _model points to our mock_model_instance (transcribe returns str list).
    # The result is first[0] which has no .text → falls back to str(first).
    # "  Hello world  " has no .text attr so str("  Hello world  ") = "  Hello world  ".strip() = "Hello world"
    # But wait — MagicMock auto-creates .text; to get str path we need a plain str.
    result = stt.transcribe(np.zeros(512, dtype=np.float32))
    # A plain string has no .text attribute, so str(first).strip() is called.
    assert result == "Hello world"


def test_parakeet_transcribe_unpacks_hypothesis_object(mock_nemo):
    """transcribe() should handle list[Hypothesis] by reading .text attribute."""
    hypothesis = MagicMock()
    hypothesis.text = "from hypothesis"
    mock_nemo["model"].transcribe.return_value = [hypothesis]
    stt = ParakeetSTT()
    result = stt.transcribe(np.zeros(512, dtype=np.float32))
    assert result == "from hypothesis"


def test_parakeet_transcribe_auto_loads(mock_nemo):
    """transcribe() should call load() automatically on first use."""
    stt = ParakeetSTT()
    assert stt._model is None
    stt.transcribe(np.zeros(512, dtype=np.float32))
    assert stt._model is not None


# ---------------------------------------------------------------------------
# WhisperSTT.load()
# ---------------------------------------------------------------------------

def test_whisper_load_calls_pipeline(mock_transformers_stt):
    """load() should call transformers.pipeline with the ASR task."""
    stt = WhisperSTT(model_id="distil-whisper/distil-large-v3", device="cpu")
    stt.load()
    mock_transformers_stt["pipeline_fn"].assert_called_once()
    call_kwargs = mock_transformers_stt["pipeline_fn"].call_args
    assert call_kwargs.args[0] == "automatic-speech-recognition"


def test_whisper_load_uses_model_id(mock_transformers_stt):
    """load() should pass the configured model_id to the pipeline."""
    stt = WhisperSTT(model_id="openai/whisper-large-v3", device="cpu")
    stt.load()
    kwargs = mock_transformers_stt["pipeline_fn"].call_args.kwargs
    assert kwargs["model"] == "openai/whisper-large-v3"


def test_whisper_load_is_idempotent(mock_transformers_stt):
    """Calling load() twice must not create a second pipeline."""
    stt = WhisperSTT()
    stt.load()
    stt.load()
    mock_transformers_stt["pipeline_fn"].assert_called_once()


def test_whisper_load_missing_transformers_raises():
    """ImportError from transformers must raise RuntimeError with hint."""
    with patch.dict(sys.modules, {"transformers": None, "torch": None}):
        stt = WhisperSTT()
        with pytest.raises(RuntimeError, match="transformers"):
            stt.load()


# ---------------------------------------------------------------------------
# WhisperSTT.transcribe()
# ---------------------------------------------------------------------------

def test_whisper_transcribe_passes_dict_to_pipeline(mock_transformers_stt):
    """transcribe() should call the pipeline with a dict containing 'array'."""
    stt = WhisperSTT()
    audio = np.zeros(16_000, dtype=np.float32)
    stt.transcribe(audio, sample_rate=16_000)
    mock_transformers_stt["pipe"].assert_called_once()
    arg = mock_transformers_stt["pipe"].call_args.args[0]
    assert "array" in arg
    assert "sampling_rate" in arg
    assert arg["sampling_rate"] == 16_000


def test_whisper_transcribe_returns_stripped_text(mock_transformers_stt):
    """transcribe() should return pipeline['text'] stripped of whitespace."""
    mock_transformers_stt["pipe"].return_value = {"text": "  hello world  "}
    stt = WhisperSTT()
    result = stt.transcribe(np.zeros(512, dtype=np.float32))
    assert result == "hello world"


def test_whisper_transcribe_auto_loads(mock_transformers_stt):
    """transcribe() should call load() automatically on first use."""
    stt = WhisperSTT()
    assert stt._pipe is None
    stt.transcribe(np.zeros(512, dtype=np.float32))
    assert stt._pipe is not None


# ---------------------------------------------------------------------------
# build_stt() factory
# ---------------------------------------------------------------------------

class _FakeSttConfig:
    model = "parakeet"
    parakeet_model_id = "nvidia/parakeet-tdt-0.6b-v3"
    whisper_model_id = "distil-whisper/distil-large-v3"
    chunk_duration_s = 20
    device = "cuda"


class _FakeVoiceConfigParakeet:
    stt = _FakeSttConfig


class _FakeSttConfigWhisper:
    model = "whisper"
    parakeet_model_id = "nvidia/parakeet-tdt-0.6b-v3"
    whisper_model_id = "distil-whisper/distil-large-v3"
    chunk_duration_s = 20
    device = "cuda"


class _FakeVoiceConfigWhisper:
    stt = _FakeSttConfigWhisper


def test_build_stt_returns_parakeet_by_default():
    result = build_stt(_FakeVoiceConfigParakeet)
    assert isinstance(result, ParakeetSTT)


def test_build_stt_returns_whisper_when_configured():
    result = build_stt(_FakeVoiceConfigWhisper)
    assert isinstance(result, WhisperSTT)


def test_build_stt_passes_model_id_to_parakeet():
    result = build_stt(_FakeVoiceConfigParakeet)
    assert result._model_id == "nvidia/parakeet-tdt-0.6b-v3"


def test_build_stt_passes_model_id_to_whisper():
    result = build_stt(_FakeVoiceConfigWhisper)
    assert result._model_id == "distil-whisper/distil-large-v3"


def test_build_stt_passes_device():
    result = build_stt(_FakeVoiceConfigParakeet)
    assert result._device == "cuda"


def test_build_stt_defaults_on_empty_config():
    """build_stt() must not raise when config has no stt sub-keys."""
    result = build_stt(object())
    assert isinstance(result, ParakeetSTT)
