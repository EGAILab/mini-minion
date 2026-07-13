"""Tests for voice/tts.py — TTS adapters, speak(), and factory.

All ML packages are mocked via sys.modules.  Tests cover:

- Qwen3TTS.load() initialises the transformers pipeline with the right task
- Qwen3TTS.synthesise() calls the pipeline and unpacks audio + sample_rate
- KokoroTTS.load() initialises KPipeline with the right lang_code
- KokoroTTS.synthesise() concatenates generator chunks
- PiperTTS.load() calls PiperVoice.load() with model path
- PiperTTS.synthesise() returns float32 samples in [-1, 1]
- speak() chains synthesise() + play_audio()
- build_tts() returns the correct adapter from config
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from minion_assist.voice.tts import (
    KokoroTTS,
    PiperTTS,
    Qwen3TTS,
    TTSAdapter,
    build_tts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_transformers_tts(monkeypatch):
    """Inject a mock transformers pipeline for TTS."""
    samples = np.ones(24_000, dtype=np.float32) * 0.5
    mock_result = {"audio": samples, "sampling_rate": 24_000}
    mock_pipe = MagicMock(return_value=mock_result)
    mock_pipeline_fn = MagicMock(return_value=mock_pipe)

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_dtype"
    mock_torch.float32 = "float32_dtype"

    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline_fn

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    return {"pipe": mock_pipe, "pipeline_fn": mock_pipeline_fn, "result": mock_result}


@pytest.fixture()
def mock_kokoro(monkeypatch):
    """Inject a mock kokoro package."""
    mock_pipeline_instance = MagicMock()
    # Simulate generator: [(samples, sr, phonemes), ...]
    chunk1 = ("graphemes1", "phonemes1", np.ones(12_000, dtype=np.float32) * 0.3)
    chunk2 = ("graphemes2", "phonemes2", np.ones(12_000, dtype=np.float32) * 0.7)
    mock_pipeline_instance.return_value = iter([chunk1, chunk2])

    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
    mock_kokoro_pkg = MagicMock()
    mock_kokoro_pkg.KPipeline = mock_pipeline_cls

    monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro_pkg)

    return {"pipeline": mock_pipeline_instance, "cls": mock_pipeline_cls}


@pytest.fixture()
def mock_piper(monkeypatch, tmp_path):
    """Inject a mock piper package and create a dummy model file.

    PiperTTS calls ``PiperVoice.load(model_path, config_path=...)`` which is
    a classmethod on the real Piper library.  The mock therefore wires
    ``pkg.PiperVoice.load.return_value`` to the fake voice instance so that
    ``synthesize()`` can be given a realistic side_effect.
    """
    model_path = str(tmp_path / "voice.onnx")
    open(model_path, "w").close()  # create empty file so load() passes path check

    def fake_synthesize(text, wav_file):
        """Write a tiny valid WAV (100 int16 samples at 22050 Hz)."""
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(np.zeros(100, dtype=np.int16).tobytes())

    # The voice instance returned by PiperVoice.load()
    mock_voice = MagicMock()
    mock_voice.synthesize.side_effect = fake_synthesize

    mock_piper_pkg = MagicMock()
    # PiperVoice.load(...) is a classmethod; make it return our fake voice.
    mock_piper_pkg.PiperVoice.load.return_value = mock_voice
    monkeypatch.setitem(sys.modules, "piper", mock_piper_pkg)

    return {"voice": mock_voice, "pkg": mock_piper_pkg, "model_path": model_path}


# ---------------------------------------------------------------------------
# TTSAdapter abstract base
# ---------------------------------------------------------------------------

def test_tts_adapter_is_abstract():
    """TTSAdapter cannot be instantiated directly (synthesise is abstract)."""
    with pytest.raises(TypeError):
        TTSAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Qwen3TTS.load()
# ---------------------------------------------------------------------------

def test_qwen3_load_calls_pipeline_with_tts_task(mock_transformers_tts):
    """load() must call transformers.pipeline with 'text-to-speech'."""
    tts = Qwen3TTS(model_id="Qwen/Qwen3-TTS-1.7B", device="cpu")
    tts.load()
    call_args = mock_transformers_tts["pipeline_fn"].call_args
    assert call_args.args[0] == "text-to-speech"


def test_qwen3_load_passes_model_id(mock_transformers_tts):
    tts = Qwen3TTS(model_id="Qwen/Qwen3-TTS-0.6B", device="cpu")
    tts.load()
    kwargs = mock_transformers_tts["pipeline_fn"].call_args.kwargs
    assert kwargs["model"] == "Qwen/Qwen3-TTS-0.6B"


def test_qwen3_load_is_idempotent(mock_transformers_tts):
    tts = Qwen3TTS()
    tts.load()
    tts.load()
    mock_transformers_tts["pipeline_fn"].assert_called_once()


def test_qwen3_load_missing_transformers_raises():
    with patch.dict(sys.modules, {"transformers": None, "torch": None}):
        tts = Qwen3TTS()
        with pytest.raises(RuntimeError, match="transformers"):
            tts.load()


# ---------------------------------------------------------------------------
# Qwen3TTS.synthesise()
# ---------------------------------------------------------------------------

def test_qwen3_synthesise_returns_ndarray_and_rate(mock_transformers_tts):
    tts = Qwen3TTS()
    samples, rate = tts.synthesise("hello")
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert isinstance(rate, int)


def test_qwen3_synthesise_calls_pipeline_with_text(mock_transformers_tts):
    tts = Qwen3TTS()
    tts.synthesise("say this")
    mock_transformers_tts["pipe"].assert_called_once()
    arg = mock_transformers_tts["pipe"].call_args.args[0]
    assert arg == "say this"


def test_qwen3_synthesise_returns_correct_sample_rate(mock_transformers_tts):
    tts = Qwen3TTS()
    _, rate = tts.synthesise("test")
    assert rate == 24_000


def test_qwen3_synthesise_auto_loads(mock_transformers_tts):
    tts = Qwen3TTS()
    assert tts._pipeline is None
    tts.synthesise("hi")
    assert tts._pipeline is not None


# ---------------------------------------------------------------------------
# KokoroTTS.load()
# ---------------------------------------------------------------------------

def test_kokoro_load_creates_kpipeline(mock_kokoro):
    tts = KokoroTTS(lang_code="a")
    tts.load()
    mock_kokoro["cls"].assert_called_once_with(lang_code="a")


def test_kokoro_load_is_idempotent(mock_kokoro):
    tts = KokoroTTS()
    tts.load()
    tts.load()
    mock_kokoro["cls"].assert_called_once()


def test_kokoro_load_missing_kokoro_raises():
    with patch.dict(sys.modules, {"kokoro": None}):
        tts = KokoroTTS()
        with pytest.raises(RuntimeError, match="kokoro not installed"):
            tts.load()


# ---------------------------------------------------------------------------
# KokoroTTS.synthesise()
# ---------------------------------------------------------------------------

def test_kokoro_synthesise_concatenates_chunks(mock_kokoro):
    """synthesise() must concatenate all generator chunks into one array."""
    tts = KokoroTTS(voice="af_heart")
    # Re-configure the mock to return a fresh iterator on each call
    chunk1 = ("gs1", "ps1", np.ones(12_000, dtype=np.float32) * 0.3)
    chunk2 = ("gs2", "ps2", np.ones(12_000, dtype=np.float32) * 0.7)
    mock_kokoro["pipeline"].return_value = iter([chunk1, chunk2])

    samples, rate = tts.synthesise("hi there")
    assert len(samples) == 24_000  # 12000 + 12000
    assert rate == 24_000


def test_kokoro_synthesise_empty_generator(mock_kokoro):
    """synthesise() with an empty generator should return an empty array."""
    mock_kokoro["pipeline"].return_value = iter([])
    tts = KokoroTTS()
    samples, rate = tts.synthesise("anything")
    assert len(samples) == 0


def test_kokoro_synthesise_passes_voice_and_speed(mock_kokoro):
    mock_kokoro["pipeline"].return_value = iter([])
    tts = KokoroTTS(voice="bf_emma", speed=1.2)
    tts.synthesise("test")
    call_kwargs = mock_kokoro["pipeline"].call_args.kwargs
    assert call_kwargs.get("voice") == "bf_emma"
    assert call_kwargs.get("speed") == 1.2


def test_kokoro_synthesise_auto_loads(mock_kokoro):
    mock_kokoro["pipeline"].return_value = iter([])
    tts = KokoroTTS()
    assert tts._pipeline is None
    tts.synthesise("hi")
    assert tts._pipeline is not None


# ---------------------------------------------------------------------------
# PiperTTS.load()
# ---------------------------------------------------------------------------

def test_piper_load_calls_piper_voice_load(mock_piper):
    """load() must call PiperVoice.load() (the classmethod) with model_path."""
    tts = PiperTTS(model_path=mock_piper["model_path"])
    tts.load()
    mock_piper["pkg"].PiperVoice.load.assert_called_once()


def test_piper_load_missing_model_path_raises():
    tts = PiperTTS(model_path="")
    with pytest.raises(RuntimeError, match="piper_model_path"):
        tts.load()


def test_piper_load_missing_package_raises():
    with patch.dict(sys.modules, {"piper": None}):
        tts = PiperTTS(model_path="/nonexistent.onnx")
        with pytest.raises(RuntimeError, match="piper-tts not installed"):
            tts.load()


def test_piper_load_is_idempotent(mock_piper):
    tts = PiperTTS(model_path=mock_piper["model_path"])
    tts.load()
    tts.load()
    mock_piper["pkg"].PiperVoice.load.assert_called_once()


# ---------------------------------------------------------------------------
# PiperTTS.synthesise()
# ---------------------------------------------------------------------------

def test_piper_synthesise_returns_float32(mock_piper):
    tts = PiperTTS(model_path=mock_piper["model_path"])
    samples, rate = tts.synthesise("hello")
    assert samples.dtype == np.float32


def test_piper_synthesise_normalises_to_minus1_to_1(mock_piper):
    """int16 PCM (32767) should map to ~1.0 in float32."""
    def fake_synthesize(text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        # Write max int16 value
        wav_file.writeframes(np.array([32767], dtype=np.int16).tobytes())

    mock_piper["voice"].synthesize.side_effect = fake_synthesize
    tts = PiperTTS(model_path=mock_piper["model_path"])
    samples, _ = tts.synthesise("peak")
    assert abs(samples[0] - (32767 / 32768.0)) < 0.001


def test_piper_synthesise_returns_sample_rate(mock_piper):
    tts = PiperTTS(model_path=mock_piper["model_path"])
    _, rate = tts.synthesise("hello")
    assert rate == 22050  # as set in fake_synthesize


# ---------------------------------------------------------------------------
# speak() integration
# ---------------------------------------------------------------------------

def test_speak_calls_play_audio(mock_transformers_tts):
    """speak() must call play_audio() with the synthesised samples."""
    tts = Qwen3TTS()
    with patch("minion_assist.voice.audio.play_audio") as mock_play:
        tts.speak("hello")
        mock_play.assert_called_once()
        call_args = mock_play.call_args
        samples_arg = call_args.args[0]
        assert isinstance(samples_arg, np.ndarray)


def test_speak_passes_sample_rate(mock_transformers_tts):
    """speak() must pass the sample_rate returned by synthesise()."""
    tts = Qwen3TTS()
    with patch("minion_assist.voice.audio.play_audio") as mock_play:
        tts.speak("hello")
        kwargs = mock_play.call_args.kwargs
        assert kwargs.get("sample_rate") == 24_000


# ---------------------------------------------------------------------------
# build_tts() factory
# ---------------------------------------------------------------------------

class _FakeTtsQwen:
    class tts:
        model = "qwen3"
        qwen3_model_id = "Qwen/Qwen3-TTS-1.7B"
        qwen3_precision = "fp16"
        kokoro_voice = "af_heart"
        device = "cuda"
        voice_ref_audio = None
        piper_model_path = ""


class _FakeTtsKokoro:
    class tts:
        model = "kokoro"
        qwen3_model_id = "Qwen/Qwen3-TTS-1.7B"
        qwen3_precision = "fp16"
        kokoro_voice = "bf_emma"
        device = "cuda"
        voice_ref_audio = None
        piper_model_path = ""


class _FakeTtsPiper:
    class tts:
        model = "piper"
        qwen3_model_id = ""
        qwen3_precision = "fp16"
        kokoro_voice = "af_heart"
        device = "cpu"
        voice_ref_audio = None
        piper_model_path = "/models/voice.onnx"


def test_build_tts_returns_qwen3_by_default():
    result = build_tts(_FakeTtsQwen)
    assert isinstance(result, Qwen3TTS)


def test_build_tts_returns_kokoro_when_configured():
    result = build_tts(_FakeTtsKokoro)
    assert isinstance(result, KokoroTTS)


def test_build_tts_returns_piper_when_configured():
    result = build_tts(_FakeTtsPiper)
    assert isinstance(result, PiperTTS)


def test_build_tts_passes_kokoro_voice():
    result = build_tts(_FakeTtsKokoro)
    assert result._voice == "bf_emma"


def test_build_tts_passes_piper_model_path():
    result = build_tts(_FakeTtsPiper)
    assert result._model_path == "/models/voice.onnx"


def test_build_tts_passes_qwen3_precision():
    result = build_tts(_FakeTtsQwen)
    assert result._precision == "fp16"


def test_build_tts_defaults_on_empty_config():
    """build_tts() must not raise when config has no tts sub-keys."""
    result = build_tts(object())
    assert isinstance(result, Qwen3TTS)
