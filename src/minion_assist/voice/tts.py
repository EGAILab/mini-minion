"""Text-to-Speech adapters — synthesise audio from agent response text.

Three backends are provided:

``Qwen3TTS``  (default)
    Uses Alibaba's Qwen3-TTS-1.7B model via Hugging Face ``transformers``.
    High quality, multilingual, voice cloning support.  ~4.5 GB VRAM at FP16;
    ~1 GB at INT4.  Apache 2.0.

``KokoroTTS``
    Uses the Kokoro-82M model.  Ultra-efficient (~2-3 GB VRAM), English-focused
    (54 voices), real-time streaming output.  Apache 2.0.

``PiperTTS``
    CPU-only fallback using the Piper TTS engine.  No GPU required, 180× RT
    on CPU, lower quality ceiling.  MIT licence.

The backend is chosen by ``voice.tts.model`` in config.json:
- ``"qwen3"``  (default) → Qwen3TTS
- ``"kokoro"``           → KokoroTTS
- ``"piper"``            → PiperTTS

All ML imports happen inside ``load()`` so the module can be imported without
the voice extra installed.

Output format
-------------
``synthesise()`` returns a 1-D float32 numpy array.  The sample rate is
returned alongside it as a tuple so the caller can pass both to ``play_audio()``.
``speak()`` is a convenience wrapper that synthesises *and* plays immediately.
"""
from __future__ import annotations

import abc

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TTSAdapter(abc.ABC):
    """Abstract text-to-speech adapter.

    Subclasses must implement ``load()`` and ``synthesise()``.
    """

    def load(self) -> None:
        """Load the model into memory.

        Call this at startup to pay the load cost once.  ``synthesise()``
        calls this automatically on first use.
        """

    @abc.abstractmethod
    def synthesise(self, text: str) -> "tuple[np.ndarray, int]":
        """Convert text to a speech waveform.

        Args:
            text: The text to speak.

        Returns:
            tuple[np.ndarray, int]: (samples, sample_rate) where samples is
                a 1-D float32 array and sample_rate is the output rate in Hz.
        """

    def speak(
        self,
        text: str,
        device: "str | int | None" = None,
    ) -> None:
        """Synthesise *text* and play it immediately.

        Convenience wrapper around ``synthesise()`` + ``audio.play_audio()``.

        Args:
            text: The text to speak aloud.
            device: sounddevice output device name/index, or None for default.
        """
        from .audio import play_audio  # noqa: PLC0415
        samples, sample_rate = self.synthesise(text)
        play_audio(samples, sample_rate=sample_rate, device=device)


# ---------------------------------------------------------------------------
# Qwen3-TTS (default)
# ---------------------------------------------------------------------------

class Qwen3TTS(TTSAdapter):
    """Alibaba Qwen3-TTS via Hugging Face transformers.

    High naturalness, multilingual, optional voice cloning via reference audio.
    Model runs as a causal language model that generates audio codec tokens,
    which are then decoded to a waveform.

    Args:
        model_id: HF model ID.  Default: ``"Qwen/Qwen3-TTS-1.7B"``.
        device: ``"cuda"`` or ``"cpu"``.
        precision: ``"fp16"`` (default, ~4.5 GB VRAM) or ``"int4"`` (~1 GB VRAM).
        voice_ref_audio: Path to a reference WAV for voice cloning.  None for
            the model's built-in default voice.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-1.7B",
        device: str = "cuda",
        precision: str = "fp16",
        voice_ref_audio: "str | None" = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._precision = precision
        self._voice_ref_audio = voice_ref_audio
        self._pipeline: "object | None" = None

    def load(self) -> None:
        """Load the Qwen3-TTS pipeline.  Downloads from HF on first call."""
        if self._pipeline is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import pipeline  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "transformers / torch not installed. Run: uv sync --extra voice"
            )
        torch_dtype = torch.float16 if self._precision == "fp16" else torch.float32
        # Qwen3-TTS is a text-to-speech pipeline in transformers >= 4.52
        self._pipeline = pipeline(
            "text-to-speech",
            model=self._model_id,
            torch_dtype=torch_dtype,
            device=self._device,
        )

    def synthesise(self, text: str) -> "tuple[np.ndarray, int]":
        """Synthesise speech from text using Qwen3-TTS.

        Args:
            text: Text to convert to speech.

        Returns:
            tuple[np.ndarray, int]: (float32 waveform, sample_rate).
        """
        self.load()
        kwargs: dict = {}
        if self._voice_ref_audio:
            kwargs["forward_params"] = {"voice": self._voice_ref_audio}
        result = self._pipeline(text, **kwargs)  # type: ignore[operator]
        # transformers TTS pipeline returns {"audio": array, "sampling_rate": int}
        audio = np.asarray(result["audio"], dtype=np.float32).squeeze()  # type: ignore[index]
        sample_rate: int = result["sampling_rate"]  # type: ignore[index]
        return audio, sample_rate


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------

class KokoroTTS(TTSAdapter):
    """Kokoro-82M text-to-speech via the kokoro Python package.

    Ultra-efficient English TTS with 54 voices and real-time streaming.
    Best for English-only use cases where VRAM headroom matters.

    Args:
        voice: Kokoro voice preset.  Default: ``"af_heart"`` (American female).
        speed: Speech rate multiplier.  1.0 is normal speed.
        lang_code: Language code passed to ``KPipeline``.  ``"a"`` = American
            English; ``"b"`` = British English.  See Kokoro docs for full list.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> None:
        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code
        self._pipeline: "object | None" = None

    def load(self) -> None:
        """Load the Kokoro pipeline."""
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "kokoro not installed. Run: uv add kokoro && uv sync"
            )
        self._pipeline = KPipeline(lang_code=self._lang_code)

    def synthesise(self, text: str) -> "tuple[np.ndarray, int]":
        """Synthesise speech using Kokoro.

        Kokoro returns a generator of (samples, sample_rate, phonemes) tuples.
        We concatenate all chunks into one array.

        Args:
            text: Text to speak.

        Returns:
            tuple[np.ndarray, int]: (float32 waveform, sample_rate).
        """
        self.load()
        chunks: list[np.ndarray] = []
        sample_rate = 24_000  # Kokoro's native output rate
        for samples, sr, _ in self._pipeline(  # type: ignore[operator]
            text, voice=self._voice, speed=self._speed
        ):
            chunks.append(np.asarray(samples, dtype=np.float32))
            sample_rate = int(sr)
        if not chunks:
            return np.zeros(0, dtype=np.float32), sample_rate
        return np.concatenate(chunks), sample_rate


# ---------------------------------------------------------------------------
# Piper TTS (CPU fallback)
# ---------------------------------------------------------------------------

class PiperTTS(TTSAdapter):
    """CPU-only TTS using the Piper engine.

    No GPU required; 180× real-time on CPU.  Best as a lightweight fallback or
    for edge deployments without a discrete GPU.  MIT licence.

    Piper voices must be downloaded separately from ``rhasspy/piper-voices``.
    Set ``model_path`` to the ``.onnx`` file and ``config_path`` to its
    accompanying ``.json`` config.

    Args:
        model_path: Absolute path to the Piper ``.onnx`` model file.
        config_path: Absolute path to the ``.onnx.json`` config, or ``None``
            to auto-detect (appends ``.json`` to ``model_path``).
    """

    def __init__(
        self,
        model_path: str = "",
        config_path: "str | None" = None,
    ) -> None:
        self._model_path = model_path
        self._config_path = config_path or (model_path + ".json")
        self._voice: "object | None" = None

    def load(self) -> None:
        """Load the Piper voice model from disk."""
        if self._voice is not None:
            return
        if not self._model_path:
            raise RuntimeError(
                "PiperTTS requires voice.tts.piper_model_path in config.json.\n"
                "Download a voice from: https://github.com/rhasspy/piper-voices"
            )
        try:
            from piper import PiperVoice  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "piper-tts not installed. Run: uv add piper-tts && uv sync"
            )
        self._voice = PiperVoice.load(self._model_path, config_path=self._config_path)

    def synthesise(self, text: str) -> "tuple[np.ndarray, int]":
        """Synthesise speech using Piper.

        Args:
            text: Text to speak.

        Returns:
            tuple[np.ndarray, int]: (int16 samples as float32, sample_rate).
        """
        self.load()
        import io  # noqa: PLC0415
        import wave  # noqa: PLC0415
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize(text, wav_file)  # type: ignore[union-attr]
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
        # Convert int16 PCM to float32 in [-1, 1]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sample_rate


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tts(voice_config: object) -> TTSAdapter:
    """Build the configured TTS adapter from voice config.

    Reads ``voice.tts.model`` to choose the backend:
    - ``"qwen3"``  (default) → Qwen3TTS
    - ``"kokoro"``           → KokoroTTS
    - ``"piper"``            → PiperTTS

    Args:
        voice_config: A ``VoiceConfig`` instance from config.py.

    Returns:
        TTSAdapter: The configured adapter (model not yet loaded).
    """
    tts_cfg = getattr(voice_config, "tts", None)
    model_name = getattr(tts_cfg, "model", "qwen3") if tts_cfg else "qwen3"
    device = getattr(tts_cfg, "device", "cuda") if tts_cfg else "cuda"

    if model_name == "kokoro":
        voice = getattr(tts_cfg, "kokoro_voice", "af_heart") if tts_cfg else "af_heart"
        return KokoroTTS(voice=voice)

    if model_name == "piper":
        model_path = getattr(tts_cfg, "piper_model_path", "") if tts_cfg else ""
        return PiperTTS(model_path=model_path)

    # Default: Qwen3-TTS
    model_id = getattr(tts_cfg, "qwen3_model_id", "Qwen/Qwen3-TTS-1.7B") if tts_cfg else "Qwen/Qwen3-TTS-1.7B"
    precision = getattr(tts_cfg, "qwen3_precision", "fp16") if tts_cfg else "fp16"
    voice_ref = getattr(tts_cfg, "voice_ref_audio", None) if tts_cfg else None
    return Qwen3TTS(
        model_id=model_id,
        device=device,
        precision=precision,
        voice_ref_audio=voice_ref,
    )
