"""Text-to-Speech adapters — synthesise audio from agent response text.

Three backends are provided:

``Qwen3TTS``  (default)
    Uses Alibaba's Qwen3-TTS via the ``qwen-tts`` library (NOT transformers).
    High quality, multilingual, voice cloning support.  ~3.4 GB VRAM at BF16
    for the 1.7B model; ~1.2 GB for 0.6B.  Apache 2.0.
    Install: ``uv pip install qwen-tts``

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
from collections.abc import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TTSAdapter(abc.ABC):
    """Abstract text-to-speech adapter.

    Subclasses must implement ``load()`` and ``synthesise()``.

    Set ``supports_streaming = True`` on a subclass to signal that
    ``synthesise_stream()`` yields audio chunks incrementally (lower latency).
    When False, ``VoiceSession`` sentence-splits the text and calls
    ``synthesise()`` once per sentence instead.
    """

    # Overridden to True by adapters whose synthesise_stream() is truly incremental.
    supports_streaming: bool = False

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

    def synthesise_stream(self, text: str) -> "Iterator[tuple[np.ndarray, int]]":
        """Yield synthesised audio chunks for *text*, one at a time.

        The default implementation synthesises the full text in one call and
        yields it as a single chunk (identical to calling ``synthesise()``).
        Backends that produce audio incrementally should override this and set
        ``supports_streaming = True`` so ``VoiceSession`` uses this path.

        Args:
            text: The text to speak.

        Yields:
            tuple[np.ndarray, int]: (samples, sample_rate) per audio chunk.
        """
        yield self.synthesise(text)

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
    """Alibaba Qwen3-TTS via the ``qwen-tts`` library.

    Uses ``Qwen3TTSModel.from_pretrained()`` — NOT the transformers pipeline.
    Supports three synthesis modes:
    - No ref audio: ``generate_custom_voice()`` with a preset speaker name.
    - With ref audio: ``generate_voice_clone()`` using a reference WAV.

    Args:
        model_id: HF model ID.  Default: ``"Qwen/Qwen3-TTS-12Hz-1.7B-Base"``.
        device: ``"cuda"`` or ``"cpu"``.
        precision: ``"bf16"`` (default, ~3.4 GB VRAM) or ``"fp16"``.
        speaker: Preset speaker name for custom-voice mode.  One of the 9
            built-in Qwen3-TTS voices, e.g. ``"Vivian"``.  Ignored when
            ``voice_ref_audio`` is set.
        voice_ref_audio: Path or URL to a WAV for voice cloning.  ``None``
            uses the preset ``speaker`` voice.
        voice_ref_text: Transcript of ``voice_ref_audio`` (improves cloning).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda",
        precision: str = "bf16",
        speaker: str = "Vivian",
        voice_ref_audio: "str | None" = None,
        voice_ref_text: "str | None" = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._precision = precision
        self._speaker = speaker
        self._voice_ref_audio = voice_ref_audio
        self._voice_ref_text = voice_ref_text
        self._model: "object | None" = None

    def load(self) -> None:
        """Load the Qwen3-TTS model.  Downloads from HF on first call."""
        if self._model is not None:
            return
        try:
            import logging  # noqa: PLC0415
            import torch  # noqa: PLC0415
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "qwen-tts not installed. Run: uv pip install qwen-tts"
            )
        # Silence the "Setting pad_token_id to eos_token_id" INFO log that
        # transformers emits on every generate() call when pad_token_id is unset.
        logging.getLogger("transformers").setLevel(logging.ERROR)
        # bfloat16 is recommended by Qwen; fall back to float16 if specified.
        dtype = torch.bfloat16 if self._precision == "bf16" else torch.float16
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_id,
            device_map=self._device,
            dtype=dtype,
        )

    def synthesise(self, text: str) -> "tuple[np.ndarray, int]":
        """Synthesise speech from text using Qwen3-TTS.

        Dispatches to the right generation method based on the loaded model's
        ``tts_model_type`` attribute:

        - ``"custom_voice"`` → ``generate_custom_voice(speaker=...)``
        - ``"base"``         → ``generate_voice_clone(ref_audio=...)``  (requires
          ``voice_ref_audio`` in config)
        - ``"voice_design"`` → ``generate_voice_design(instruct=...)``

        Both ``generate_*`` methods return ``(wavs_batch, sample_rate)`` where
        ``wavs_batch[0]`` is the first (only) utterance's numpy array.

        Args:
            text: Text to convert to speech.

        Returns:
            tuple[np.ndarray, int]: (float32 waveform, sample_rate).
        """
        self.load()
        # Read the model type from the loaded model so any model variant works.
        model_type = getattr(
            getattr(self._model, "model", None), "tts_model_type", "custom_voice"
        )

        if model_type == "base":
            # Base model: voice cloning — ref_audio is required.
            if not self._voice_ref_audio:
                raise RuntimeError(
                    "Qwen3-TTS Base model requires voice.tts.voice_ref_audio in config.json.\n"
                    "For preset speakers without a reference WAV, use model "
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice instead."
                )
            wavs, sr = self._model.generate_voice_clone(  # type: ignore[union-attr]
                text=text,
                language="Auto",
                ref_audio=self._voice_ref_audio,
                ref_text=self._voice_ref_text or None,
            )
        elif model_type == "voice_design":
            # Voice-design model: natural-language description of the desired voice.
            # Reuse the speaker field as the voice description prompt.
            wavs, sr = self._model.generate_voice_design(  # type: ignore[union-attr]
                text=text,
                language="Auto",
                instruct=self._speaker,
            )
        else:
            # custom_voice (default): one of 9 preset speaker names.
            wavs, sr = self._model.generate_custom_voice(  # type: ignore[union-attr]
                text=text,
                language="Auto",
                speaker=self._speaker,
            )
        audio = np.asarray(wavs[0], dtype=np.float32)
        return audio, int(sr)


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------

class KokoroTTS(TTSAdapter):
    """Kokoro-82M text-to-speech via the kokoro Python package.

    Ultra-efficient English TTS with 54 voices and real-time streaming.
    Best for English-only use cases where VRAM headroom matters.

    Kokoro's ``KPipeline`` is a generator — each ``next()`` call returns one
    audio chunk (typically one sentence) without waiting for the full response
    to finish.  ``synthesise_stream()`` exposes this natively so ``VoiceSession``
    can start playing the first chunk while the rest are still being synthesised.

    Args:
        voice: Kokoro voice preset.  Default: ``"af_heart"`` (American female).
        speed: Speech rate multiplier.  1.0 is normal speed.
        lang_code: Language code passed to ``KPipeline``.  ``"a"`` = American
            English; ``"b"`` = British English.  See Kokoro docs for full list.
    """

    # Kokoro yields chunks incrementally; VoiceSession will call synthesise_stream().
    supports_streaming: bool = True

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
        sample_rate = 24_000  # Kokoro's fixed output rate; not in the tuple
        # Kokoro v1.x yields (graphemes, phonemes, audio) — audio is the 3rd element.
        for _, _, samples in self._pipeline(  # type: ignore[operator]
            text, voice=self._voice, speed=self._speed
        ):
            chunks.append(np.asarray(samples, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32), sample_rate
        return np.concatenate(chunks), sample_rate

    def synthesise_stream(self, text: str) -> "Iterator[tuple[np.ndarray, int]]":
        """Yield one audio chunk per Kokoro pipeline segment.

        Kokoro's pipeline produces chunks one at a time (typically one per
        sentence).  Each chunk is yielded immediately so the caller can pipe
        audio to the output stream before synthesis of the next chunk begins,
        giving near-zero first-audio latency for long responses.

        Args:
            text: Full text to synthesise.

        Yields:
            tuple[np.ndarray, int]: (float32 audio chunk, sample_rate=24000).
        """
        self.load()
        for _, _, samples in self._pipeline(  # type: ignore[operator]
            text, voice=self._voice, speed=self._speed
        ):
            yield np.asarray(samples, dtype=np.float32), 24_000


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

    # Default: Qwen3-TTS (CustomVoice variant supports preset speakers without ref audio)
    _default_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_id = getattr(tts_cfg, "qwen3_model_id", _default_id) if tts_cfg else _default_id
    precision = getattr(tts_cfg, "qwen3_precision", "bf16") if tts_cfg else "bf16"
    speaker = getattr(tts_cfg, "qwen3_speaker", "Vivian") if tts_cfg else "Vivian"
    voice_ref = getattr(tts_cfg, "voice_ref_audio", None) if tts_cfg else None
    voice_ref_text = getattr(tts_cfg, "voice_ref_text", None) if tts_cfg else None
    return Qwen3TTS(
        model_id=model_id,
        device=device,
        precision=precision,
        speaker=speaker,
        voice_ref_audio=voice_ref,
        voice_ref_text=voice_ref_text,
    )
