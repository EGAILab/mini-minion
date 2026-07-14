"""Speech-to-Text adapters — transcribe audio to text.

Two backends are provided:

``ParakeetSTT``
    Uses NVIDIA NeMo's Parakeet TDT 0.6B v3 model.  Best-in-class English
    word error rate, Apache 2.0, ~2 GB VRAM.  English only.

``WhisperSTT``
    Uses Distil-Whisper large-v3 (or any Whisper-family model) via
    Hugging Face ``transformers``.  ~1.6 GB VRAM at int8, multilingual.

The backend is chosen by ``voice.stt.model`` in config.json:
- ``"parakeet"`` (default) → ParakeetSTT
- ``"whisper"``            → WhisperSTT

All ML imports happen inside ``load()`` so the module can be imported without
the voice extra.  RuntimeError with install instructions is raised at inference
time if the dependency is missing.

Audio format expected by both adapters
---------------------------------------
- 1-D numpy float32 array
- 16 000 Hz sample rate
- Mono channel
- Values normalised to [-1.0, 1.0]
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class STTAdapter(abc.ABC):
    """Abstract speech-to-text adapter.

    Subclasses must implement ``load()`` (to load the model) and
    ``transcribe()`` (to convert audio to text).
    """

    def load(self) -> None:
        """Load the model into GPU / CPU memory.

        Call this explicitly at startup to pay the model-load cost once rather
        than on the first transcription request.  The ``transcribe()`` method
        calls this automatically if ``load()`` has not been called yet.
        """

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Convert audio to text.

        Args:
            audio: 1-D float32 array of speech audio.
            sample_rate: Rate in Hz; must match what the model expects (16 kHz).

        Returns:
            str: Transcribed text, stripped of leading/trailing whitespace.
        """


# ---------------------------------------------------------------------------
# Parakeet (NeMo)
# ---------------------------------------------------------------------------

class ParakeetSTT(STTAdapter):
    """NVIDIA NeMo Parakeet TDT ASR model.

    Best-in-class English WER, Apache 2.0, ~2 GB VRAM at float16.  Uses the
    NeMo toolkit's ``ASRModel.from_pretrained()`` for model loading.

    Args:
        model_id: Hugging Face model ID.  Default: ``"nvidia/parakeet-tdt-0.6b-v3"``.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model: "object | None" = None

    def load(self) -> None:
        """Load Parakeet from NeMo.  Downloads from Hugging Face on first call."""
        if self._model is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "NeMo ASR not installed. Run: uv sync --extra voice\n"
                "Then: uv add nemo_toolkit[asr]"
            )
        self._model = nemo_asr.models.ASRModel.from_pretrained(self._model_id)
        # Move to the requested device if the model supports it.
        if hasattr(self._model, "to"):
            self._model.to(self._device)  # type: ignore[union-attr]
        # Warm-up pass: compile CUDA kernels now so the first real utterance
        # doesn't pay the JIT compilation cost (~1–2 s on first GPU call).
        self._model.transcribe([np.zeros(1600, dtype=np.float32)])  # type: ignore[union-attr]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe using Parakeet.

        NeMo's ``transcribe()`` accepts a list of numpy arrays when
        ``batch_size`` and ``channel_selector`` are set.  We wrap the single
        array in a list and unwrap the result.

        Args:
            audio: 1-D float32 audio array at ``sample_rate`` Hz.
            sample_rate: Must be 16 000 for Parakeet.

        Returns:
            str: Transcribed text.
        """
        self.load()
        # NeMo expects the audio as a list of arrays or file paths.
        # verbose=False suppresses the per-batch tqdm progress bar.
        results = self._model.transcribe([audio], verbose=False)  # type: ignore[union-attr]
        # Depending on NeMo version, results is either list[str] or list[Hypothesis].
        first = results[0]
        text = first.text if hasattr(first, "text") else str(first)
        return text.strip()


# ---------------------------------------------------------------------------
# Distil-Whisper (transformers)
# ---------------------------------------------------------------------------

class WhisperSTT(STTAdapter):
    """Distil-Whisper / Whisper STT via Hugging Face transformers.

    Supports all Whisper-family models including ``distil-whisper/distil-large-v3``
    (recommended: ~1.6 GB VRAM int8, multilingual, 6× faster than large-v3).

    Args:
        model_id: HF model ID.  Default: ``"distil-whisper/distil-large-v3"``.
        device: ``"cuda"``, ``"cpu"``, or ``"mps"`` for Apple Silicon.
        torch_dtype: ``"float16"`` or ``"int8"`` for reduced memory.  Only
            ``"float16"`` is natively supported by transformers; int8 requires
            ``bitsandbytes``.  Defaults to ``"float16"``.
    """

    def __init__(
        self,
        model_id: str = "distil-whisper/distil-large-v3",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._torch_dtype = torch_dtype
        self._pipe: "object | None" = None

    def load(self) -> None:
        """Load Whisper pipeline.  Downloads from Hugging Face on first call."""
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import pipeline  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "transformers / torch not installed. Run: uv sync --extra voice"
            )
        dtype = torch.float16 if self._torch_dtype == "float16" else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            torch_dtype=dtype,
            device=self._device,
        )
        # Whisper large-v3 ships with `forced_decoder_ids` in its generation_config
        # which triggers a deprecation warning on every call.  Clear it here so
        # callers can use the task/language API instead (see transcribe()).
        gen_cfg = getattr(getattr(self._pipe, "model", None), "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "forced_decoder_ids", None):
            gen_cfg.forced_decoder_ids = None
        # Same story for the feature extractor's return_token_timestamps flag.
        fe = getattr(self._pipe, "feature_extractor", None)
        if fe is not None and getattr(fe, "return_token_timestamps", False):
            fe.return_token_timestamps = False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe using the Whisper pipeline.

        Passes ``task="transcribe"`` and ``language=None`` explicitly so
        transformers uses the task/language API rather than the deprecated
        ``forced_decoder_ids`` path.  Setting ``language=None`` lets Whisper
        auto-detect the input language, preserving full multilingual support.

        Args:
            audio: 1-D float32 audio array.
            sample_rate: Sampling rate of the audio (must be 16 000 for Whisper).

        Returns:
            str: Transcribed text, in whatever language was spoken.
        """
        self.load()
        # transformers pipeline accepts a dict with 'array' and 'sampling_rate'.
        inputs = {"array": audio.astype(np.float32), "sampling_rate": sample_rate}
        result = self._pipe(  # type: ignore[operator]
            inputs,
            generate_kwargs={"task": "transcribe", "language": None},
        )
        return result["text"].strip()  # type: ignore[index]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_stt(voice_config: object) -> STTAdapter:
    """Build the configured STT adapter from voice config.

    Reads ``voice.stt.model`` to choose the backend:
    - ``"parakeet"`` (default) → ParakeetSTT
    - ``"whisper"``            → WhisperSTT

    Args:
        voice_config: A ``VoiceConfig`` instance from config.py.

    Returns:
        STTAdapter: The configured adapter (model not yet loaded).
    """
    stt_cfg = getattr(voice_config, "stt", None)
    model_name = getattr(stt_cfg, "model", "parakeet") if stt_cfg else "parakeet"
    device = getattr(stt_cfg, "device", "cuda") if stt_cfg else "cuda"

    if model_name == "whisper":
        model_id = (
            getattr(stt_cfg, "whisper_model_id", "distil-whisper/distil-large-v3")
            if stt_cfg else "distil-whisper/distil-large-v3"
        )
        return WhisperSTT(model_id=model_id, device=device)

    # Default: Parakeet
    model_id = (
        getattr(stt_cfg, "parakeet_model_id", "nvidia/parakeet-tdt-0.6b-v3")
        if stt_cfg else "nvidia/parakeet-tdt-0.6b-v3"
    )
    return ParakeetSTT(model_id=model_id, device=device)
