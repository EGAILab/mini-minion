"""Silero VAD wrapper — detects speech boundaries in real-time audio.

Voice Activity Detection (VAD) sits between the microphone and the STT model.
Its job is to decide when the user has *finished* speaking so that STT receives
a clean, complete utterance rather than a partial one or a stream of silence.

How it works
------------
1. ``VadCapture`` opens a ``MicrophoneStream`` in a background thread.
2. Each 512-sample chunk (32 ms at 16 kHz) is passed to a ``SileroVAD`` instance.
3. ``SileroVAD`` delegates to the Silero VAD model's ``VADIterator``, which
   tracks speech / silence boundaries internally.
4. When the iterator emits an end-of-speech event, all buffered speech chunks
   are concatenated into one utterance array and pushed to a ``queue.Queue``.
5. The main thread consumes utterances from that queue and passes them to STT.

Why Silero VAD?
---------------
- Tiny (~2 MB), CPU-only, Apache 2.0
- <1 ms latency per chunk
- Works with the ``silero-vad`` pip package (no internet access needed at inference)
- Native Python API with a clean ``VADIterator`` abstraction

The silero_vad package is imported lazily so the module can be imported when
the voice extra is not installed; errors surface at runtime with install hints.
"""
from __future__ import annotations

import collections
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

from .audio import MicrophoneStream

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Low-level VAD model wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """Wraps the Silero VAD model for frame-level speech detection.

    Lazy-loads the model on first use so the class can be instantiated without
    the voice extra installed.

    Args:
        threshold: Speech probability above which a frame is considered speech.
            Silero recommends 0.5 for normal conditions.
        sample_rate: Audio sample rate in Hz.  Must be 8000 or 16000.
        silence_ms: Minimum silence duration (ms) to trigger end-of-utterance.
        speech_pad_ms: Milliseconds of padding added around speech segments.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        sample_rate: int = 16_000,
        silence_ms: int = 1200,
        speech_pad_ms: int = 100,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._silence_ms = silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._iterator: "object | None" = None  # VADIterator, loaded lazily

    def _load(self) -> None:
        """Load the Silero VAD model if not already loaded."""
        if self._iterator is not None:
            return
        try:
            from silero_vad import VADIterator, load_silero_vad  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "silero-vad not installed. Run: uv sync --extra voice"
            )
        model = load_silero_vad()
        # VADIterator manages state across chunks; it returns start/end dicts.
        self._iterator = VADIterator(
            model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
            min_silence_duration_ms=self._silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )

    def process(self, chunk: np.ndarray) -> "dict | None":
        """Process one audio chunk and return a VAD event, if any.

        Args:
            chunk: 1-D float32 array of exactly 512 samples (at 16 kHz).

        Returns:
            dict | None:
                ``{'start': sample_index}`` when speech begins,
                ``{'end': sample_index}`` when speech ends,
                ``None`` during mid-speech or sustained silence.
        """
        self._load()
        try:
            import torch  # noqa: PLC0415
            tensor = torch.from_numpy(chunk).float()
            return self._iterator(tensor, return_seconds=False)  # type: ignore[union-attr]
        except ImportError:
            raise RuntimeError(
                "torch not installed. Run: uv sync --extra voice"
            )

    def reset(self) -> None:
        """Reset internal state between utterances or after a long silence."""
        if self._iterator is not None:
            self._iterator.reset_states()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# High-level capture manager
# ---------------------------------------------------------------------------

class VadCapture:
    """Manages a background capture thread that emits utterances to a queue.

    Call ``start(utterance_queue)`` to begin; the background thread runs until
    ``stop()`` is called or the process exits.  Each complete utterance is a
    1-D float32 numpy array placed in ``utterance_queue``.

    Args:
        vad: A ``SileroVAD`` instance to use for speech detection.
        sample_rate: Microphone capture rate in Hz (must match VAD expectation).
        device: sounddevice input device name/index, or None for system default.
    """

    # Silero VAD requires exactly 512 samples per chunk at 16 kHz.
    BLOCKSIZE = 512
    # Rolling pre-speech buffer size.  VAD onset lags ~1–3 chunks behind the
    # first voiced sound; keeping 10 chunks (~320 ms) ensures the first word is
    # always included even at high thresholds.
    PRE_BUFFER_CHUNKS = 10

    def __init__(
        self,
        vad: SileroVAD,
        sample_rate: int = 16_000,
        device: "str | int | None" = None,
    ) -> None:
        self._vad = vad
        self._sample_rate = sample_rate
        self._device = device
        self._thread: "threading.Thread | None" = None
        self._stop_event: "threading.Event | None" = None
        self._utterance_queue: "queue.Queue | None" = None
        # Set the moment VAD detects speech START; cleared after the utterance
        # is committed to the queue.  Lets _speak_streaming interrupt playback
        # immediately when the user begins talking, rather than waiting for the
        # full utterance + silence_ms to elapse.
        self.speech_started: threading.Event = threading.Event()
        # Set by VoiceSession while TTS audio is playing through the speaker.
        # When speech onset fires during TTS, the pre-buffer is discarded
        # because it contains loudspeaker bleed-through rather than genuine
        # pre-speech silence — including it causes STT to transcribe TTS audio
        # mixed with the user's voice (e.g. "stop" → "C'est tout.").
        self.tts_playing: threading.Event = threading.Event()

    def start(self, utterance_queue: "queue.Queue[np.ndarray]") -> None:
        """Start the background capture + VAD thread.

        Args:
            utterance_queue: Caller-provided queue; complete utterances are put here.
        """
        self._utterance_queue = utterance_queue
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,  # dies when the main thread exits
            name="voice-vad-capture",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to exit."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _capture_loop(self) -> None:
        """Run in the background thread: capture audio, feed VAD, emit utterances."""
        assert self._stop_event is not None
        assert self._utterance_queue is not None

        speech_buffer: list[np.ndarray] = []
        # Rolling buffer of the most recent PRE_BUFFER_CHUNKS audio chunks
        # recorded *before* speech is detected.  When the VAD fires a "start"
        # event, this audio is prepended to speech_buffer so the first word is
        # not lost — the model needs 1–3 chunks (~32–96 ms) to confirm onset.
        pre_buffer: collections.deque = collections.deque(maxlen=self.PRE_BUFFER_CHUNKS)
        in_speech = False

        try:
            with MicrophoneStream(
                sample_rate=self._sample_rate,
                blocksize=self.BLOCKSIZE,
                device=self._device,
            ) as mic:
                while not self._stop_event.is_set():
                    chunk = mic.read()
                    event = self._vad.process(chunk)

                    if event is not None and "start" in event:
                        in_speech = True
                        if self.tts_playing.is_set():
                            # Barge-in: pre-buffer holds loudspeaker bleed-through,
                            # not genuine pre-speech audio.  Discard it so STT
                            # only sees the user's voice.
                            speech_buffer = [chunk]
                        else:
                            # Normal onset: include pre-buffer so the first word
                            # is captured even when VAD fires 32–96 ms late.
                            speech_buffer = list(pre_buffer) + [chunk]
                        pre_buffer.clear()
                        # Signal barge-in immediately so _speak_streaming can
                        # abort TTS the instant the user starts talking.
                        self.speech_started.set()
                    elif event is not None and "end" in event:
                        # Speech ended — emit the buffered utterance.
                        if in_speech and speech_buffer:
                            speech_buffer.append(chunk)
                            utterance = np.concatenate(speech_buffer)
                            self._utterance_queue.put(utterance)
                        in_speech = False
                        speech_buffer = []
                        # Clear the barge-in signal now that the utterance is queued.
                        self.speech_started.clear()
                    elif in_speech:
                        # Mid-speech — keep buffering.
                        speech_buffer.append(chunk)
                    else:
                        # Silence before speech — keep rolling pre-buffer full.
                        pre_buffer.append(chunk)

        except Exception as exc:
            # Surface capture errors rather than silently dying.
            import sys  # noqa: PLC0415
            print(f"[voice-vad] Capture thread error: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_vad_capture(voice_config: object) -> VadCapture:
    """Build a VadCapture from config.

    Args:
        voice_config: A ``VoiceConfig`` instance from config.py.

    Returns:
        VadCapture: Ready to use; call ``.start(queue)`` to begin capturing.
    """
    vad_cfg = getattr(voice_config, "vad", None)
    audio_cfg = getattr(voice_config, "audio", None)

    threshold = getattr(vad_cfg, "threshold", 0.7) if vad_cfg else 0.7
    silence_ms = getattr(vad_cfg, "silence_ms", 1200) if vad_cfg else 1200
    sample_rate = getattr(audio_cfg, "sample_rate", 16_000) if audio_cfg else 16_000
    device = getattr(audio_cfg, "input_device", None) if audio_cfg else None

    vad = SileroVAD(
        threshold=threshold,
        sample_rate=sample_rate,
        silence_ms=silence_ms,
    )
    return VadCapture(vad=vad, sample_rate=sample_rate, device=device)
