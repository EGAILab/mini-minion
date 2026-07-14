"""Audio I/O helpers — microphone capture and speaker playback.

Uses sounddevice (a thin wrapper around PortAudio) for cross-platform audio.
sounddevice is imported lazily so the module loads even when it is not
installed; a clear error is raised at the point of actual use.

Typical flow
------------
1. Call ``list_devices()`` to find device indices if the defaults don't work.
2. Open a ``MicrophoneStream`` as a context manager to receive audio chunks.
3. Call ``play_audio(samples, sample_rate)`` to output synthesised speech.

Sample rate
-----------
Silero VAD and most ASR models expect 16 000 Hz mono audio.  Everything here
defaults to 16 000 Hz; change ``sample_rate`` in config.json if your device
only supports a different rate (sounddevice will not resample automatically).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_devices() -> str:
    """Return a human-readable table of available audio devices.

    Useful for picking ``input_device`` / ``output_device`` in config.json.
    Requires sounddevice to be installed.

    Returns:
        str: Text table from sounddevice.query_devices().

    Raises:
        RuntimeError: If sounddevice is not installed.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "sounddevice not installed. Run: uv sync --extra voice"
        )
    return str(sd.query_devices())


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_audio(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    device: "str | int | None" = None,
) -> None:
    """Play a numpy audio array through the speaker and block until done.

    Args:
        samples: 1-D float32 array of audio samples in [-1.0, 1.0].
        sample_rate: Playback rate in Hz.  Must match the synthesis rate.
        device: sounddevice device name or index, or None for the system default.

    Raises:
        RuntimeError: If sounddevice is not installed.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "sounddevice not installed. Run: uv sync --extra voice"
        )
    # Ensure float32 mono for consistent behaviour across TTS backends.
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        # Convert stereo to mono by averaging channels.
        arr = arr.mean(axis=1)

    sd.play(arr, samplerate=sample_rate, device=device)
    sd.wait()  # Block until playback is complete.


def stop_playback() -> None:
    """Stop any currently playing audio immediately.

    Useful for interrupting TTS output when the user starts speaking again.
    No-op if nothing is playing or sounddevice is not available.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415
        sd.stop()
    except (ImportError, Exception):
        pass


# ---------------------------------------------------------------------------
# Capture stream
# ---------------------------------------------------------------------------

class MicrophoneStream:
    """Context manager that captures microphone audio in fixed-size chunks.

    Each call to ``read()`` blocks until one chunk of ``blocksize`` samples is
    available.  The stream runs in a background thread managed by sounddevice.

    Example::

        with MicrophoneStream(sample_rate=16000, blocksize=512) as mic:
            while True:
                chunk = mic.read()   # np.ndarray shape (512,) float32
                process(chunk)

    Args:
        sample_rate: Capture rate in Hz.  Must match VAD model expectations.
        blocksize: Samples per chunk.  Silero VAD requires 512 at 16 kHz.
        device: Input device name/index, or None for the system default.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        blocksize: int = 512,
        device: "str | int | None" = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._device = device
        self._queue: "object | None" = None  # queue.Queue at runtime
        self._stream: "object | None" = None

    def __enter__(self) -> "MicrophoneStream":
        import queue  # noqa: PLC0415
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "sounddevice not installed. Run: uv sync --extra voice"
            )

        self._queue = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            # Called on a background audio thread; must not block.
            # indata shape: (blocksize, channels) — slice [:,0] for mono.
            if status:
                import sys  # noqa: PLC0415
                print(f"[audio] {status}", file=sys.stderr)
            self._queue.put(indata[:, 0].copy())  # type: ignore[union-attr]

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=_callback,
            # High latency asks the OS for a larger internal ring buffer so the
            # callback is not starved when the GIL is held during TTS GPU work.
            latency="high",
        )
        self._stream.start()  # type: ignore[union-attr]
        return self

    def read(self) -> np.ndarray:
        """Block until the next audio chunk is available and return it.

        Returns:
            np.ndarray: 1-D float32 array of ``blocksize`` samples.
        """
        return self._queue.get()  # type: ignore[union-attr,return-value]

    def __exit__(self, *args: object) -> None:
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
        self._queue = None
