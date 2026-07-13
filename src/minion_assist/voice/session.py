"""VoiceSession — orchestrates the full speech-to-speech pipeline.

Threading model
---------------
The main voice loop is intentionally single-threaded from Python's perspective:
it calls ``utterance_queue.get()`` (blocking), then runs STT, then calls the
synchronous ``AgentSession.send()``, then runs TTS.  Parallelism lives at a
lower level:

- The VAD + audio capture run in a **background daemon thread** (inside
  ``VadCapture._capture_loop``), pushing utterances to a Queue.
- TTS playback is a blocking sounddevice call on the main thread; it completes
  before the next utterance is consumed.

This keeps the design simple and matches how ``AgentSession.send()`` works
(it is synchronous and not thread-safe).

Interruption handling
---------------------
When the user speaks while TTS is playing (barge-in), ``audio.stop_playback()``
is called before TTS finishes.  The VAD capture thread continues running
throughout, so any speech that started while TTS was active is already buffered.
Because ``play_audio`` is blocking, barge-in is detected by the utterance queue
becoming non-empty *after* synthesis but before playback blocks — a future
enhancement would poll the queue during playback.  Current design: TTS finishes
before the next turn.

Usage
-----
    session = VoiceSession(
        agent_session=sessions["main"],
        stt=build_stt(voice_cfg),
        tts=build_tts(voice_cfg),
        vad_capture=build_vad_capture(voice_cfg),
    )
    session.run()   # blocks until KeyboardInterrupt or stop() is called
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..agents import AgentSession
    from .stt import STTAdapter
    from .tts import TTSAdapter
    from .vad import VadCapture


class VoiceSession:
    """Runs the full voice pipeline: VAD → STT → agent → TTS → playback.

    Args:
        agent_session: An already-configured ``AgentSession`` from ``minion.py``.
            ``send()`` is called on this session for each transcribed utterance.
        stt: An ``STTAdapter`` instance (e.g. ``ParakeetSTT``).
        tts: A ``TTSAdapter`` instance (e.g. ``Qwen3TTS``).
        vad_capture: A ``VadCapture`` instance controlling the microphone thread.
        on_event: Optional event callback; forwarded to ``AgentSession.send()``
            so tool calls and streaming tokens appear in the terminal.
        output_device: sounddevice output device for TTS playback (None = default).
    """

    def __init__(
        self,
        agent_session: "AgentSession",
        stt: "STTAdapter",
        tts: "TTSAdapter",
        vad_capture: "VadCapture",
        on_event: "Callable[[object], None] | None" = None,
        output_device: "str | int | None" = None,
    ) -> None:
        self._agent_session = agent_session
        self._stt = stt
        self._tts = tts
        self._vad_capture = vad_capture
        self._on_event = on_event
        self._output_device = output_device
        # Controls whether the main loop keeps running (set to False by stop()).
        self._running = False
        self._utterance_queue: "queue.Queue[np.ndarray | None]" = queue.Queue()

    def run(self) -> None:
        """Start the voice loop and block until interrupted or ``stop()`` called.

        Pre-loads both models before beginning capture so the user's first
        utterance is not delayed by model loading.

        Raises:
            RuntimeError: If a required ML dependency is missing.
        """
        print("[voice] Loading STT model...", flush=True)
        self._stt.load()
        print("[voice] Loading TTS model...", flush=True)
        self._tts.load()

        self._running = True
        self._vad_capture.start(self._utterance_queue)  # type: ignore[arg-type]
        print("[voice] Ready — speak now. (Ctrl+C to quit)", flush=True)

        try:
            self._loop()
        finally:
            self._running = False
            self._vad_capture.stop()

    def stop(self) -> None:
        """Signal the voice loop to exit on the next iteration.

        Also enqueues a sentinel ``None`` so ``queue.get()`` unblocks
        immediately rather than waiting for the next utterance.
        """
        self._running = False
        self._utterance_queue.put(None)  # sentinel to unblock get()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main voice loop: consume utterances, transcribe, respond, speak."""
        while self._running:
            # Block until VAD emits a complete utterance (or stop() sends None).
            utterance = self._utterance_queue.get()
            if utterance is None:
                # Sentinel: we were asked to stop.
                break

            # --- Transcription ---
            try:
                text = self._stt.transcribe(utterance).strip()
            except Exception as exc:
                print(f"[voice] STT error: {exc}", flush=True)
                continue

            if not text:
                continue

            print(f"\n[you] {text}", flush=True)

            # --- Agent turn ---
            response: str = ""
            try:
                response = self._agent_session.send(
                    text,
                    on_event=self._on_event,
                ) or ""
            except KeyboardInterrupt:
                raise  # Let the caller handle Ctrl+C.
            except Exception as exc:
                print(f"[voice] Agent error: {exc}", flush=True)
                continue

            if not response.strip():
                continue

            # --- TTS + playback ---
            try:
                self._tts.speak(response, device=self._output_device)
            except Exception as exc:
                print(f"[voice] TTS error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_voice_session(
    agent_session: "AgentSession",
    voice_config: object,
    on_event: "Callable[[object], None] | None" = None,
) -> VoiceSession:
    """Build a fully configured ``VoiceSession`` from config.

    Imports the factory functions from the other voice modules so callers
    only need to import this function.

    Args:
        agent_session: The target ``AgentSession`` from ``minion.py``.
        voice_config: A ``VoiceConfig`` instance from ``config.py``.
        on_event: Optional event callback for terminal output during agent turns.

    Returns:
        VoiceSession: Ready to call ``.run()`` on.
    """
    from .stt import build_stt  # noqa: PLC0415
    from .tts import build_tts  # noqa: PLC0415
    from .vad import build_vad_capture  # noqa: PLC0415

    audio_cfg = getattr(voice_config, "audio", None)
    output_device = getattr(audio_cfg, "output_device", None) if audio_cfg else None

    return VoiceSession(
        agent_session=agent_session,
        stt=build_stt(voice_config),
        tts=build_tts(voice_config),
        vad_capture=build_vad_capture(voice_config),
        on_event=on_event,
        output_device=output_device,
    )
