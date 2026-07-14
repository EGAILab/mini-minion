"""VoiceSession — orchestrates the full speech-to-speech pipeline.

Threading model
---------------
Three concurrent threads are active during a voice turn:

- **VAD capture thread** (daemon): ``VadCapture._capture_loop`` reads the mic
  and pushes completed utterances to ``_utterance_queue``.
- **TTS synth thread** (daemon): ``_speak_streaming`` starts a producer that
  fills a bounded ``Queue(maxsize=2)`` with audio chunks.  Streaming adapters
  (e.g. KokoroTTS, ``supports_streaming=True``) pass the full text to
  ``synthesise_stream()`` and yield one chunk per pipeline segment.  Batch
  adapters fall back to sentence-splitting: each sentence is synthesised
  individually so the first sentence plays while the next is being generated.
- **Audio output** (``sd.OutputStream``): a single stream kept open across
  all sentences so there is no gap between them.  Audio is written in 50 ms
  chunks; ``stream.abort()`` discards buffered audio on barge-in.

The main voice loop drives the STT → agent → TTS pipeline sequentially.

Barge-in (interruption)
-----------------------
While TTS is playing, the VAD capture thread continues to run.  The moment
the user starts speaking, ``VadCapture.speech_started`` (a
``threading.Event``) is set.  ``_speak_streaming`` checks
``speech_started.is_set()`` before every 50 ms audio write chunk.  On
detection it calls ``stream.abort()`` to discard buffered audio immediately,
signals the synth thread to exit via ``cancelled``, then returns — the
completed utterance lands in ``_utterance_queue`` and is picked up on the
next ``_loop`` iteration.

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
import re
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

# Sentence splitter: break after . ! ? when followed by whitespace.
# Keeps abbreviations inside a sentence (e.g. "Dr. Smith") mostly intact
# because they're unlikely to be followed by a capital letter.
_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# Markdown-stripping patterns applied before TTS so the model reads clean prose
# instead of speaking punctuation like "asterisk asterisk".
_MD_HORIZ   = re.compile(r'^\s*[-*_]{3,}\s*$', re.MULTILINE)
_MD_HEADING = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_LINK    = re.compile(r'\[([^\]]+)\]\([^)]*\)')
_MD_B_I_S   = re.compile(r'\*{3}(.+?)\*{3}', re.DOTALL)   # ***bold italic***
_MD_B_I_U   = re.compile(r'_{3}(.+?)_{3}', re.DOTALL)      # ___bold italic___
_MD_BOLD_S  = re.compile(r'\*{2}(.+?)\*{2}', re.DOTALL)    # **bold**
_MD_BOLD_U  = re.compile(r'_{2}(.+?)_{2}', re.DOTALL)      # __bold__
_MD_ITAL_S  = re.compile(r'\*(.+?)\*')                      # *italic*
_MD_ITAL_U  = re.compile(r'_(.+?)_')                        # _italic_
_MD_CODE    = re.compile(r'`(.+?)`')                         # `code`
_MD_BULLET  = re.compile(r'^\s*[-*]\s+', re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation; return non-empty parts.

    Args:
        text: Any prose string.

    Returns:
        list[str]: One or more non-empty sentence strings.  Returns the whole
            text as a single element if no sentence boundary is found.
    """
    parts = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _strip_markdown(text: str) -> str:
    """Remove common Markdown syntax so TTS reads clean prose.

    Strips bold, italic, headings, inline code, links, bullet markers, and
    horizontal rules.  Numbered list prefixes (e.g. ``"1. "`` ) are left
    intact because they read naturally aloud.  Em dashes (``—``) are also
    left intact as TTS handles them correctly.

    Args:
        text: Raw agent response, potentially containing Markdown formatting.

    Returns:
        str: Plain-text version suitable for speech synthesis.
    """
    text = _MD_HORIZ.sub('', text)
    text = _MD_HEADING.sub('', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_B_I_S.sub(r'\1', text)
    text = _MD_B_I_U.sub(r'\1', text)
    text = _MD_BOLD_S.sub(r'\1', text)
    text = _MD_BOLD_U.sub(r'\1', text)
    text = _MD_ITAL_S.sub(r'\1', text)
    text = _MD_ITAL_U.sub(r'\1', text)
    text = _MD_CODE.sub(r'\1', text)
    text = _MD_BULLET.sub('', text)
    return text.strip()


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
        language: BCP-47 tag injected as a prompt prefix on every voice turn
            so the agent replies in the TTS engine's supported language (e.g.
            ``"en"`` for Kokoro, which is English-only).  Pass ``""`` to
            disable the injection.
    """

    def __init__(
        self,
        agent_session: "AgentSession",
        stt: "STTAdapter",
        tts: "TTSAdapter",
        vad_capture: "VadCapture",
        on_event: "Callable[[object], None] | None" = None,
        output_device: "str | int | None" = None,
        language: str = "en",
    ) -> None:
        self._agent_session = agent_session
        self._stt = stt
        self._tts = tts
        self._vad_capture = vad_capture
        self._on_event = on_event
        self._output_device = output_device
        self._language = language
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
            # Poll with a short timeout so Ctrl+C (SIGINT) can be delivered
            # on Windows — queue.get() with no timeout blocks in C and swallows
            # KeyboardInterrupt until the next utterance arrives.
            try:
                utterance = self._utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue
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
            # Prepend a language constraint so the LLM replies in the TTS
            # engine's supported language (Kokoro is English-only).
            send_text = (
                f"[Voice mode: reply in {self._language} only.]\n{text}"
                if self._language else text
            )
            response: str = ""
            try:
                response = self._agent_session.send(
                    send_text,
                    on_event=self._on_event,
                ) or ""
            except KeyboardInterrupt:
                raise  # Let the caller handle Ctrl+C.
            except Exception as exc:
                print(f"[voice] Agent error: {exc}", flush=True)
                continue

            if not response.strip():
                continue

            # --- Streaming TTS + playback ---
            try:
                self._speak_streaming(response)
            except Exception as exc:
                print(f"[voice] TTS error: {exc}", flush=True)


    def _speak_streaming(self, text: str) -> None:
        """Synthesise and play *text* with smooth continuous output.

        A single ``sd.OutputStream`` stays open across all audio chunks so
        there is no audio-stream restart gap.  Audio is written in 50 ms
        chunks; between chunks ``speech_started`` is polled so barge-in fires
        the instant the user starts talking.

        Two synthesis paths feed the same ``audio_q → OutputStream`` pipeline:

        - **Streaming** (``tts.supports_streaming = True``, e.g. KokoroTTS):
          ``synthesise_stream(full_text)`` is called once; each chunk it yields
          is queued immediately, giving near-zero first-audio latency.
        - **Sentence-split** (default): the text is split on ``. ! ?`` boundaries
          and ``synthesise()`` is called per sentence in a background thread so
          the first sentence plays while the next is synthesising.

        Falls back to the blocking ``tts.speak()`` path when sounddevice is not
        installed.

        Args:
            text: The agent's full text response.
        """
        if not text.strip():
            return
        text = _strip_markdown(text)
        if not text:
            return

        cancelled = threading.Event()
        audio_q: "queue.Queue[tuple[np.ndarray, int] | None]" = queue.Queue(maxsize=2)

        def _synth_worker() -> None:
            # Kokoro and other streaming-capable adapters produce chunks naturally;
            # batch adapters fall back to sentence-splitting.
            if getattr(self._tts, "supports_streaming", False):
                try:
                    for chunk in self._tts.synthesise_stream(text):
                        if cancelled.is_set() or not self._running:
                            break
                        audio_q.put(chunk)
                except Exception as exc:
                    print(f"[voice] TTS error: {exc}", flush=True)
            else:
                for sent in _split_sentences(text):
                    if cancelled.is_set() or not self._running:
                        break
                    try:
                        audio_q.put(self._tts.synthesise(sent))
                    except Exception as exc:
                        print(f"[voice] TTS error: {exc}", flush=True)
            audio_q.put(None)

        threading.Thread(target=_synth_worker, daemon=True, name="tts-synth").start()

        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError:
            self._tts.speak(text, device=self._output_device)
            return

        def _bargin() -> bool:
            """True if the user has started speaking or the voice loop is stopping."""
            return self._vad_capture.speech_started.is_set() or not self._running

        # Fetch first sentence to learn the sample rate before opening the stream.
        try:
            first = audio_q.get(timeout=60)
        except queue.Empty:
            return
        if first is None:
            return
        samples_0, sr = first

        # 50 ms of audio at this sample rate.
        CHUNK = max(1, sr // 20)

        # Single OutputStream kept open for the whole response — no restart gap.
        with sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=self._output_device,
        ) as stream:

            def _write_sentence(arr: np.ndarray) -> bool:
                """Write *arr* in CHUNK-sized blocks; return False on barge-in."""
                data = arr.astype(np.float32)
                for i in range(0, len(data), CHUNK):
                    if _bargin():
                        cancelled.set()
                        stream.abort()
                        return False
                    stream.write(data[i : i + CHUNK])
                return True

            if not _write_sentence(samples_0):
                return

            while self._running:
                if _bargin():
                    cancelled.set()
                    stream.abort()
                    return

                try:
                    item = audio_q.get(timeout=60)
                except queue.Empty:
                    break
                if item is None:
                    break

                samples, _ = item
                if not _write_sentence(samples):
                    return

            # Drain: wait for the output buffer to finish playing.
            drain = getattr(stream, "latency", 0)
            end = time.monotonic() + drain
            while time.monotonic() < end and not _bargin():
                time.sleep(0.02)
            if _bargin():
                cancelled.set()
                stream.abort()


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
    language = getattr(voice_config, "language", "en") or ""

    return VoiceSession(
        agent_session=agent_session,
        stt=build_stt(voice_config),
        tts=build_tts(voice_config),
        vad_capture=build_vad_capture(voice_config),
        on_event=on_event,
        output_device=output_device,
        language=language,
    )
