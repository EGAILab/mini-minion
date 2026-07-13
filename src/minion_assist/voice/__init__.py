"""Voice chat subsystem for minion-assist.

Provides a fully local speech-to-speech pipeline:
  Mic → Silero VAD → STT (Parakeet / Whisper) → agent → TTS (Qwen3 / Kokoro) → Speaker

All ML dependencies are imported lazily inside each module so this package can
be imported even when the 'voice' extra is not installed.  Runtime errors are
raised with helpful install instructions rather than import-time crashes.

Enable voice mode by running:  minion-assist --voice
Or toggle mid-session with:    /voice

Configure under the 'voice' key in config.json.  See VoiceConfig in config.py.
"""
