"""Audio waveform downsampling for the TUI's attachment preview (Phase 2).

Pure function, no Textual dependency — kept separate from
attachment_widgets.py so it can be unit-tested without spinning up a
Textual App, the same separation-of-concerns reasoning voice/audio.py
already follows for audio I/O versus voice/session.py's orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Number of bars Sparkline renders the waveform as. Arbitrary but generous
# enough to show real shape in a typical terminal-width chat pane without
# the widget growing unreasonably wide.
_DEFAULT_BUCKETS = 60


def build_waveform(path: Path, buckets: int = _DEFAULT_BUCKETS) -> list[float]:
    """Return a downsampled amplitude envelope for an audio file.

    Reads the full file (audio attachments are capped at 50 MB — see
    media.py's _MAX_AUDIO_BYTES — so this is always a bounded, one-off
    decode, not a streaming operation), converts to mono, and reduces it to
    ``buckets`` values by taking the peak absolute amplitude within each
    equal-sized chunk — peak (not mean/RMS) because a waveform preview's
    whole purpose is showing transients (where the loud parts are), which
    an averaging reduction would smooth away.

    Args:
        path: Path to a staged audio file (any format soundfile can decode
            — wav/flac/ogg/mp3, matching media.py's ALLOWED_AUDIO_TYPES).
        buckets: Number of output values.

    Returns:
        list[float]: ``buckets`` peak-amplitude values in [0.0, 1.0]. Empty
            list if the file can't be decoded (soundfile missing, corrupt
            file, etc.) — callers should treat that as "no preview
            available," never as a reason to fail whatever triggered this.
    """
    try:
        import soundfile as sf  # noqa: PLC0415 — optional dependency (tui extra)
    except ImportError:
        return []

    try:
        samples, _samplerate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return []

    if samples.ndim > 1:
        # Multi-channel — average to mono, same convention voice/audio.py's
        # play_audio() uses for stereo input.
        samples = samples.mean(axis=1)

    samples = np.abs(samples)
    total = len(samples)
    if total == 0:
        return [0.0] * buckets

    # Split into `buckets` equal-ish chunks (np.array_split handles a
    # length that doesn't divide evenly by giving the first few chunks one
    # extra sample rather than requiring an even split).
    chunks = np.array_split(samples, buckets)
    peaks = [float(chunk.max()) if chunk.size else 0.0 for chunk in chunks]

    # Normalize to [0, 1] so a quiet recording's waveform isn't a flat
    # near-invisible line — Sparkline's own color scale expects the data's
    # own min/max, but a visually meaningful *shape* still needs contrast
    # within the recording itself, which absolute amplitude alone won't
    # always provide for a quiet clip.
    peak_max = max(peaks) if peaks else 0.0
    if peak_max > 0:
        peaks = [p / peak_max for p in peaks]

    return peaks
