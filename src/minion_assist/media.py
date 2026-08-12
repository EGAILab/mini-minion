"""File-backed media attachment ingestion.

Stages user-provided files into a workspace attachment store so session JSONL
never contains large base64 blobs. Base64 is materialized on-demand by
providers just before the API request.

Storage layout:
    {workspace}/attachments/YYYY-MM-DD/<sha256-prefix>-<safe-name>

Why stage to disk instead of holding bytes in memory?
------------------------------------------------------
- Session JSONL history survives process restarts, so paths survive too.
- Multiple turns can reference the same staged file without re-reading bytes.
- The attachment store doubles as a local audit trail of what was sent.

Security:
    - Rejects paths containing dangerous segments (.env, .git, etc.)
    - Sniffs file bytes to verify image/audio MIME type (catches renamed ZIPs etc.)
    - Enforces size limits before staging

Audio (TUI Phase 2)
--------------------
Audio attachments are staged and validated the same way images are, and can
be locally previewed (waveform) and played back in the TUI (tui/waveform.py,
tui/attachment_widgets.py) — but unlike images, they are never sent to an
LLM provider as multimodal content. No provider in this codebase understands
an audio input block; messages.make_user_content() folds an audio
attachment's description into the prompt's *text* instead (filename,
duration), the same way it would describe any other file reference.
"""
from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .messages import ALLOWED_IMAGE_TYPES

# Max image file size in bytes (15 MB).
# Anthropic's vision limit is 20 MB; OpenAI's effective limit via base64 is ~20 MB
# after encoding overhead.  15 MB gives a comfortable buffer below both limits.
_MAX_IMAGE_BYTES = 15 * 1024 * 1024

# Max audio file size in bytes (50 MB) — deliberately more generous than
# images: audio is never base64-encoded into a provider request (see the
# module docstring), so there is no wire-size budget to protect. This cap
# exists purely to stop an accidental multi-hour recording from being
# staged and decoded for a waveform preview.
_MAX_AUDIO_BYTES = 50 * 1024 * 1024

# MIME types accepted for audio attachments. Both the canonical and the
# legacy "x-" variants are listed because Python's mimetypes module (used as
# the extension-based fallback below) returns the "x-" form for wav/flac on
# most platforms, while byte-sniffing (_sniff_audio_mime) returns the
# canonical form — either can end up being the value actually checked.
ALLOWED_AUDIO_TYPES: frozenset[str] = frozenset({
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac", "audio/ogg", "audio/vorbis",
})

# Path segments that indicate dangerous or secret files.
# If any component of the resolved path matches one of these, we reject it.
_REJECT_SEGMENTS = frozenset({
    ".env", ".git", ".venv", "node_modules", "__pycache__",
    ".aws", ".ssh", ".gnupg", ".config",
})



@dataclass
class MediaAttachment:
    """Metadata for one staged media file.

    The actual bytes live at `path` in the attachment store. Providers
    read from this path when building API requests (via materialize_image_data).

    Attributes:
        id:          Short unique ID (first 8 chars of sha256 of file content).
        kind:        "image" or "audio".
        path:        Absolute path to the staged file in the attachment store.
        media_type:  MIME type string, e.g. "image/png" or "audio/wav".
        size_bytes:  Original file size in bytes.
        source_name: Original filename, shown to the user in the REPL/TUI.
        duration_seconds: Audio duration, probed via soundfile at staging
            time. Always None for images, and None for audio when soundfile
            isn't installed or duration couldn't be determined — never a
            reason to fail staging (informational metadata only).
    """
    id: str
    kind: str
    path: Path
    media_type: str
    size_bytes: int
    source_name: str
    duration_seconds: float | None = None


def safe_filename(name: str) -> str:
    """Convert a filename to a path-safe ASCII string.

    Replaces anything that isn't alphanumeric, dot, dash, or underscore
    with an underscore, then caps at 80 characters to avoid filesystem limits.
    Public (no longer module-private) because matrix/handler.py also needs
    it: an inbound image's caption/body — arbitrary, sender-controlled text
    that can contain characters illegal in a Windows path (``?``, ``:``,
    etc.) — is used as a display/temp-file name before ``stage_attachment``
    ever sees it, so it must go through the same sanitizer.
    """
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return safe[:80]


def _sniff_image_mime(path: Path) -> str | None:
    """Sniff the actual image format from the first bytes of the file.

    Returns the MIME type string or None if the file is not a recognized image.
    This catches mislabeled files (e.g. a ZIP renamed to .png).

    Reads only the first 12 bytes — fast and avoids loading large images.
    Does not use the deprecated imghdr module (removed in Python 3.13).
    """
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return None

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # WEBP: starts with "RIFF" at bytes 0-3, then "WEBP" at bytes 8-11.
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sniff_audio_mime(path: Path) -> str | None:
    """Sniff the actual audio format from the first bytes of the file.

    Same reasoning as _sniff_image_mime: don't trust the extension. Reads
    only the first 12 bytes.
    """
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return None

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio/wav"
    if header[:4] == b"fLaC":
        return "audio/flac"
    if header[:4] == b"OggS":
        return "audio/ogg"
    # MP3: either an ID3v2 tag prefix, or a raw MPEG frame sync (the first
    # 11 bits of a frame header are always set).
    if header[:3] == b"ID3":
        return "audio/mpeg"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return None


def _probe_audio_duration(path: Path) -> float | None:
    """Best-effort audio duration in seconds. None if it can't be determined.

    soundfile is an optional dependency (the tui extra) — staging an audio
    file must still succeed without it, just without a duration to display.
    """
    try:
        import soundfile as sf  # noqa: PLC0415 — optional dependency
    except ImportError:
        return None
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def _reject_path(source: Path) -> str | None:
    """Return an error message if source should be rejected, else None.

    Checks against known dangerous path segments and filename patterns.
    Called after resolve() so symlinks cannot escape the check.
    """
    resolved = source.resolve()
    # Check every component of the absolute path, not just the filename.
    for part in resolved.parts:
        if part.lower() in _REJECT_SEGMENTS:
            return f"Rejected path: '{part}' segment is not allowed."
    # Catch dotfiles that look like secrets even without a dangerous parent dir.
    name = resolved.name.lower()
    if name.startswith(".env") or name in (".gitconfig", ".netrc"):
        return f"Rejected: '{resolved.name}' looks like a secrets file."
    return None


def stage_attachment(source: Path, media_dir: Path) -> MediaAttachment:
    """Validate and copy a file into the attachment store.

    Args:
        source:    Path provided by the user (may be relative; resolved internally).
        media_dir: Base attachment directory ({workspace}/attachments/).

    Returns:
        MediaAttachment with metadata pointing to the staged copy.

    Raises:
        ValueError: If the file is invalid (wrong type, too large, dangerous path).
        FileNotFoundError: If the source file does not exist.
    """
    source = source.resolve()

    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    # Security check — reject paths containing dangerous segments.
    err = _reject_path(source)
    if err:
        raise ValueError(err)

    size = source.stat().st_size

    # Determine MIME type:
    # 1. Sniff bytes (most reliable — catches extension spoofing). Image and
    #    audio signatures never overlap, so trying both in sequence is safe.
    # 2. Fall back to extension guess.
    sniffed_mime = _sniff_image_mime(source) or _sniff_audio_mime(source)
    ext_mime, _ = mimetypes.guess_type(str(source))

    mime = sniffed_mime or ext_mime or ""
    duration_seconds: float | None = None

    if mime in ALLOWED_IMAGE_TYPES:
        kind = "image"
        if size > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image too large: {size:,} bytes (max {_MAX_IMAGE_BYTES:,} bytes). "
                "Please resize the image before attaching."
            )
        if sniffed_mime is None:
            # Extension matches but bytes don't look like an image.
            # This catches .png files that are actually ZIP archives, etc.
            raise ValueError(
                f"File '{source.name}' does not appear to be a valid image "
                f"(extension suggests {ext_mime}, but bytes don't match)."
            )
    elif mime in ALLOWED_AUDIO_TYPES:
        kind = "audio"
        if size > _MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio file too large: {size:,} bytes (max {_MAX_AUDIO_BYTES:,} bytes)."
            )
        if sniffed_mime is None:
            raise ValueError(
                f"File '{source.name}' does not appear to be a valid audio file "
                f"(extension suggests {ext_mime}, but bytes don't match)."
            )
        duration_seconds = _probe_audio_duration(source)
    else:
        raise ValueError(
            f"Unsupported file type '{mime or 'unknown'}' for '{source.name}'. "
            f"Supported images: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}. "
            f"Supported audio: {', '.join(sorted(ALLOWED_AUDIO_TYPES))}."
        )

    # Compute sha256 of file content to generate a stable, collision-resistant ID.
    # Using content hash means the same file staged twice lands at the same path.
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    file_id = sha[:8]

    # Destination layout: {media_dir}/YYYY-MM-DD/<id>-<safe-name>
    # The date subdirectory keeps the store organized for manual inspection.
    today = date.today().isoformat()
    dest_dir = media_dir / today
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"{file_id}-{safe_filename(source.name)}"
    dest = dest_dir / dest_name

    # Skip the copy if the content-hashed destination already exists.
    # This is safe because the destination name includes the sha256 prefix.
    if not dest.exists():
        shutil.copy2(str(source), str(dest))

    return MediaAttachment(
        id=file_id,
        kind=kind,
        path=dest,
        media_type=mime,
        size_bytes=size,
        source_name=source.name,
        duration_seconds=duration_seconds,
    )


def describe_attachment(att: MediaAttachment) -> str:
    """Return a short human-readable description for REPL/TUI display.

    Example output: "screenshot.png (image/png, 142 KB)"
             or:     "voice-memo.wav (audio/wav, 812 KB, 12.3s)"
    """
    size_kb = att.size_bytes / 1024
    if att.duration_seconds is not None:
        return (
            f"{att.source_name} ({att.media_type}, {size_kb:.0f} KB, "
            f"{att.duration_seconds:.1f}s)"
        )
    return f"{att.source_name} ({att.media_type}, {size_kb:.0f} KB)"
