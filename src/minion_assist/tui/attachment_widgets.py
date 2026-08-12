"""Inline attachment rendering for the chat log (TUI Phase 2).

Two widgets, one per MediaAttachment kind:
- ImageAttachmentView — renders the image inline via textual-image (Kitty
  Terminal Graphics Protocol or Sixel, falling back to Unicode half-block
  rendering on terminals that support neither).
- AudioAttachmentView — a waveform preview (tui/waveform.py) plus a Play
  button. Audio is never sent to the LLM (see media.py's module docstring)
  — this is purely a local preview/playback affordance.

Both are mounted into the chat log by app.py's _log_user() right after the
message's text, for every attachment on that turn.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Sparkline, Static
from textual_image.widget import Image

from ..media import MediaAttachment, describe_attachment
from .waveform import build_waveform


class PlayAudioRequested(Message):
    """Posted by AudioAttachmentView when its Play button is pressed.

    Bubbles up to MinionApp, which owns the actual playback worker — kept
    as a message rather than the widget calling self.app directly, so this
    widget doesn't need to know anything about the App class it lives in.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()


class ImageAttachmentView(Vertical):
    """Inline image render for one staged image MediaAttachment."""

    DEFAULT_CSS = """
    ImageAttachmentView {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }
    """

    def __init__(self, attachment: MediaAttachment) -> None:
        super().__init__()
        self._attachment = attachment

    def compose(self) -> ComposeResult:
        yield Static(Text(describe_attachment(self._attachment), style="dim"))
        yield Image(str(self._attachment.path))


class AudioAttachmentView(Vertical):
    """Waveform preview + Play button for one staged audio MediaAttachment."""

    DEFAULT_CSS = """
    AudioAttachmentView {
        height: auto;
        border: round $accent;
        padding: 0 1;
        margin: 1 0;
    }
    AudioAttachmentView Sparkline {
        height: 3;
    }
    """

    def __init__(self, attachment: MediaAttachment) -> None:
        super().__init__()
        self._attachment = attachment

    def compose(self) -> ComposeResult:
        yield Static(Text(describe_attachment(self._attachment)))
        waveform = build_waveform(self._attachment.path)
        if waveform:
            yield Sparkline(waveform, summary_function=max)
        else:
            # soundfile missing, or the file couldn't be decoded for a
            # preview — playback may still work (sounddevice/PortAudio has
            # its own, separate decode path), so still offer the button.
            yield Static(Text("(no waveform preview available)", style="dim"))
        yield Button("Play", id="play")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PlayAudioRequested(self._attachment.path))
