"""CLI input layer — wraps prompt_toolkit for interactive prompt reading.

Provides Up/Down arrow key history navigation in the REPL, file-backed
history persistence, and safe fallback to plain input() when not in a TTY.
"""

from __future__ import annotations

import sys
from pathlib import Path

# prompt_toolkit is listed in pyproject.toml dependencies but we guard the
# import so that if something goes wrong at install time, minion-assistant can
# still fall back to plain input() rather than crashing at startup.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    _PROMPT_TOOLKIT_AVAILABLE = False


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate characters with the replacement character U+FFFD.

    Windows terminals can produce lone surrogate code points when handling
    certain input. Python's str allows lone surrogates internally but they
    cause encoding errors when written to UTF-8 files (which FileHistory does).
    Re-encoding through UTF-8 with 'replace' strips them safely.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


class SafeFileHistory(FileHistory):  # type: ignore[misc]
    """FileHistory subclass that sanitizes lone surrogates before writing.

    prompt_toolkit's FileHistory.store_string() writes each history entry to
    a UTF-8 text file. On Windows, arrow-key or paste sequences can contain
    lone surrogate characters that crash the write. This subclass sanitizes
    the string first.
    """

    def store_string(self, string: str) -> None:
        """Sanitize surrogates then delegate to the parent implementation."""
        super().store_string(_sanitize_surrogates(string))


class PromptReader:
    """Reads user prompts with Up/Down history navigation via prompt_toolkit.

    Creates the history file and its parent directory on construction.
    Falls back to plain input() when:
      - prompt_toolkit could not be imported
      - stdin or stdout are not TTYs (e.g. piped input in scripts/tests)

    This fallback means minion-assistant still works non-interactively even though
    prompt_toolkit is listed as a dependency.

    Args:
        history_path: Where to persist submitted prompts across restarts.
                      The parent directory is created if it does not exist.
    """

    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self._session: object | None = None
        if _PROMPT_TOOLKIT_AVAILABLE:
            # PromptSession keeps the SafeFileHistory open for the lifetime of
            # the process so every submitted prompt is appended automatically.
            # Guard construction in a try/except because prompt_toolkit's Win32
            # output layer raises NoConsoleScreenBufferError when stdout is not
            # a real Windows console (e.g. in Cygwin/mintty, CI, or pytest).
            # Falling back to _session=None triggers the plain input() path.
            try:
                self._session = PromptSession(
                    history=SafeFileHistory(str(history_path)),
                    enable_open_in_editor=False,  # disable Ctrl+X Ctrl+E — we don't need it
                    multiline=False,              # Enter always submits (no Shift+Enter newline)
                )
            except Exception:
                # Any failure (no console, Cygwin, bad env) → plain input() fallback.
                self._session = None

    def read(self) -> str:
        """Read one prompt from the user.

        Returns the raw input string (not stripped). The caller (minion.py)
        applies .strip() so the behaviour matches the old input() call.

        Uses prompt_toolkit when available and running interactively; falls
        back to plain input() otherwise.
        """
        if self._session is None or not sys.stdin.isatty() or not sys.stdout.isatty():
            # Non-interactive path — plain input() as before.
            return input("\nYou: ")

        # patch_stdout() ensures that any print() calls happening concurrently
        # (e.g. from background memory extraction threads) do not corrupt the
        # active prompt line by overwriting it mid-draw.
        with patch_stdout():
            return self._session.prompt(HTML("<b fg='ansiblue'>You:</b> "))
