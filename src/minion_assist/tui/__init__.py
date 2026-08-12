"""Textual-based full-screen terminal UI — an opt-in alternative to the plain REPL.

Started via ``minion-assist --tui`` (see ``minion.py``). Requires the ``tui``
extra (``uv sync --extra tui``); importing this package without ``textual``
installed raises a clear ``ImportError`` at the point of use, the same
guarded-import convention ``matrix/`` uses for ``matrix-nio``.
"""
