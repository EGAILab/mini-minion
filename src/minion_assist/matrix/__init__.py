"""Matrix channel package for minion-assist.

Provides a bidirectional Matrix integration that runs a background listener
alongside the CLI REPL, sharing the same AgentSession objects.

Public entry point:
    MatrixChannel — start(sessions) / stop() lifecycle manager.
"""

from .channel import MatrixChannel

# __all__ limits what `from minion_assist.matrix import *` exposes.
# Consumers only need MatrixChannel; all internal modules stay private.
__all__ = ["MatrixChannel"]
