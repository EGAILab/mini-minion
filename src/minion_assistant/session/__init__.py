"""Public API for the session subsystem.

Re-exports :class:`SessionInfo` and :class:`SessionStore` so callers can import
from ``minion_assistant.session`` directly.

Sub-modules
-----------
- ``store`` — :class:`SessionStore` (the store) and :class:`SessionInfo`
  (the data model). Tracks per-agent metadata: creation time, last activity,
  and total turn count, persisted in a single ``sessions.json`` file.
"""

from .store import SessionInfo, SessionStore

__all__ = ["SessionInfo", "SessionStore"]
