"""Public API for the memory subsystem.

Re-exports the key memory classes so callers can import from
``minion_assist.memory`` directly instead of from the sub-modules.

Sub-modules
-----------
- ``short_term`` — :class:`ShortTermMemory`: JSONL-backed conversation history,
  one file per agent. Holds the transcript of the current and past sessions.
- ``files``      — :class:`MemoryFileRepository`: the on-disk store behind
  :class:`MemoryService`, targeting the merged per-agent
  ``workspaces/{agent_id}/memory/`` layout (Stage One Phase 1).
- ``service``    — :class:`MemoryService`: the orchestration facade
  ``AgentSession``/tools depend on for explicit notes, quarantined
  (unreviewed) notes, daily logs, search, and bounded exact reads.
- ``long_term``  — :class:`LongTermMemory`: the legacy Markdown-backed note
  store, superseded by :class:`MemoryService` (which nothing in the runtime
  wiring constructs anymore) but kept — and still tested — as the format
  ``memory/migration.py``'s Phase 0 tooling reads *from*.
"""

from .files import MemoryFileRepository
from .long_term import LongTermMemory
from .service import MemoryService
from .short_term import ShortTermMemory

__all__ = ["LongTermMemory", "MemoryFileRepository", "MemoryService", "ShortTermMemory"]
