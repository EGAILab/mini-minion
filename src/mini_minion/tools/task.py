"""Task progress tools — read and update a structured task file.

These tools implement the "Ralph Loop" pattern for long-running tasks:
- At session start the agent calls read_task to check for active work and
  resume from where it left off.
- After completing each step the agent calls update_task to record progress.
- When the context window fills and the session restarts, the agent calls
  read_task again to orient itself — no conversation history needed.

Why JSON and not Markdown?
--------------------------
1. Individual steps can be updated without rewriting the whole file.
2. JSON is data; Markdown could be misread as instructions.
3. The schema stays consistent regardless of model creativity.

File location
-------------
``{workspace}/tasks/{agent_id}.json`` — stored outside the project workspace
so other file tools cannot accidentally modify it.

Talks to
--------
- ``base.py``      — extends :class:`Tool`, uses :class:`ToolSchema`.
- ``__init__.py``  — registered in :func:`default_registry` when
                     ``tasks_dir`` and ``agent_id`` are provided.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .base import Tool, ToolSchema

_VALID_STATUSES = frozenset({"pending", "in_progress", "done", "blocked"})

_STATUS_ICON = {"done": "✓", "in_progress": "→", "blocked": "✗", "pending": "○"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load(path: Path) -> dict:
    """Load the task JSON file, returning an empty dict if absent or corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: Path, data: dict) -> None:
    """Atomically write the task JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class ReadTaskTool(Tool):
    """Read the current task progress for this agent.

    Returns a compact summary of the active task so the agent can orient
    itself at the start of a new session or context window.

    Args:
        task_path: Full path to this agent's task JSON file.
    """

    def __init__(self, task_path: Path) -> None:
        self._path = task_path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_task",
            description=(
                "Read the current active task and step-by-step progress. "
                "Call this at the start of each session to check for ongoing work "
                "and resume from where you left off."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:  # noqa: ARG002
        data = _load(self._path)
        if not data:
            return (
                "No active task. To start one, call update_task with "
                "a 'goal' and a 'steps' list."
            )

        goal = data.get("goal", "(no goal set)")
        steps = data.get("steps", [])
        context = data.get("context", "")
        last_updated = data.get("last_updated", "unknown")

        lines = [
            "## Active Task",
            f"**Goal:** {goal}",
            f"**Last updated:** {last_updated}",
        ]
        if context:
            lines.append(f"**Context:** {context}")

        if steps:
            lines.append("\n**Steps:**")
            for step in steps:
                icon = _STATUS_ICON.get(step.get("status", "pending"), "?")
                notes = step.get("notes", "")
                note_str = f" — {notes}" if notes else ""
                lines.append(
                    f"  [{icon}] {step.get('id', '?')}. "
                    f"{step.get('description', '')}{note_str}"
                )
            next_steps = [
                s for s in steps if s.get("status") in ("pending", "in_progress")
            ]
            if next_steps:
                ns = next_steps[0]
                lines.append(
                    f"\n**Next:** Step {ns.get('id')} — {ns.get('description', '')}"
                )
            else:
                lines.append("\n**All steps complete.**")
        else:
            lines.append("No steps defined yet.")

        return "\n".join(lines)


class UpdateTaskTool(Tool):
    """Create or update the task progress file.

    - Start a new task: provide ``goal`` + ``steps``.
    - Mark a step done: provide ``step_id`` + ``status``, optionally ``notes``.
    - Save key context: provide ``context``.
    - Clear a finished task: provide ``clear=true``.

    Args:
        task_path: Full path to this agent's task JSON file.
    """

    def __init__(self, task_path: Path) -> None:
        self._path = task_path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="update_task",
            description=(
                "Create or update the active task. "
                "Use to start a new task (provide goal + steps), "
                "record step completion (provide step_id + status), "
                "save key context (provide context), "
                "or clear a finished task (provide clear=true). "
                "Call after every completed step so progress survives a restart."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": (
                            "High-level objective. Set once when starting a new task."
                        ),
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of step descriptions to initialise the task. "
                            "Only used when creating a new task alongside 'goal'."
                        ),
                    },
                    "step_id": {
                        "type": "integer",
                        "description": "1-indexed ID of the step to update.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "blocked"],
                        "description": "New status for the step.",
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Brief notes about what was done or found. "
                            "Stored with the step and visible in read_task."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Key context to preserve across sessions: file paths, "
                            "decisions, key findings. Replaces any previous context."
                        ),
                    },
                    "clear": {
                        "type": "boolean",
                        "description": "If true, delete the task file (task complete).",
                    },
                },
                "required": [],
            },
        )

    def execute(self, **kwargs: object) -> str:
        if kwargs.get("clear"):
            if self._path.exists():
                self._path.unlink()
            return "Task cleared."

        data = _load(self._path)

        # --- Initialise a new task ---
        goal = kwargs.get("goal")
        if goal:
            step_descs = kwargs.get("steps", [])
            if not isinstance(step_descs, list):
                step_descs = []
            data = {
                "goal": str(goal),
                "created_at": _now(),
                "steps": [
                    {
                        "id": i + 1,
                        "description": str(d),
                        "status": "pending",
                        "notes": "",
                    }
                    for i, d in enumerate(step_descs)
                ],
                "context": "",
                "last_updated": _now(),
            }
            _save(self._path, data)
            return (
                f"Task created: {goal!r} with {len(data['steps'])} step(s). "
                "Call update_task(step_id=1, status='in_progress') when ready to start."
            )

        if not data:
            return "No active task. Provide a 'goal' to create one."

        updated = False

        # --- Update a step ---
        step_id = kwargs.get("step_id")
        status = kwargs.get("status")
        notes = kwargs.get("notes")

        if step_id is not None:
            step_id = int(step_id)
            steps = data.get("steps", [])
            matched = [s for s in steps if s.get("id") == step_id]
            if not matched:
                ids = [s["id"] for s in steps]
                return f"Step {step_id} not found. Available IDs: {ids}"
            step = matched[0]
            if status:
                if status not in _VALID_STATUSES:
                    return f"Invalid status {status!r}. Use: {sorted(_VALID_STATUSES)}"
                step["status"] = status
                updated = True
            if notes is not None:
                step["notes"] = str(notes)
                updated = True

        # --- Update context ---
        ctx = kwargs.get("context")
        if ctx is not None:
            data["context"] = str(ctx)
            updated = True

        if not updated:
            return "Nothing updated. Provide step_id+status, notes, or context."

        data["last_updated"] = _now()
        _save(self._path, data)

        parts = []
        if step_id is not None and status:
            parts.append(f"step {step_id} → {status}")
        if notes is not None:
            parts.append("notes saved")
        if ctx is not None:
            parts.append("context updated")
        return "Updated: " + ", ".join(parts) + "."
