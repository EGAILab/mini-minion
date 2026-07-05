"""HeartbeatRespondTool — lets the agent send a proactive notification.

During a heartbeat turn the agent is given this tool (injected as an extra_tool
for that turn only).  When the agent has something worth reporting it calls
``heartbeat_respond`` instead of just replying with prose.

The tool captures the message in a shared :class:`HeartbeatResponseCapture`
mutable container so the :class:`HeartbeatScheduler` can retrieve it after the
turn ends and route it to the configured notification room or terminal.

Design note: the capture object is created fresh per heartbeat invocation so
there is no cross-turn state leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Tool, ToolSchema


@dataclass
class HeartbeatResponseCapture:
    """Shared mutable container between HeartbeatRespondTool and HeartbeatScheduler.

    The tool writes to ``messages``; the scheduler reads from it after the turn.

    Attributes:
        messages: List of notification strings the agent emitted this turn.
    """
    messages: list[str] = field(default_factory=list)


class HeartbeatRespondTool(Tool):
    """Tool that lets the agent post a proactive heartbeat notification.

    Injected as an extra_tool for heartbeat turns only — never permanently
    registered in the global ToolRegistry.

    Args:
        capture: The shared capture object to write messages into.
    """

    def __init__(self, capture: HeartbeatResponseCapture) -> None:
        self._capture = capture

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="heartbeat_respond",
            description=(
                "Send a proactive notification during a heartbeat turn. "
                "Call this when you have something worth reporting to the user — "
                "an urgent email, an upcoming calendar event, or anything that "
                "warrants interrupting quiet time. Do NOT call it for routine checks "
                "where nothing new was found; just reply HEARTBEAT_OK instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The notification message to send to the user.",
                    },
                },
                "required": ["message"],
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        """Record the notification message for the scheduler to deliver.

        Args:
            message (str): Notification text.

        Returns:
            str: Confirmation that the notification was queued.
        """
        message = str(kwargs.get("message", "")).strip()
        if not message:
            return "[heartbeat_respond] Empty message — nothing recorded."
        self._capture.messages.append(message)
        return f"[heartbeat_respond] Notification queued ({len(self._capture.messages)} total)."
