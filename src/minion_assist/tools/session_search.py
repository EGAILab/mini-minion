"""Session history search tool backed by PostgreSQL FTS.

Three modes:
  DISCOVER — full-text search across all sessions, returns match + context window + bookends
  SCROLL   — paginate within a session by anchor message ID
  BROWSE   — list recent sessions chronologically
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .base import Tool, ToolSchema

if TYPE_CHECKING:
    from ..session.db import SessionDB


class SessionSearchTool(Tool):
    """Search, scroll, or browse past conversation sessions stored in PostgreSQL.

    Scoped to a single owning agent (MEM-GAP-002): every DB call below
    passes ``self._agent_id``, so this tool can only ever see the
    conversation history that belongs to the agent it was built for. Agents
    are meant to have completely private memory from each other, even when
    they share one PostgreSQL database — one agent's `session_search` must
    never be able to enumerate or read another agent's sessions.
    """

    def __init__(self, db: "SessionDB", agent_id: str) -> None:
        self._db = db
        self._agent_id = agent_id

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="session_search",
            description=(
                "Search, scroll, or browse past conversation history stored in the database.\n"
                "DISCOVER: full-text search across ALL sessions — supports AND (default), OR, "
                'quoted phrase "exact match", minus -exclude, prefix word*.\n'
                "SCROLL: read messages around a specific message ID within one session.\n"
                "BROWSE: list recent sessions chronologically with titles and previews."
            ),
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["DISCOVER", "SCROLL", "BROWSE"],
                        "description": "Which mode to use",
                    },
                    "query": {
                        "type": "string",
                        "description": "DISCOVER only — keywords to search (FTS websearch syntax)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "SCROLL only — session UUID to read messages from",
                    },
                    "anchor_message_id": {
                        "type": "integer",
                        "description": "SCROLL only — center window on this message ID; 0 = end of session",
                    },
                    "window": {
                        "type": "integer",
                        "description": "SCROLL only — messages before/after the anchor (default 5, max 20)",
                    },
                },
                "required": ["mode"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        mode = str(kwargs.get("mode", "")).upper()
        if mode == "DISCOVER":
            return self._discover(str(kwargs.get("query", "")))
        if mode == "SCROLL":
            return self._scroll(
                str(kwargs.get("session_id", "")),
                int(kwargs.get("anchor_message_id", 0)),
                int(kwargs.get("window", 5)),
            )
        if mode == "BROWSE":
            return self._browse()
        return f"Unknown mode {mode!r}. Use DISCOVER, SCROLL, or BROWSE."

    # ------------------------------------------------------------------

    def _discover(self, query: str) -> str:
        if not query.strip():
            return "Error: 'query' is required for DISCOVER mode."
        try:
            matches = self._db.search_messages(query, self._agent_id, limit=15)
        except Exception as exc:
            return f"Search error: {exc}"

        if not matches:
            return f"No messages found matching {query!r}."

        # Deduplicate: keep best-ranked match per session
        seen: dict[str, dict] = {}
        for m in matches:
            sid = m["session_id"]
            if sid not in seen or m["rank"] > seen[sid]["rank"]:
                seen[sid] = m

        session_metas = self._db.get_sessions_by_ids(list(seen.keys()), self._agent_id)

        parts = [
            f"[Session search: {query!r} — {len(seen)} session(s) matched]\n"
        ]
        for m in seen.values():
            session_id = m["session_id"]
            meta = session_metas.get(session_id, {})
            title = meta.get("title") or "(untitled)"
            agent_id = meta.get("agent_id", "?")

            start_msgs, end_msgs = self._db.get_session_bookends(session_id, self._agent_id, n=2)
            context_msgs = self._db.get_messages_around(
                session_id, self._agent_id, m["id"], window=3
            )

            parts.append(f"## [{agent_id}] {session_id[:8]}… — {title}")
            snippet = m.get("snippet") or (m["content"] or "")[:200]
            parts.append(f"Match (rank={m['rank']:.3f}): {snippet}")
            parts.append(f"Message ID: {m['id']}")

            if start_msgs:
                parts.append("Start of session:")
                for msg in start_msgs:
                    parts.append(f"  [{msg['role']}] {(msg['content'] or '')[:120]}")

            if context_msgs:
                parts.append("Context (±3 messages):")
                for msg in context_msgs:
                    arrow = " ← match" if msg["id"] == m["id"] else ""
                    parts.append(
                        f"  [#{msg['id']}][{msg['role']}] "
                        f"{(msg['content'] or '')[:150]}{arrow}"
                    )

            if end_msgs:
                parts.append("End of session:")
                for msg in end_msgs:
                    parts.append(f"  [{msg['role']}] {(msg['content'] or '')[:120]}")

            parts.append(
                f"→ Read more: session_search SCROLL "
                f"session_id={session_id} anchor_message_id={m['id']}"
            )
            parts.append("")

        return "\n".join(parts)

    def _scroll(self, session_id: str, anchor_id: int, window: int) -> str:
        if not session_id:
            return "Error: 'session_id' is required for SCROLL mode."
        window = max(1, min(20, window))
        try:
            msgs = self._db.get_messages_around(
                session_id, self._agent_id, anchor_id, window=window
            )
        except Exception as exc:
            return f"Scroll error: {exc}"

        if not msgs:
            return f"No messages found in session {session_id!r} around #{anchor_id}."

        parts = [f"[Session {session_id} — {len(msgs)} messages around #{anchor_id}]"]
        for m in msgs:
            arrow = " ←" if m["id"] == anchor_id else ""
            tool = f" ({m['tool_name']})" if m["tool_name"] else ""
            content = (m["content"] or "")[:300]
            parts.append(f"[#{m['id']}][{m['role']}]{tool} {content}{arrow}")

        first_id = msgs[0]["id"]
        last_id = msgs[-1]["id"]
        parts.append(f"→ Scroll earlier: anchor_message_id={max(1, first_id - window)}")
        parts.append(f"→ Scroll later:   anchor_message_id={last_id + 1}")
        return "\n".join(parts)

    def _browse(self) -> str:
        try:
            sessions = self._db.list_sessions(self._agent_id, limit=20)
        except Exception as exc:
            return f"Browse error: {exc}"

        if not sessions:
            return "No sessions in the database yet."

        now = time.time()
        parts = [f"[Recent sessions — {len(sessions)} shown]"]
        for s in sessions:
            title = s["title"] or "(untitled)"
            preview = (s["first_message"] or "")[:100]
            age = now - (s["last_active"] or now)
            if age < 3600:
                age_str = f"{int(age / 60)}m ago"
            elif age < 86400:
                age_str = f"{int(age / 3600)}h ago"
            else:
                age_str = f"{int(age / 86400)}d ago"
            parts.append(
                f"  [{s['agent_id']}] {s['id'][:8]}… "
                f"{title} — {s['turn_count']} turns, {age_str}"
            )
            if preview:
                parts.append(f"    {preview}")
        return "\n".join(parts)
