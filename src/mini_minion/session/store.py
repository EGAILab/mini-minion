"""JSON-backed session metadata store."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SessionInfo:
    agent_id: str
    created_at: str
    last_active: str
    turn_count: int


class SessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_or_create(self, agent_id: str) -> SessionInfo:
        data = self._load()
        if agent_id not in data:
            now = _now()
            data[agent_id] = asdict(SessionInfo(
                agent_id=agent_id,
                created_at=now,
                last_active=now,
                turn_count=0,
            ))
            self._save(data)
        return SessionInfo(**data[agent_id])

    def touch(self, agent_id: str, increment_turns: bool = False) -> SessionInfo:
        data = self._load()
        if agent_id not in data:
            return self.get_or_create(agent_id)
        data[agent_id]["last_active"] = _now()
        if increment_turns:
            data[agent_id]["turn_count"] += 1
        self._save(data)
        return SessionInfo(**data[agent_id])

    def list_sessions(self) -> list[SessionInfo]:
        return [SessionInfo(**v) for v in self._load().values()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
