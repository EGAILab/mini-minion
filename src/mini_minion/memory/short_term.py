"""JSONL-backed short-term conversation history per agent."""

import json
from pathlib import Path


class ShortTermMemory:
    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.jsonl"

    def load(self, key: str) -> list[dict]:
        p = self._path(key)
        if not p.exists():
            return []
        messages = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                messages.append(json.loads(line))
        return messages

    def save(self, key: str, messages: list[dict]) -> None:
        self._path(key).write_text(
            "\n".join(json.dumps(m) for m in messages),
            encoding="utf-8",
        )

    def append(self, key: str, message: dict) -> None:
        with self._path(key).open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

    def clear(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()
