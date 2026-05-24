"""Markdown-file-backed long-term memory store."""

from pathlib import Path


class LongTermMemory:
    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.md"

    def save(self, key: str, content: str) -> None:
        self._path(key).write_text(content, encoding="utf-8")

    def load(self, key: str) -> str | None:
        p = self._path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def search(self, query: str) -> list[tuple[str, str]]:
        """Return [(key, content)] for files containing query (case-insensitive)."""
        results = []
        q = query.lower()
        for p in sorted(self._dir.glob("*.md")):
            content = p.read_text(encoding="utf-8")
            if q in content.lower():
                results.append((p.stem, content))
        return results

    def list_keys(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.md"))]

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.exists():
            p.unlink()
            return True
        return False
