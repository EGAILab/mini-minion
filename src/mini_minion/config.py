"""Loads configuration from config.toml and .env."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")

with open(_ROOT / "config.json", encoding="utf-8") as _f:
    _raw = json.load(_f)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    api: str


@dataclass(frozen=True)
class ModelConfig:
    id: str
    max_tokens: int


@dataclass(frozen=True)
class AgentModelConfig:
    provider: ProviderConfig
    model: ModelConfig


def _resolve_provider(provider_name: str, model_id: str) -> AgentModelConfig:
    provider_raw = _raw["models"]["providers"][provider_name]
    model_raw = next(m for m in provider_raw["models"] if m["id"] == model_id)
    api_key = provider_raw.get("apiKey") or os.environ.get(f"{provider_name.upper()}_API_KEY", "")
    return AgentModelConfig(
        provider=ProviderConfig(
            name=provider_name,
            base_url=provider_raw.get("baseUrl", ""),
            api_key=api_key,
            api=provider_raw["api"],
        ),
        model=ModelConfig(
            id=model_id,
            max_tokens=model_raw.get("maxTokens", 4096),
        ),
    )


def _resolve_all() -> dict[str, AgentModelConfig]:
    result: dict[str, AgentModelConfig] = {}
    for agent_id, agent_raw in _raw.get("agents", {}).items():
        provider_name, model_id = agent_raw["model"].split("/", 1)
        result[agent_id] = _resolve_provider(provider_name, model_id)
    return result


agents: dict[str, AgentModelConfig] = _resolve_all()

workspace: Path = Path(_raw.get("workspace", {}).get("path", "~/.mini-minion")).expanduser()
