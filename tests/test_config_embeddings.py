"""Tests for EmbeddingConfig resolution and validation (Stage One Phase 4, slice A).

Follows tests/test_config_memory.py's pattern: patch config._raw and call the
private resolver/validator functions directly, so no real config.json on
disk is needed.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from minion_assist.config import ConfigIssue, EmbeddingConfig, _resolve_embeddings, _validate

_PROVIDERS = {
    "lmstudio": {
        "api": "lmstudio",
        "baseUrl": "http://localhost:1234/v1",
        "models": [{"id": "qwen-9b", "contextWindow": 8192, "maxOutputTokens": 4096}],
    }
}

_VALID: dict = {
    "models": {"providers": _PROVIDERS},
    "agents": {"main": {"model": "lmstudio/qwen-9b"}},
}


def _paths(issues: list[ConfigIssue]) -> list[str]:
    return [i.path for i in issues]


# ---------------------------------------------------------------------------
# _resolve_embeddings
# ---------------------------------------------------------------------------

def test_resolve_embeddings_returns_none_when_section_absent():
    with patch("minion_assist.config._raw", {}):
        assert _resolve_embeddings() is None


def test_resolve_embeddings_returns_none_when_section_empty():
    raw = {"embeddings": {}}
    with patch("minion_assist.config._raw", raw):
        assert _resolve_embeddings() is None


def test_resolve_embeddings_builds_config_from_existing_provider():
    raw = {
        "models": {"providers": _PROVIDERS},
        "embeddings": {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": 768},
    }
    with patch("minion_assist.config._raw", raw):
        cfg = _resolve_embeddings()

    assert cfg is not None
    assert cfg.model == "nomic-embed-text"
    assert cfg.dimensions == 768
    assert cfg.provider.name == "lmstudio"
    assert cfg.provider.base_url == "http://localhost:1234/v1"
    assert cfg.provider.api == "lmstudio"


def test_resolve_embeddings_model_need_not_be_in_the_providers_chat_models_list():
    # The embedding model ("nomic-embed-text") is deliberately NOT one of
    # lmstudio's chat models ("qwen-9b") — embedding models are typically
    # served separately, so this must not raise or fail to resolve.
    raw = {
        "models": {"providers": _PROVIDERS},
        "embeddings": {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": 768},
    }
    with patch("minion_assist.config._raw", raw):
        cfg = _resolve_embeddings()
    assert cfg.model == "nomic-embed-text"


def test_resolve_embeddings_resolves_api_key_from_environment(monkeypatch):
    raw = {
        "models": {"providers": _PROVIDERS},
        "embeddings": {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": 768},
    }
    monkeypatch.setenv("LMSTUDIO_API_KEY", "test-key-123")
    with patch("minion_assist.config._raw", raw):
        cfg = _resolve_embeddings()
    assert cfg.provider.api_key == "test-key-123"


def test_embedding_config_is_frozen():
    from minion_assist.config import ProviderConfig

    cfg = EmbeddingConfig(
        provider=ProviderConfig(name="lmstudio", base_url="", api_key="", api="lmstudio"),
        model="nomic-embed-text",
        dimensions=768,
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.dimensions = 1536  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _validate — embeddings section
# ---------------------------------------------------------------------------

def test_validate_accepts_absent_embeddings_section():
    assert _validate(copy.deepcopy(_VALID)) == []


def test_validate_accepts_well_formed_embeddings_section():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": 768}
    assert _validate(raw) == []


def test_validate_rejects_non_object_embeddings_section():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = "not an object"
    issues = _validate(raw)
    assert "embeddings" in _paths(issues)


def test_validate_rejects_missing_provider():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"model": "nomic-embed-text", "dimensions": 768}
    issues = _validate(raw)
    assert "embeddings.provider" in _paths(issues)


def test_validate_rejects_unknown_provider():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudi", "model": "nomic-embed-text", "dimensions": 768}
    issues = _validate(raw)
    matching = [i for i in issues if i.path == "embeddings.provider"]
    assert len(matching) == 1
    assert "lmstudio" in matching[0].message  # close-match hint


def test_validate_rejects_missing_model():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudio", "dimensions": 768}
    issues = _validate(raw)
    assert "embeddings.model" in _paths(issues)


def test_validate_rejects_missing_dimensions():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudio", "model": "nomic-embed-text"}
    issues = _validate(raw)
    assert "embeddings.dimensions" in _paths(issues)


def test_validate_rejects_non_positive_dimensions():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": 0}
    issues = _validate(raw)
    assert "embeddings.dimensions" in _paths(issues)


def test_validate_rejects_non_integer_dimensions():
    raw = copy.deepcopy(_VALID)
    raw["embeddings"] = {"provider": "lmstudio", "model": "nomic-embed-text", "dimensions": "768"}
    issues = _validate(raw)
    assert "embeddings.dimensions" in _paths(issues)
