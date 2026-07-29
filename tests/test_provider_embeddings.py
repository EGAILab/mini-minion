"""Tests for providers/embeddings.py: EmbeddingProvider (Stage One Phase 4, slice A).

The OpenAI SDK client itself makes no network calls at construction time, so
these tests build a real EmbeddingProvider and then replace its ._client
with a mock — no HTTP mocking library needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from minion_assist.providers.embeddings import EmbeddingProvider


def _provider() -> EmbeddingProvider:
    return EmbeddingProvider(
        base_url="http://localhost:1234/v1",
        api_key="test-key",
        model="nomic-embed-text",
        dimensions=768,
    )


def _fake_response(vectors: list[list[float]]):
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


def test_embed_returns_empty_list_for_empty_input_without_a_request():
    provider = _provider()
    provider._client = Mock()

    result = provider.embed([])

    assert result == []
    provider._client.embeddings.create.assert_not_called()


def test_embed_returns_one_vector_per_input_in_order():
    provider = _provider()
    provider._client = Mock()
    provider._client.embeddings.create.return_value = _fake_response(
        [[0.1, 0.2], [0.3, 0.4]]
    )

    result = provider.embed(["first text", "second text"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_sends_the_configured_model_and_all_inputs():
    provider = _provider()
    provider._client = Mock()
    provider._client.embeddings.create.return_value = _fake_response([[0.1]])

    provider.embed(["only text"])

    provider._client.embeddings.create.assert_called_once_with(
        model="nomic-embed-text", input=["only text"]
    )


def test_embed_propagates_client_exceptions():
    provider = _provider()
    provider._client = Mock()
    provider._client.embeddings.create.side_effect = RuntimeError("connection refused")

    try:
        provider.embed(["text"])
        raised = False
    except RuntimeError as exc:
        raised = "connection refused" in str(exc)
    assert raised


def test_stores_model_and_dimensions():
    provider = _provider()
    assert provider.model == "nomic-embed-text"
    assert provider.dimensions == 768
