"""Tests for provider base types and create_provider factory."""

import pytest

from mini_minion.providers.base import LLMResponse, ToolCall
from mini_minion.providers import create_provider
from mini_minion.providers.openai_compatible import OpenAICompatibleProvider
from mini_minion.providers.lmstudio import LMStudioProvider
from mini_minion.providers.anthropic import AnthropicProvider


def test_tool_call_fields():
    tc = ToolCall(id="abc", name="read", arguments={"path": "/tmp"})
    assert tc.id == "abc"
    assert tc.name == "read"
    assert tc.arguments == {"path": "/tmp"}


def test_llm_response_defaults():
    r = LLMResponse(text="hello")
    assert r.text == "hello"
    assert r.tool_calls == []
    assert r.finish_reason == "stop"


def test_llm_response_with_tool_calls():
    tc = ToolCall(id="1", name="bash", arguments={"command": "echo"})
    r = LLMResponse(text="", tool_calls=[tc], finish_reason="tool_calls")
    assert len(r.tool_calls) == 1
    assert r.finish_reason == "tool_calls"


def test_create_provider_openai_completions():
    p = create_provider(api="openai-completions", base_url="http://x", api_key="k", model="m")
    assert isinstance(p, OpenAICompatibleProvider)


def test_create_provider_openai_responses():
    p = create_provider(api="openai-responses", base_url="http://x", api_key="k", model="m")
    assert isinstance(p, OpenAICompatibleProvider)


def test_create_provider_lmstudio():
    p = create_provider(api="lmstudio", base_url="http://x", api_key="k", model="m")
    assert isinstance(p, LMStudioProvider)
    assert isinstance(p, OpenAICompatibleProvider)  # LMStudio is an alias


def test_create_provider_anthropic():
    pytest.importorskip("anthropic", reason="anthropic package not installed")
    p = create_provider(api="anthropic", base_url="", api_key="k", model="claude-3")
    assert isinstance(p, AnthropicProvider)


def test_create_provider_unknown_falls_back_to_openai():
    p = create_provider(api="some-unknown-api", base_url="http://x", api_key="k", model="m")
    assert isinstance(p, OpenAICompatibleProvider)
