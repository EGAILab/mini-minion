"""Tests for AskUserTool."""

from minion_assist.tools.ask_user import AskUserTool


def test_ask_user_calls_prompt_fn():
    tool = AskUserTool(prompt_fn=lambda q: f"Answer to: {q}")
    result = tool.execute(question="What is your name?")
    assert result == "Answer to: What is your name?"


def test_ask_user_passes_question_to_prompt_fn():
    received: list[str] = []
    tool = AskUserTool(prompt_fn=lambda q: received.append(q) or "ok")
    tool.execute(question="specific question text")
    assert received == ["specific question text"]


def test_ask_user_returns_prompt_fn_response():
    tool = AskUserTool(prompt_fn=lambda _: "blue")
    result = tool.execute(question="What is your favourite colour?")
    assert result == "blue"


def test_ask_user_headless_returns_error():
    tool = AskUserTool(prompt_fn=None)
    result = tool.execute(question="Hello?")
    assert "Error" in result


def test_ask_user_default_is_headless():
    tool = AskUserTool()
    result = tool.execute(question="Are you there?")
    assert "Error" in result


def test_ask_user_schema_name():
    assert AskUserTool().schema.name == "ask_user"


def test_ask_user_schema_requires_question():
    assert "question" in AskUserTool().schema.parameters["required"]


def test_ask_user_schema_has_question_property():
    assert "question" in AskUserTool().schema.parameters["properties"]
