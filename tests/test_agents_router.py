"""Tests for agent message routing."""

from mini_minion.agents.router import resolve


def test_default_routes_to_main():
    agent_id, message = resolve("hello world")
    assert agent_id == "main"
    assert message == "hello world"


def test_research_prefix_routes_to_researcher():
    agent_id, message = resolve("/research find something")
    assert agent_id == "researcher"
    assert message == "find something"


def test_research_prefix_strips_command_only():
    _, message = resolve("/research   extra spaces")
    assert message == "  extra spaces"


def test_research_without_space_routes_to_main():
    agent_id, _ = resolve("/researchfoo")
    assert agent_id == "main"


def test_research_alone_no_trailing_space_routes_to_main():
    """'/research' with no trailing space is NOT the research command."""
    agent_id, _ = resolve("/research")
    assert agent_id == "main"


def test_research_with_trailing_space_routes_to_researcher():
    """'/research ' (with space, empty payload) still routes to researcher."""
    agent_id, message = resolve("/research ")
    assert agent_id == "researcher"
    assert message == ""


def test_empty_message_routes_to_main():
    agent_id, message = resolve("")
    assert agent_id == "main"
    assert message == ""
