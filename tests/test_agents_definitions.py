"""Tests for agent definitions."""

from mini_minion.agents.definitions import AGENTS, AgentConfig


def test_agents_has_main_and_researcher():
    assert "main" in AGENTS
    assert "researcher" in AGENTS


def test_agents_are_agent_config_instances():
    for agent in AGENTS.values():
        assert isinstance(agent, AgentConfig)


def test_agent_names():
    assert AGENTS["main"].name == "Ada"
    assert AGENTS["researcher"].name == "Elizabeth"


def test_agents_have_non_empty_souls():
    for agent in AGENTS.values():
        assert agent.soul.strip()


def test_agent_souls_are_distinct():
    souls = [a.soul for a in AGENTS.values()]
    assert len(souls) == len(set(souls))
