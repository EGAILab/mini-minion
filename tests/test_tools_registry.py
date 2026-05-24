"""Tests for ToolRegistry."""

from mini_minion.tools.base import Tool, ToolSchema
from mini_minion.tools.registry import ToolRegistry


class _AddTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="add",
            description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        return str(float(kwargs["a"]) + float(kwargs["b"]))


class _ConstTool(Tool):
    """Always returns 'const'."""
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="add", description="Const.", parameters={"type": "object", "properties": {}})

    def execute(self, **kwargs: object) -> str:
        return "const"


def test_empty_registry_has_no_definitions():
    assert ToolRegistry().definitions == []


def test_register_and_definitions():
    reg = ToolRegistry()
    reg.register(_AddTool())
    defs = reg.definitions
    assert len(defs) == 1
    d = defs[0]
    assert d["type"] == "function"
    assert d["function"]["name"] == "add"
    assert d["function"]["description"] == "Add two numbers."
    assert "parameters" in d["function"]


def test_definitions_openai_format():
    reg = ToolRegistry()
    reg.register(_AddTool())
    fn = reg.definitions[0]["function"]
    assert set(fn.keys()) >= {"name", "description", "parameters"}


def test_execute_known_tool():
    reg = ToolRegistry()
    reg.register(_AddTool())
    assert reg.execute("add", {"a": 2, "b": 3}) == "5.0"


def test_execute_unknown_tool():
    result = ToolRegistry().execute("nope", {})
    assert "nope" in result


def test_execute_tool_exception_returns_error_string():
    class _BrokenTool(Tool):
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(name="broken", description="", parameters={"type": "object", "properties": {}})

        def execute(self, **kwargs: object) -> str:
            raise ValueError("boom")

    reg = ToolRegistry()
    reg.register(_BrokenTool())
    result = reg.execute("broken", {})
    assert "Error" in result
    assert "boom" in result


def test_register_overwrites_same_name():
    """Re-registering a tool with the same name replaces the previous one."""
    reg = ToolRegistry()
    reg.register(_AddTool())
    reg.register(_ConstTool())
    assert reg.execute("add", {"a": 1, "b": 2}) == "const"
    assert len(reg.definitions) == 1
