"""Tests for ToolRegistry."""

from minion_assist.tools.base import Tool, ToolSchema
from minion_assist.tools.registry import ToolRegistry


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


# ---------------------------------------------------------------------------
# IMP-11: Per-tool execution hooks
# ---------------------------------------------------------------------------


class _EchoTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="echo", description="Echo.", parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})

    def execute(self, **kwargs: object) -> str:
        return str(kwargs.get("text", ""))


def test_after_execute_hook_called_with_correct_args():
    """after_execute hook receives (name, args, output, elapsed_ms)."""
    calls: list = []
    reg = ToolRegistry(after_execute=lambda n, a, o, e: calls.append((n, a, o, e)))
    reg.register(_EchoTool())
    output = reg.execute("echo", {"text": "hi"})
    assert len(calls) == 1
    name, args, out, elapsed = calls[0]
    assert name == "echo"
    assert args == {"text": "hi"}
    assert out == "hi"
    assert elapsed >= 0


def test_before_execute_hook_called():
    calls: list = []
    reg = ToolRegistry(before_execute=lambda n, a: calls.append(n))
    reg.register(_EchoTool())
    reg.execute("echo", {"text": "x"})
    assert calls == ["echo"]


def test_hook_exception_does_not_crash_execution():
    """A buggy hook must never prevent the tool from running."""
    def bad_hook(n, a, o, e):
        raise RuntimeError("hook broke")

    reg = ToolRegistry(after_execute=bad_hook)
    reg.register(_EchoTool())
    assert reg.execute("echo", {"text": "hello"}) == "hello"


def test_is_read_only_returns_true_for_read_only_tool():
    from minion_assist.tools.read import ReadTool
    reg = ToolRegistry()
    reg.register(ReadTool(root=None))
    assert reg.is_read_only("read") is True


def test_is_read_only_returns_false_for_unknown_tool():
    assert ToolRegistry().is_read_only("nonexistent") is False
