from .base import Tool, ToolSchema
from .registry import ToolRegistry
from .read import ReadTool
from .write import WriteTool
from .glob import GlobTool
from .bash import BashTool
from .memory import SaveMemoryTool, SearchMemoryTool


def default_registry(long_term=None) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [ReadTool(), WriteTool(), GlobTool(), BashTool()]:
        registry.register(tool)
    if long_term is not None:
        registry.register(SaveMemoryTool(long_term))
        registry.register(SearchMemoryTool(long_term))
    return registry


__all__ = ["Tool", "ToolSchema", "ToolRegistry", "default_registry"]
