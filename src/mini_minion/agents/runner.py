"""Agent turn execution (TAO loop)."""

from __future__ import annotations

import json

from ..providers import LLMProvider, LLMResponse
from ..tools import ToolRegistry


def run_turn(
    provider: LLMProvider,
    agent_name: str,
    system: str,
    max_tokens: int,
    tools: ToolRegistry,
    messages: list[dict],
) -> None:
    """Drive a single user turn to completion, executing tools as needed.

    Mutates ``messages`` in place by appending assistant and tool-result
    messages until the model stops requesting tool calls.
    """
    while True:
        response: LLMResponse = provider.chat(system, messages, tools.definitions, max_tokens)

        assistant_msg: dict = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        if response.finish_reason != "tool_calls":
            if response.text:
                print(f"\n{agent_name}: {response.text}")
            return

        for tc in response.tool_calls:
            print(f"  [tool: {tc.name}({tc.arguments})]")
            output = tools.execute(tc.name, tc.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
