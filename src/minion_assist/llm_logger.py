"""Verbose LLM request/response logger.

Appends timestamped entries to a daily log file in the same format as LM Studio's
server log so both files can be correlated by timestamp:

    [2026-07-06 12:41:00][DEBUG] Received request: POST to {endpoint} with body {JSON}
    [2026-07-06 12:41:55][INFO][{model}] Generated prediction: {JSON}

Usage:
    from minion_assist.llm_logger import log_request, log_response

    log_request(log_dir, "http://127.0.0.1:1234/v1/chat/completions", body_dict)
    log_response(log_dir, "qwen3.5-9b", response_dict)

Failures are swallowed silently so a disk error never crashes the LLM call.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_file(log_dir: Path) -> Path:
    return log_dir / f"{date.today().isoformat()}.log"


def log_request(log_dir: Path, endpoint: str, body: dict) -> None:
    """Append a request entry to today's log file."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        body_str = json.dumps(body, indent=2, ensure_ascii=False, default=str)
        entry = f"[{_now()}][DEBUG] Received request: POST to {endpoint} with body {body_str}\n"
        with _log_file(log_dir).open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


def log_response(log_dir: Path, model: str, response: dict) -> None:
    """Append a response entry to today's log file."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        resp_str = json.dumps(response, indent=2, ensure_ascii=False, default=str)
        entry = f"[{_now()}][INFO][{model}] Generated prediction: {resp_str}\n"
        with _log_file(log_dir).open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


_TURN_SNIPPET = 200
_TOOL_RESULT_TRUNCATE = 2000


def log_turn_start(log_dir: Path, agent_name: str, user_message: str) -> None:
    """Append a turn-start marker showing the user's message."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        snippet = user_message[:_TURN_SNIPPET]
        if len(user_message) > _TURN_SNIPPET:
            snippet += "..."
        entry = f"[{_now()}][TURN][{agent_name}] User: {snippet!r}\n"
        with _log_file(log_dir).open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


def log_tool_call(log_dir: Path, agent_name: str, tool_name: str, arguments: dict) -> None:
    """Append a tool-call entry showing what tool was invoked and with what arguments."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        args_str = json.dumps(arguments, ensure_ascii=False, default=str)
        entry = f"[{_now()}][TOOL_CALL][{agent_name}] {tool_name} {args_str}\n"
        with _log_file(log_dir).open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


def log_tool_result(log_dir: Path, agent_name: str, tool_name: str, result: str) -> None:
    """Append a tool-result entry showing what the tool returned (truncated at 2000 chars)."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        total = len(result)
        snippet = result[:_TOOL_RESULT_TRUNCATE]
        suffix = f" [truncated, {total} chars total]" if total > _TOOL_RESULT_TRUNCATE else ""
        entry = f"[{_now()}][TOOL_RESULT][{agent_name}] {tool_name} → {total} chars: {snippet}{suffix}\n"
        with _log_file(log_dir).open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass
