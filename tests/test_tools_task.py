"""Tests for ReadTaskTool and UpdateTaskTool."""

import json
import pytest
from pathlib import Path

from minion_assistant.tools.task import ReadTaskTool, UpdateTaskTool


def _pair(tmp_path: Path):
    """Return a (ReadTaskTool, UpdateTaskTool) pair sharing one task file."""
    p = tmp_path / "test.json"
    return ReadTaskTool(p), UpdateTaskTool(p)


# ---------------------------------------------------------------------------
# ReadTaskTool — no active task
# ---------------------------------------------------------------------------

def test_read_returns_no_active_task_when_file_absent(tmp_path):
    reader, _ = _pair(tmp_path)
    result = reader.execute()
    assert "No active task" in result


# ---------------------------------------------------------------------------
# UpdateTaskTool — create
# ---------------------------------------------------------------------------

def test_create_task_with_goal_and_steps(tmp_path):
    _, updater = _pair(tmp_path)
    result = updater.execute(goal="Build the API", steps=["Read code", "Write tests"])
    assert "created" in result.lower()
    assert "Build the API" in result
    assert "2 step" in result


def test_create_task_writes_json_file(tmp_path):
    p = tmp_path / "task.json"
    updater = UpdateTaskTool(p)
    updater.execute(goal="Test goal", steps=["Step A"])
    data = json.loads(p.read_text())
    assert data["goal"] == "Test goal"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["description"] == "Step A"
    assert data["steps"][0]["status"] == "pending"


def test_create_task_with_no_steps(tmp_path):
    _, updater = _pair(tmp_path)
    result = updater.execute(goal="Just a goal")
    assert "0 step" in result


def test_create_task_atomic_write(tmp_path):
    """The task file should not leave a .tmp file behind after a successful write."""
    p = tmp_path / "task.json"
    UpdateTaskTool(p).execute(goal="G", steps=["S"])
    assert not (tmp_path / "task.tmp").exists()
    assert p.exists()


# ---------------------------------------------------------------------------
# UpdateTaskTool — update step status
# ---------------------------------------------------------------------------

def test_update_step_status(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["Step 1", "Step 2"])
    result = updater.execute(step_id=1, status="in_progress")
    assert "step 1" in result.lower()
    assert "in_progress" in result

    data = json.loads((tmp_path / "test.json").read_text())
    assert data["steps"][0]["status"] == "in_progress"


def test_update_step_notes(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1"])
    updater.execute(step_id=1, notes="Found 3 files")
    data = json.loads((tmp_path / "test.json").read_text())
    assert data["steps"][0]["notes"] == "Found 3 files"


def test_update_step_status_and_notes_together(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1"])
    result = updater.execute(step_id=1, status="done", notes="All done")
    assert "step 1" in result.lower()
    assert "done" in result
    assert "notes" in result.lower()


def test_update_unknown_step_id_returns_error(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1"])
    result = updater.execute(step_id=99, status="done")
    assert "not found" in result.lower() or "99" in result


def test_update_invalid_status_returns_error(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1"])
    result = updater.execute(step_id=1, status="flying")
    assert "invalid" in result.lower() or "flying" in result


# ---------------------------------------------------------------------------
# UpdateTaskTool — context
# ---------------------------------------------------------------------------

def test_update_context(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S"])
    result = updater.execute(context="Key file: src/main.py")
    assert "context" in result.lower()
    data = json.loads((tmp_path / "test.json").read_text())
    assert data["context"] == "Key file: src/main.py"


# ---------------------------------------------------------------------------
# UpdateTaskTool — clear
# ---------------------------------------------------------------------------

def test_clear_removes_task_file(tmp_path):
    p = tmp_path / "test.json"
    updater = UpdateTaskTool(p)
    updater.execute(goal="G", steps=["S"])
    assert p.exists()
    result = updater.execute(clear=True)
    assert "cleared" in result.lower()
    assert not p.exists()


def test_clear_when_no_task_does_not_raise(tmp_path):
    _, updater = _pair(tmp_path)
    result = updater.execute(clear=True)
    assert "cleared" in result.lower()


# ---------------------------------------------------------------------------
# ReadTaskTool — with active task
# ---------------------------------------------------------------------------

def test_read_shows_goal_and_steps(tmp_path):
    reader, updater = _pair(tmp_path)
    updater.execute(goal="Build API", steps=["Read code", "Write tests", "Deploy"])
    result = reader.execute()
    assert "Build API" in result
    assert "Read code" in result
    assert "Write tests" in result


def test_read_shows_step_statuses(tmp_path):
    reader, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1", "S2"])
    updater.execute(step_id=1, status="done")
    result = reader.execute()
    assert "✓" in result   # done icon


def test_read_shows_next_step(tmp_path):
    reader, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1", "S2"])
    updater.execute(step_id=1, status="done")
    result = reader.execute()
    assert "Next" in result
    assert "S2" in result


def test_read_shows_all_complete_when_done(tmp_path):
    reader, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S1"])
    updater.execute(step_id=1, status="done")
    result = reader.execute()
    assert "All steps complete" in result


def test_read_shows_context_if_set(tmp_path):
    reader, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S"])
    updater.execute(context="Important: use v2 API")
    result = reader.execute()
    assert "Important: use v2 API" in result


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

def test_read_task_schema(tmp_path):
    reader, _ = _pair(tmp_path)
    schema = reader.schema
    assert schema.name == "read_task"
    assert "task" in schema.description.lower()


def test_update_task_schema(tmp_path):
    _, updater = _pair(tmp_path)
    schema = updater.schema
    assert schema.name == "update_task"
    assert "goal" in schema.parameters["properties"]
    assert "step_id" in schema.parameters["properties"]
    assert "status" in schema.parameters["properties"]
    assert "context" in schema.parameters["properties"]
    assert "clear" in schema.parameters["properties"]


# ---------------------------------------------------------------------------
# UpdateTaskTool — nothing updated
# ---------------------------------------------------------------------------

def test_update_with_no_fields_returns_nothing_updated(tmp_path):
    _, updater = _pair(tmp_path)
    updater.execute(goal="G", steps=["S"])
    result = updater.execute()
    assert "nothing updated" in result.lower()


def test_update_no_active_task_without_goal(tmp_path):
    _, updater = _pair(tmp_path)
    result = updater.execute(step_id=1, status="done")
    assert "no active task" in result.lower()
