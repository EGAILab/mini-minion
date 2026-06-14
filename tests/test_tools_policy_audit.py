"""Tests for PermissionPolicy audit log integration (NEW-02)."""

from pathlib import Path

from minion_assistant.tools.audit import AuditLog
from minion_assistant.tools.policy import PermissionPolicy


def test_policy_has_audit_log_by_default():
    policy = PermissionPolicy()
    assert isinstance(policy.audit_log, AuditLog)


def test_check_path_records_sensitive_path_denial(tmp_path):
    policy = PermissionPolicy.default(workspace=tmp_path)
    sensitive = Path.home() / ".ssh" / "id_rsa"
    policy.check_path(sensitive)
    entries = policy.audit_log.entries
    assert any(e.decision == "denied" for e in entries)


def test_check_path_records_workspace_boundary_denial(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path / "project")
    outside = tmp_path / "outside.txt"
    policy.check_path(outside)
    entries = policy.audit_log.entries
    assert any("boundary" in e.reason or "workspace" in e.reason for e in entries)


def test_check_path_records_no_entry_when_allowed(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path)
    allowed = tmp_path / "allowed.txt"
    policy.check_path(allowed)
    # No denial should be recorded for an allowed path.
    entries = policy.audit_log.entries
    assert entries == []


def test_check_url_records_ssrf_denial(tmp_path):
    policy = PermissionPolicy.default()
    policy.check_url("http://169.254.169.254/latest/meta-data/")
    entries = policy.audit_log.entries
    assert any(e.decision == "denied" and "SSRF" in e.reason for e in entries)


def test_check_command_records_ssrf_denial():
    policy = PermissionPolicy.default()
    policy.check_command("curl http://169.254.169.254/")
    entries = policy.audit_log.entries
    assert any(e.decision == "denied" for e in entries)


def test_check_write_records_read_only_mode_denial(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    policy.check_write(tmp_path / "file.txt")
    entries = policy.audit_log.entries
    assert any("read-only" in e.reason for e in entries)


def test_check_command_records_read_only_mode_denial():
    policy = PermissionPolicy(read_only_mode=True)
    policy.check_command("rm -rf /")
    entries = policy.audit_log.entries
    assert any("read-only" in e.reason for e in entries)


def test_audit_log_shared_across_multiple_checks(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path / "project")
    policy.check_path(tmp_path / "outside.txt")    # denied
    policy.check_path(tmp_path / "project" / "ok.txt")  # allowed (no record)
    policy.check_url("http://169.254.169.254/")   # denied
    denied = [e for e in policy.audit_log.entries if e.decision == "denied"]
    assert len(denied) == 2
