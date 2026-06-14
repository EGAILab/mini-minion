"""Tests for the permission audit log (NEW-02)."""

from minion_assistant.tools.audit import ApprovalDecision, AuditEntry, AuditLog, _utcnow


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------

def test_audit_entry_stores_all_fields():
    entry = AuditEntry(
        timestamp="2026-01-01T00:00:00+00:00",
        tool_name="bash",
        args_repr="ls -la",
        decision="allowed",
        reason="user approved",
    )
    assert entry.tool_name == "bash"
    assert entry.args_repr == "ls -la"
    assert entry.decision == "allowed"
    assert entry.reason == "user approved"


def test_audit_entry_reason_defaults_empty():
    entry = AuditEntry(timestamp="t", tool_name="write", args_repr="/tmp/f", decision="denied")
    assert entry.reason == ""


# ---------------------------------------------------------------------------
# AuditLog — record and entries
# ---------------------------------------------------------------------------

def test_audit_log_starts_empty():
    log = AuditLog()
    assert log.entries == []


def test_audit_log_record_appends_entry():
    log = AuditLog()
    e = AuditEntry(timestamp=_utcnow(), tool_name="bash", args_repr="ls", decision="allowed")
    log.record(e)
    assert len(log.entries) == 1
    assert log.entries[0].tool_name == "bash"


def test_audit_log_entries_returns_defensive_copy():
    log = AuditLog()
    log.record(AuditEntry(timestamp=_utcnow(), tool_name="x", args_repr="y", decision="allowed"))
    copy = log.entries
    copy.clear()
    # Original entries must still be intact.
    assert len(log.entries) == 1


def test_audit_log_evicts_oldest_when_at_capacity():
    log = AuditLog()
    for i in range(log._MAX):
        log.record(AuditEntry(timestamp=_utcnow(), tool_name="t", args_repr=str(i), decision="allowed"))
    # Add one more — the very first entry should be evicted.
    log.record(AuditEntry(timestamp=_utcnow(), tool_name="t", args_repr="overflow", decision="allowed"))
    assert len(log.entries) == log._MAX
    # First entry should now be index=1 (args_repr="1"), and last is "overflow".
    assert log.entries[0].args_repr == "1"
    assert log.entries[-1].args_repr == "overflow"


# ---------------------------------------------------------------------------
# AuditLog — session allow/deny state
# ---------------------------------------------------------------------------

def test_is_session_allowed_false_initially():
    log = AuditLog()
    assert not log.is_session_allowed("echo hello")


def test_set_session_allowed_marks_command():
    log = AuditLog()
    log.set_session_allowed("echo hello")
    assert log.is_session_allowed("echo hello")
    assert not log.is_session_allowed("rm -rf /")  # different command is not affected


def test_is_session_denied_false_initially():
    log = AuditLog()
    assert not log.is_session_denied("rm -rf /")


def test_set_session_denied_marks_command():
    log = AuditLog()
    log.set_session_denied("rm -rf /")
    assert log.is_session_denied("rm -rf /")
    assert not log.is_session_denied("echo hello")


# ---------------------------------------------------------------------------
# ApprovalDecision enum
# ---------------------------------------------------------------------------

def test_approval_decision_values():
    assert ApprovalDecision.ALLOW_ONCE.value == "allow_once"
    assert ApprovalDecision.ALLOW_SESSION.value == "allow_session"
    assert ApprovalDecision.DENY.value == "deny"
    assert ApprovalDecision.ALWAYS_DENY.value == "always_deny"


def test_approval_decision_enum_members():
    members = {d.value for d in ApprovalDecision}
    assert members == {"allow_once", "allow_session", "deny", "always_deny"}
