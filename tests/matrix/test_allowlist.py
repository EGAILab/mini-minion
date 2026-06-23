"""Tests for matrix/allowlist.py."""

from minion_assist.matrix.allowlist import check_allowlist, normalise_matrix_user_id


class TestNormaliseMatrixUserId:
    def test_plain_id(self):
        assert normalise_matrix_user_id("@Alice:example.org") == "@alice:example.org"

    def test_strips_matrix_prefix(self):
        assert normalise_matrix_user_id("matrix:@Alice:example.org") == "@alice:example.org"

    def test_strips_matrix_prefix_case_insensitive(self):
        assert normalise_matrix_user_id("MATRIX:@Alice:example.org") == "@alice:example.org"

    def test_strips_whitespace(self):
        assert normalise_matrix_user_id("  @bob:example.org  ") == "@bob:example.org"

    def test_lowercases(self):
        assert normalise_matrix_user_id("@BOB:EXAMPLE.ORG") == "@bob:example.org"

    def test_empty_string(self):
        assert normalise_matrix_user_id("") == ""


class TestCheckAllowlist:
    def test_wildcard_permits_any(self):
        assert check_allowlist("@anyone:example.org", ["*"]) is True

    def test_exact_match(self):
        assert check_allowlist("@alice:example.org", ["@alice:example.org"]) is True

    def test_case_insensitive_match(self):
        assert check_allowlist("@Alice:Example.Org", ["@alice:example.org"]) is True

    def test_no_match(self):
        assert check_allowlist("@bob:example.org", ["@alice:example.org"]) is False

    def test_empty_allowlist_denies(self):
        assert check_allowlist("@alice:example.org", []) is False

    def test_matrix_prefix_stripped_in_allowlist(self):
        assert check_allowlist("@alice:example.org", ["matrix:@alice:example.org"]) is True

    def test_multiple_entries_first_match(self):
        assert (
            check_allowlist("@bob:example.org", ["@alice:example.org", "@bob:example.org"])
            is True
        )

    def test_multiple_entries_no_match(self):
        assert (
            check_allowlist("@charlie:example.org", ["@alice:example.org", "@bob:example.org"])
            is False
        )
