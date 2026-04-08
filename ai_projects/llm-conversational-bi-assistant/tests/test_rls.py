"""Unit tests for Row-Level Security."""

from dataclasses import FrozenInstanceError

import pytest

from src.security.rls import (
    SQLSecurityError,
    UserProfile,
    apply_rls,
    load_user,
    validate_sql,
)


@pytest.fixture
def alice() -> UserProfile:
    return UserProfile(username="alice", regions=["Southeast"], divisions=["IT"])


@pytest.fixture
def multi_region_user() -> UserProfile:
    return UserProfile(
        username="admin",
        regions=["Southeast", "West Coast"],
        divisions=["IT", "Finance"],
    )


class TestValidateSQL:
    def test_select_is_allowed(self):
        validate_sql("SELECT * FROM jobs")  # should not raise

    def test_select_with_where(self):
        validate_sql("SELECT id, title FROM jobs WHERE status = 'open'")

    def test_insert_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="INSERT"):
            validate_sql("INSERT INTO jobs VALUES (1, 'Dev', 'SE', 'IT', 'open', 100, '2026-01-01')")

    def test_update_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="UPDATE"):
            validate_sql("UPDATE jobs SET status = 'filled' WHERE id = 1")

    def test_delete_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="DELETE"):
            validate_sql("DELETE FROM jobs")

    def test_drop_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="DROP"):
            validate_sql("DROP TABLE jobs")

    def test_multi_statement_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="multiple"):
            validate_sql("SELECT * FROM jobs; DROP TABLE jobs")

    def test_comment_injection_blocked(self):
        """Classic SQL injection via comment."""
        with pytest.raises(SQLSecurityError):
            validate_sql("SELECT * FROM jobs; -- DROP TABLE jobs\nDROP TABLE jobs")

    def test_non_select_start_is_blocked(self):
        with pytest.raises(SQLSecurityError, match="SELECT"):
            validate_sql("PRAGMA table_info(jobs)")

    def test_case_insensitive_blocking(self):
        with pytest.raises(SQLSecurityError):
            validate_sql("delete from jobs")


class TestApplyRLS:
    def test_injects_region_filter(self, alice):
        sql = "SELECT * FROM jobs"
        result = apply_rls(sql, alice)
        assert "jobs.region IN ('Southeast')" in result

    def test_injects_division_filter(self, alice):
        sql = "SELECT * FROM jobs"
        result = apply_rls(sql, alice)
        assert "jobs.division IN ('IT')" in result

    def test_multiple_regions(self, multi_region_user):
        sql = "SELECT COUNT(*) FROM jobs"
        result = apply_rls(sql, multi_region_user)
        assert "'Southeast'" in result
        assert "'West Coast'" in result

    def test_preserves_existing_where(self, alice):
        sql = "SELECT * FROM jobs WHERE status = 'open'"
        result = apply_rls(sql, alice)
        rls_pos = result.index("jobs.region")
        orig_pos = result.index("status")
        assert rls_pos < orig_pos

    def test_inserts_before_order_by(self, alice):
        sql = "SELECT * FROM jobs ORDER BY created_at DESC"
        result = apply_rls(sql, alice)
        where_pos = result.upper().index("WHERE")
        order_pos = result.upper().index("ORDER BY")
        assert where_pos < order_pos

    def test_inserts_before_limit(self, alice):
        sql = "SELECT * FROM jobs LIMIT 10"
        result = apply_rls(sql, alice)
        where_pos = result.upper().index("WHERE")
        limit_pos = result.upper().index("LIMIT")
        assert where_pos < limit_pos

    def test_no_rls_table_passthrough(self, alice):
        """Table without RLS should not be modified."""
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        result = apply_rls(sql, alice)
        assert result == sql

    def test_placements_table_filtered(self, alice):
        sql = "SELECT * FROM placements"
        result = apply_rls(sql, alice)
        assert "placements.region" in result

    def test_join_query_filters_both_tables(self, alice):
        """RLS must use the alias (j, p) not the table name when aliases are present."""
        sql = "SELECT j.title, p.bill_rate FROM jobs j JOIN placements p ON j.id = p.job_id"
        result = apply_rls(sql, alice)
        assert "j.region" in result
        assert "p.region" in result

    def test_alias_with_as_keyword(self, alice):
        """RLS must handle AS alias syntax."""
        sql = "SELECT j.title FROM jobs AS j WHERE j.status = 'open'"
        result = apply_rls(sql, alice)
        assert "j.region" in result
        assert "jobs.region" not in result

    def test_no_alias_uses_table_name(self, alice):
        """Without alias, RLS must fall back to the full table name."""
        sql = "SELECT COUNT(*) FROM jobs"
        result = apply_rls(sql, alice)
        assert "jobs.region" in result

    def test_alias_count_query_real_world(self, alice):
        """Reproduces the exact error seen in production: aliased COUNT query."""
        sql = "SELECT COUNT(*) AS open_positions FROM jobs AS j WHERE j.status = 'open'"
        result = apply_rls(sql, alice)
        assert "j.region IN ('Southeast')" in result
        assert "jobs.region" not in result

    def test_blocks_ddl_before_rls(self, alice):
        with pytest.raises(SQLSecurityError):
            apply_rls("DROP TABLE jobs", alice)


class TestLoadUser:
    def test_loads_existing_user(self, permissions_file):
        user = load_user("alice", permissions_path=permissions_file)
        assert user.username == "alice"
        assert "Southeast" in user.regions

    def test_raises_for_unknown_user(self, permissions_file):
        with pytest.raises(KeyError, match="not found"):
            load_user("nonexistent_user_xyz", permissions_path=permissions_file)

    def test_user_profile_is_frozen(self, permissions_file):
        user = load_user("alice", permissions_path=permissions_file)
        with pytest.raises(FrozenInstanceError):
            user.username = "hacker"  # type: ignore[misc]
