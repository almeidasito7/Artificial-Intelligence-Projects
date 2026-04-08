"""
Row-Level Security (RLS)

Responsibilities:
  1. Load user permission profiles from user_permissions.json
  2. Validate SQL queries (SELECT only, no DDL/DML)
  3. Inject WHERE clauses for region/division BEFORE any execution
     — the LLM never sees data the user is not authorized to access.

Design note: injection uses string-based SQL parsing rather than a full
parser (sqlglot/sqlparse) to keep dependencies minimal.
For production, replace with sqlglot for full robustness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.utils import get_logger

logger = get_logger("security.rls")

_RLS_TABLES = {"jobs", "candidates", "placements"}

_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

_MULTI_STATEMENT = re.compile(r";\s*\S")

_ALIAS_SKIP = {
    "WHERE", "ON", "SET", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
    "CROSS", "GROUP", "ORDER", "HAVING", "LIMIT", "AS", "SELECT",
    "FROM", "AND", "OR",
}

_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_RLS_TABLES)) + r")\b\s+(?:AS\s+)?(\w+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UserProfile:
    """
    Immutable profile of an authenticated user.

    Attributes:
        username:  Unique user identifier.
        regions:   List of regions the user can access.
        divisions: List of divisions the user can access.
    """

    username: str
    regions: list[str] = field(default_factory=list)
    divisions: list[str] = field(default_factory=list)

    def permissions_summary(self) -> str:
        """Human-readable permission summary — injected into LLM prompts."""
        return (
            f"User '{self.username}' has access to regions: {self.regions} "
            f"and divisions: {self.divisions}."
        )


_permissions_cache: dict[str, dict] = {}


def _load_permissions_raw(path: str) -> dict:
    """
    Load and cache permissions JSON by file path.

    Supports two JSON shapes:
      - Flat:   { "alice": { "regions": [...], ... } }
      - Nested: { "users": { "alice": { ... } }, "_metadata": { ... } }

    Always returns a flat dict of { username -> profile }.
    """
    if path not in _permissions_cache:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if "users" in raw and isinstance(raw["users"], dict):
            _permissions_cache[path] = raw["users"]
        else:
            _permissions_cache[path] = raw
    return _permissions_cache[path]


def _clear_permissions_cache() -> None:
    """Clear the permissions cache — used in tests for isolation."""
    _permissions_cache.clear()


def load_user(username: str, permissions_path: Path | None = None) -> UserProfile:
    """
    Load a user profile by username.

    Args:
        username:         Name of the user to load.
        permissions_path: Path to the JSON file. Defaults to settings value.

    Raises:
        KeyError: If the user does not exist in the permissions file.
    """
    from src.config import settings

    path = permissions_path or settings.user_permissions_path
    data = _load_permissions_raw(str(path))

    if username not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"User '{username}' not found. "
            f"Available users: {available}"
        )

    profile = data[username]
    return UserProfile(
        username=username,
        regions=profile.get("regions", []),
        divisions=profile.get("divisions", []),
    )


def list_users(permissions_path: Path | None = None) -> list[str]:
    """Return a sorted list of registered usernames."""
    from src.config import settings

    path = permissions_path or settings.user_permissions_path
    return sorted(_load_permissions_raw(str(path)).keys())


class SQLSecurityError(ValueError):
    """Raised when a SQL query violates security rules."""


def validate_sql(sql: str) -> None:
    """
    Validate that a SQL query is safe to execute.

    Rules:
      - No multiple statements (prevents SQLi via ;) — checked first
      - Only SELECT is allowed (no INSERT/UPDATE/DELETE/DDL)
      - Must begin with SELECT after comment removal

    Raises:
        SQLSecurityError: If any rule is violated.
    """
    clean = re.sub(r"--[^\n]*", "", sql)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    clean = clean.strip()

    if _MULTI_STATEMENT.search(clean):
        raise SQLSecurityError(
            "Query blocked: multiple statements detected. "
            "Submit only one query at a time."
        )

    if _BLOCKED_KEYWORDS.search(clean):
        keyword = _BLOCKED_KEYWORDS.search(clean).group(1).upper()  # type: ignore[union-attr]
        raise SQLSecurityError(
            f"Query blocked: contains '{keyword}'. "
            "Only SELECT statements are allowed."
        )

    if not re.match(r"^\s*SELECT\b", clean, re.IGNORECASE):
        raise SQLSecurityError(
            "Query blocked: only SELECT queries are permitted."
        )

    logger.debug("SQL validated successfully")


def _extract_table_aliases(sql: str) -> dict[str, str]:
    """
    Extract table aliases from a SQL query.

    Detects patterns like:
      FROM jobs AS j    -> {"jobs": "j"}
      FROM jobs j       -> {"jobs": "j"}
      JOIN placements p -> {"placements": "p"}

    Returns a dict mapping table_name (lowercase) -> alias.
    Tables without an alias are not included — callers fall back to the
    full table name when the key is absent.
    """
    aliases: dict[str, str] = {}
    for match in _ALIAS_PATTERN.finditer(sql):
        table = match.group(1).lower()
        candidate = match.group(2)
        if candidate.upper() not in _ALIAS_SKIP:
            aliases[table] = candidate
    return aliases


def _build_rls_conditions(user: UserProfile, tables: set[str], sql: str = "") -> list[str]:
    """
    Build SQL condition list for the detected tables.

    Uses the alias found in the query (e.g. 'j') instead of the full table
    name (e.g. 'jobs') to avoid 'no such column: jobs.region' when the LLM
    generates aliased queries.
    """
    aliases = _extract_table_aliases(sql)
    conditions: list[str] = []

    for table in sorted(tables):
        ref = aliases.get(table.lower(), table)

        if user.regions:
            quoted = ", ".join(f"'{r}'" for r in user.regions)
            conditions.append(f"{ref}.region IN ({quoted})")

        if user.divisions:
            quoted = ", ".join(f"'{d}'" for d in user.divisions)
            conditions.append(f"{ref}.division IN ({quoted})")

    return conditions


def apply_rls(sql: str, user: UserProfile) -> str:
    """
    Inject region and division filters into the SQL query.

    Injection happens BEFORE any execution — the LLM never sees results
    beyond the user's permissions.

    Strategy:
      1. Detect which RLS-aware tables are present in the query.
      2. Detect table aliases (e.g. jobs AS j -> use j.region, not jobs.region).
      3. Build WHERE conditions for each table/alias.
      4. Prepend conditions to any existing WHERE, or create a new one.

    Args:
        sql:  SQL query generated by the LLM (already validated).
        user: User profile containing allowed regions and divisions.

    Returns:
        SQL query with RLS filters injected.
    """
    validate_sql(sql)

    if not user.regions and not user.divisions:
        logger.warning(
            f"User '{user.username}' has no regions/divisions — query will return 0 rows"
        )

    sql_upper = sql.upper()
    tables_in_query = {t for t in _RLS_TABLES if t.upper() in sql_upper}

    if not tables_in_query:
        return sql

    conditions = _build_rls_conditions(user, tables_in_query, sql)

    if not conditions:
        return sql

    combined = " AND ".join(conditions)
    filtered_sql = _inject_where(sql, combined)

    logger.debug(
        f"RLS applied for '{user.username}': "
        f"tables={tables_in_query}, conditions={combined}"
    )

    return filtered_sql


def _inject_where(sql: str, conditions: str) -> str:
    """
    Inject conditions into the SQL query.

    - If WHERE already exists, prepend RLS conditions with AND before the
      original conditions (prevents bypass via OR introduced by the LLM).
    - If no WHERE, insert before GROUP BY / ORDER BY / LIMIT or at the end.
    """
    where_match = re.search(r"\bWHERE\b", sql, re.IGNORECASE)

    if where_match:
        pos = where_match.end()
        return sql[:pos] + f" ({conditions}) AND " + sql[pos:]

    for keyword in (r"\bGROUP\s+BY\b", r"\bORDER\s+BY\b", r"\bLIMIT\b", r"\bHAVING\b"):
        match = re.search(keyword, sql, re.IGNORECASE)
        if match:
            pos = match.start()
            return sql[:pos] + f"WHERE ({conditions}) " + sql[pos:]

    return sql.rstrip().rstrip(";") + f" WHERE ({conditions})"
