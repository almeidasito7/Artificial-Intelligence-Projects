"""
Shared fixtures for all tests.

Imported automatically by pytest — no need to import in individual test files.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.security.rls import UserProfile


@pytest.fixture
def user_southeast() -> UserProfile:
    return UserProfile(
        username="alice",
        regions=["Southeast"],
        divisions=["IT", "Finance"],
    )


@pytest.fixture
def user_west() -> UserProfile:
    return UserProfile(
        username="bob",
        regions=["West Coast"],
        divisions=["IT"],
    )


@pytest.fixture
def user_admin() -> UserProfile:
    return UserProfile(
        username="admin",
        regions=["Southeast", "West Coast", "Northeast", "Midwest"],
        divisions=["IT", "Finance", "HR", "Operations"],
    )


@pytest.fixture
def permissions_file(tmp_path: Path) -> Path:
    """Write a temporary user_permissions.json to disk."""
    data = {
        "alice": {"regions": ["Southeast"], "divisions": ["IT", "Finance"]},
        "bob":   {"regions": ["West Coast"], "divisions": ["IT"]},
        "admin": {
            "regions": ["Southeast", "West Coast", "Northeast", "Midwest"],
            "divisions": ["IT", "Finance", "HR", "Operations"],
        },
    }
    p = tmp_path / "user_permissions.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def in_memory_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, title TEXT, region TEXT,
            division TEXT, status TEXT, bill_rate REAL, created_at TEXT
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY, name TEXT, region TEXT, division TEXT
        );
        CREATE TABLE placements (
            id INTEGER PRIMARY KEY, candidate_id INTEGER, job_id INTEGER,
            region TEXT, division TEXT, start_date TEXT, bill_rate REAL
        );
        INSERT INTO jobs VALUES
            (1,'Python Dev','Southeast','IT','open',95.0,'2026-01-01'),
            (2,'Data Analyst','West Coast','Finance','open',80.0,'2026-01-05'),
            (3,'DevOps','Southeast','IT','filled',110.0,'2026-01-10');
        INSERT INTO candidates VALUES
            (1,'Alice Smith','Southeast','IT'),
            (2,'Bob Jones','West Coast','Finance');
        INSERT INTO placements VALUES
            (1,1,3,'Southeast','IT','2026-02-01',110.0),
            (2,2,2,'West Coast','Finance','2026-02-15',80.0);
    """)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clear_rls_cache():
    """Ensure the rls.py permissions cache does not leak between tests."""
    from src.security.rls import _clear_permissions_cache
    _clear_permissions_cache()
    yield
    _clear_permissions_cache()
