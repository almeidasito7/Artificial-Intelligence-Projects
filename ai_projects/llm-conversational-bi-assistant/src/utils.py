"""
Shared utilities.

Only truly cross-cutting helpers live here — no circular dependencies
with other src/ modules.
"""

import hashlib
import json
import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from tabulate import tabulate

from src.config import settings


def setup_logging() -> logging.Logger:
    """
    Configure the root logger with RichHandler for coloured terminal output.
    Should be called exactly once in the entry point (main.py).
    """
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_path=False,
                markup=True,
            )
        ],
    )
    return logging.getLogger("bi_assistant")


def get_logger(name: str) -> logging.Logger:
    """Return a hierarchical child logger (e.g. 'bi_assistant.cache')."""
    return logging.getLogger(f"bi_assistant.{name}")

def format_sql_results(
    rows: list[dict[str, Any]],
    max_rows: int = 50,
) -> str:
    """
    Format SQL result rows into a readable table string.

    Args:
        rows:     List of dicts {column: value} returned by SQLite.
        max_rows: Maximum rows to display (prevents flooding the terminal).

    Returns:
        Formatted table string with a truncation note if applicable.
    """
    if not rows:
        return "_No results found._"

    truncated = len(rows) > max_rows
    display_rows = rows[:max_rows]

    headers = list(display_rows[0].keys())
    table_data = [[row.get(h, "") for h in headers] for row in display_rows]

    result = tabulate(table_data, headers=headers, tablefmt="rounded_outline")

    if truncated:
        result += f"\n_... showing {max_rows} of {len(rows)} results._"

    return result

def hash_permissions(regions: list[str], divisions: list[str]) -> str:
    """
    Generate a deterministic hash from a user's permission set.

    Ensures two users with different permissions never share cache entries,
    regardless of their username.

    Returns:
        16-character hex string (first 8 bytes of SHA-256).
    """
    payload = json.dumps(
        {"regions": sorted(regions), "divisions": sorted(divisions)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

_stdout_console = Console(stderr=False)
_stderr_console = Console(stderr=True)


def print_answer(answer: str, source: str | None = None, cache_hit: bool = False) -> None:
    """Print the agent response formatted for the terminal."""
    _stdout_console.print()

    if cache_hit:
        _stdout_console.print("[dim]⚡ Cache hit[/dim]")

    _stdout_console.print(answer)

    if source:
        _stdout_console.print(f"\n[dim]Source: {source}[/dim]")

    _stdout_console.print()


def print_error(message: str) -> None:
    """Print a formatted error message to stderr."""
    _stderr_console.print(f"[red]✗ Error:[/red] {message}")
