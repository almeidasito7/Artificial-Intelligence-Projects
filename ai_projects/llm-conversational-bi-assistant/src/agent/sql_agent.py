"""
SQL Agent — Text-to-SQL

Flow:
  1. Receive a natural language question + UserProfile
  2. Introspect the real database schema at startup (no hardcoded columns)
  3. Send to LLM with the live schema and user permission context
  4. Extract the generated SQL from the response
  5. Apply RLS (inject region/division filters)
  6. Execute against SQLite (read-only)
  7. Format and return the result in natural language

  On any error (SQL execution, RLS violation, LLM failure), return a
  user-friendly message instead of a raw exception.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from openai import OpenAI

from src.config import settings
from src.security.rls import SQLSecurityError, UserProfile, apply_rls
from src.utils import format_sql_results, get_logger

logger = get_logger("agent.sql")


def _introspect_schema(db_path: Path) -> str:
    """
    Build the schema description by reading PRAGMA table_info from the
    real database — no hardcoded column names that can drift from reality.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        lines = ["Database: staffing.db (SQLite, read-only)", "", "Tables:"]
        for (table,) in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            lines.append(f"\n{table}(")
            for col in cols:
                cid, name, col_type, notnull, default, pk = col
                pk_note = " PRIMARY KEY" if pk else ""
                lines.append(f"    {name:<20} {col_type}{pk_note},")
            lines.append(")")
        conn.close()
        return "\n".join(lines)
    except Exception as exc:
        logger.warning(f"Schema introspection failed, using fallback: {exc}")
        return _FALLBACK_SCHEMA


_FALLBACK_SCHEMA = """
Database: staffing.db (SQLite, read-only)
Tables: jobs, candidates, placements
(schema unavailable — avoid column-specific queries)
""".strip()


def _build_system_prompt(db_path: Path) -> str:
    schema = _introspect_schema(db_path)
    return f"""You are a SQL expert for a staffing operations platform.

{schema}

Rules:
- Generate ONLY a SELECT statement — no INSERT, UPDATE, DELETE, or DDL.
- Return the SQL query inside a ```sql ... ``` code block and nothing else.
- IMPORTANT: always use the exact column names from the schema above.
- Use table aliases when joining (e.g. j for jobs, c for candidates, p for placements).
- When using aliases, reference columns with the alias (e.g. j.title, not jobs.title).
- For date comparisons use SQLite strftime or date() functions.
- Current date reference: use DATE('now').
- Do NOT add LIMIT unless the user explicitly asks for top-N results.
- If the question cannot be answered with the available schema, respond with:
  NO_SQL: <brief explanation>
"""


_MSG_SQL_ERROR = (
    "I'm sorry, I wasn't able to retrieve that information right now. "
    "This might be a temporary issue — please try rephrasing your question "
    "or ask something different."
)

_MSG_RLS_ERROR = (
    "Your query could not be completed because it references data "
    "outside your permitted regions or divisions. "
    "Please adjust your question and try again."
)

_MSG_LLM_ERROR = (
    "We're having trouble reaching the AI service at the moment. "
    "Please try again in a few seconds."
)

_MSG_NO_RESULTS = (
    "No results found for your query within your accessible regions ({regions}). "
    "The data you're looking for may not exist or may be outside your access scope."
)


class SQLAgent:
    """Translates natural language questions into SQL and executes them."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        default_headers = None
        if "openrouter.ai" in settings.openrouter_base_url:
            default_headers = {
                "HTTP-Referer": settings.openrouter_app_url,
                "X-Title": settings.openrouter_app_name,
            }
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=default_headers,
        )
        self._system_prompt = _build_system_prompt(self._db_path)

    def query(self, question: str, user: UserProfile) -> dict:
        """
        Process a natural language question and return the answer.

        Args:
            question: User's natural language question.
            user:     User profile for RLS application.

        Returns:
            Dict with keys: answer (str), sql (str | None), rows (list[dict])
            Never raises — errors are returned as friendly answer strings.
        """
        logger.info(f"SQL query for '{user.username}': {question!r}")

        try:
            raw_sql = self._generate_sql(question, user)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return {"answer": _MSG_LLM_ERROR, "sql": None, "rows": []}

        if raw_sql.startswith("NO_SQL:"):
            explanation = raw_sql.removeprefix("NO_SQL:").strip()
            return {"answer": explanation, "sql": None, "rows": []}

        try:
            secured_sql = apply_rls(raw_sql, user)
        except SQLSecurityError as exc:
            logger.warning(f"RLS blocked SQL: {exc}")
            return {"answer": _MSG_RLS_ERROR, "sql": None, "rows": []}

        logger.debug(f"SQL with RLS:\n{secured_sql}")

        try:
            rows = self._execute(secured_sql)
        except Exception as exc:
            logger.error(f"SQL execution error: {exc}\nSQL: {secured_sql}")
            return {"answer": _MSG_SQL_ERROR, "sql": secured_sql, "rows": []}

        try:
            answer = self._format_answer(question, rows, secured_sql, user)
        except Exception as exc:
            logger.error(f"Answer formatting failed: {exc}")
            answer = format_sql_results(rows) if rows else _MSG_NO_RESULTS.format(
                regions=", ".join(user.regions)
            )

        return {"answer": answer, "sql": secured_sql, "rows": rows}

    def _generate_sql(self, question: str, user: UserProfile) -> str:
        """Call the LLM to generate the SQL query."""
        user_context = (
            f"Current user permissions: {user.permissions_summary()}\n"
            "Note: security filters will be applied automatically — "
            "do NOT add region/division filters yourself."
        )

        response = self._client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"{user_context}\n\nQuestion: {question}"},
            ],
            temperature=0,
            max_tokens=512,
        )

        content = response.choices[0].message.content or ""
        return self._extract_sql(content)

    def _extract_sql(self, llm_output: str) -> str:
        """
        Extract the SQL query from the LLM response.

        Supports:
          - ```sql ... ``` code blocks
          - ``` ... ``` plain code blocks
          - Raw SQL without delimiters
          - NO_SQL: <message>
        """
        if llm_output.strip().upper().startswith("NO_SQL"):
            return llm_output.strip()

        match = re.search(r"```(?:sql)?\s*(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return llm_output.strip()

    def _execute(self, sql: str) -> list[dict]:
        """
        Execute the query against SQLite in read-only mode.

        Returns:
            List of dicts {column: value}.

        Raises:
            RuntimeError: On any SQLite error (caught by caller).
        """
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Query returned {len(rows)} rows")
            return rows
        except sqlite3.Error as exc:
            logger.error(f"Error executing SQL: {exc}\nSQL: {sql}")
            raise RuntimeError(f"Error executing query: {exc}") from exc
        finally:
            conn.close()

    def _format_answer(
        self,
        question: str,
        rows: list[dict],
        sql: str,
        user: UserProfile,
    ) -> str:
        """Use the LLM to format raw results into a natural language answer."""
        if not rows:
            return _MSG_NO_RESULTS.format(regions=", ".join(user.regions))

        table_str = format_sql_results(rows, max_rows=20)

        prompt = (
            f"The user asked: {question!r}\n\n"
            f"SQL executed (with security filters for {user.permissions_summary()}):\n"
            f"```sql\n{sql}\n```\n\n"
            f"Results ({len(rows)} rows):\n{table_str}\n\n"
            "Provide a concise, friendly answer in the user's language. "
            "Mention the region scope naturally. Do not repeat the SQL."
        )

        response = self._client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )

        return (response.choices[0].message.content or "").strip()
