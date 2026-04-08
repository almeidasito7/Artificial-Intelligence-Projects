"""
Agent Router — Central orchestrator

Responsibilities:
  1. Check the semantic cache before any LLM call
  2. Detect unauthorized region/division references and deny access explicitly
  3. Classify the query as SQL (structured data) or RAG (documents)
  4. Dispatch to the correct agent
  5. Store the result in the cache
  6. Return a standardised AgentResponse

Classification uses a lightweight LLM call to avoid brittle keyword heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

from src.cache.semantic_cache import get_cache
from src.config import settings
from src.security.rls import UserProfile
from src.utils import get_logger

if TYPE_CHECKING:
    from src.agent.rag_agent import RAGAgent
    from src.agent.sql_agent import SQLAgent

logger = get_logger("agent.router")

_ROUTER_PROMPT = """You are a query classifier for a staffing BI assistant.

Classify the user's question into one of two categories:
- SQL: questions about data (jobs, candidates, placements, counts, rates, dates, stats)
- RAG: questions about policies, SOPs, procedures, guidelines, or how things work

Respond with ONLY the word SQL or RAG — nothing else."""


@dataclass
class AgentResponse:
    """Standardised agent response, independent of the route taken."""

    answer: str
    source: str | None = None
    cache_hit: bool = False
    similarity_score: float | None = None
    route: str | None = None  # "sql" | "rag" | "denied"


_KNOWN_REGIONS = {
    "southeast", "west coast", "northeast", "midwest", "southwest",
}


def _detect_unauthorized_regions(query: str, user: UserProfile) -> list[str]:
    """
    Detect region names mentioned in the query that are outside the user's
    permitted regions.

    Returns a list of unauthorised region names found in the query.
    Returns an empty list if the query only mentions permitted regions or
    no regions at all.
    """
    query_lower = query.lower()
    user_regions_lower = {r.lower() for r in user.regions}

    unauthorized = []
    for region in _KNOWN_REGIONS:
        if region in query_lower and region not in user_regions_lower:
            unauthorized.append(region.title())

    return unauthorized


def _access_denied_message(regions: list[str], user: UserProfile) -> str:
    """Return a clear, friendly access denial message."""
    region_list = ", ".join(regions)
    permitted = ", ".join(user.regions) if user.regions else "none"
    return (
        f"Access denied: you do not have permission to view data for "
        f"{region_list}. "
        f"Your access is limited to: {permitted}."
    )


class AgentRouter:
    """Orchestrates the full flow: cache → access check → classify → execute → store."""

    def __init__(self) -> None:
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
        self._cache = get_cache()

        self._sql_agent: SQLAgent | None = None
        self._rag_agent: RAGAgent | None = None

    def route(self, query: str, user: UserProfile) -> AgentResponse:
        """
        Process a query and return the response.

        Flow:
          cache hit      → return cached response
          access denied  → return denial message (not cached)
          cache miss     → classify → execute → store in cache → return

        Args:
            query: Natural language question.
            user:  User profile (for RLS and cache namespace).
        """
        cached = self._cache.get(
            query=query,
            regions=user.regions,
            divisions=user.divisions,
        )
        if cached is not None:
            response_dict, score = cached
            return AgentResponse(
                answer=response_dict["answer"],
                source=response_dict.get("source"),
                cache_hit=True,
                similarity_score=score,
                route=response_dict.get("route"),
            )

        unauthorized = _detect_unauthorized_regions(query, user)
        if unauthorized:
            logger.warning(
                f"Access denied for '{user.username}': "
                f"query references unauthorized regions {unauthorized}"
            )
            return AgentResponse(
                answer=_access_denied_message(unauthorized, user),
                source=None,
                cache_hit=False,
                route="denied",
            )

        route = self._classify(query)
        logger.info(f"Query classified as: {route.upper()}")

        if route == "rag":
            response = self._run_rag(query)
        else:
            response = self._run_sql(query, user)

        self._cache.set(
            query=query,
            regions=user.regions,
            divisions=user.divisions,
            response={
                "answer": response.answer,
                "source": response.source,
                "route": response.route,
            },
        )

        return response

    def _classify(self, query: str) -> str:
        """
        Classify the query as 'sql' or 'rag' via LLM.
        Falls back to 'sql' on any unexpected response or error.
        """
        try:
            resp = self._client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": _ROUTER_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=5,
            )
            label = (resp.choices[0].message.content or "SQL").strip().upper()
            return "rag" if "RAG" in label else "sql"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Classification failed ({exc}), falling back to SQL")
            return "sql"

    def _run_sql(self, query: str, user: UserProfile) -> AgentResponse:
        if self._sql_agent is None:
            from src.agent.sql_agent import SQLAgent
            self._sql_agent = SQLAgent()

        result = self._sql_agent.query(question=query, user=user)
        return AgentResponse(
            answer=result["answer"],
            source="SQL query" if result.get("sql") else None,
            cache_hit=False,
            route="sql",
        )

    def _run_rag(self, query: str) -> AgentResponse:
        if self._rag_agent is None:
            from src.agent.rag_agent import RAGAgent
            self._rag_agent = RAGAgent()

        result = self._rag_agent.query(question=query)
        sources = result.get("sources", [])
        return AgentResponse(
            answer=result["answer"],
            source=", ".join(sources) if sources else None,
            cache_hit=False,
            route="rag",
        )
