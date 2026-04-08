"""Unit tests for the AgentRouter."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from src.agent.router import AgentRouter
from src.security.rls import UserProfile


@pytest.fixture
def user_se() -> UserProfile:
    return UserProfile(username="alice", regions=["Southeast"], divisions=["IT"])


@pytest.fixture
def mock_router(user_se) -> Iterator[AgentRouter]:
    """Router with LLM and agents fully mocked."""
    with patch("src.agent.router.OpenAI"):
        router = AgentRouter()
        router._cache = MagicMock()
        router._cache.get.return_value = None
        router._cache.set.return_value = None
        yield router


class TestRouterClassification:
    def test_routes_data_query_to_sql(self, mock_router, user_se):
        mock_router._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SQL"))]
        )

        mock_router._sql_agent = MagicMock()
        mock_router._sql_agent.query.return_value = {
            "answer": "There are 23 open jobs.",
            "sql": "SELECT COUNT(*) FROM jobs",
            "rows": [{"count": 23}],
        }

        response = mock_router.route("how many open jobs?", user_se)

        assert response.route == "sql"
        assert "23" in response.answer
        mock_router._sql_agent.query.assert_called_once()

    def test_routes_policy_query_to_rag(self, mock_router, user_se):
        mock_router._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="RAG"))]
        )

        mock_router._rag_agent = MagicMock()
        mock_router._rag_agent.query.return_value = {
            "answer": "Contractors must submit requests 5 days in advance.",
            "sources": ["policy_contractor.md"],
            "chunks_used": 2,
        }

        response = mock_router.route("what is the contractor time-off policy?", user_se)

        assert response.route == "rag"
        assert "policy_contractor.md" in (response.source or "")
        mock_router._rag_agent.query.assert_called_once()

    def test_cache_hit_skips_llm(self, mock_router, user_se):
        """When there is a cache hit, the LLM must not be called."""
        mock_router._cache.get.return_value = (
            {"answer": "Cached answer.", "source": None, "route": "sql"},
            0.95,
        )

        response = mock_router.route("how many open jobs?", user_se)

        assert response.cache_hit is True
        assert response.similarity_score == 0.95
        assert response.answer == "Cached answer."
        mock_router._client.chat.completions.create.assert_not_called()

    def test_response_stored_in_cache_after_llm_call(self, mock_router, user_se):
        """Response must be stored in cache after LLM call."""
        mock_router._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SQL"))]
        )
        mock_router._sql_agent = MagicMock()
        mock_router._sql_agent.query.return_value = {
            "answer": "12 placements.",
            "sql": "SELECT COUNT(*) FROM placements",
            "rows": [],
        }

        mock_router.route("how many placements?", user_se)

        mock_router._cache.set.assert_called_once()
        call_kwargs = mock_router._cache.set.call_args
        assert call_kwargs.kwargs["regions"] == user_se.regions

    def test_classifier_fallback_on_error(self, mock_router, user_se):
        """If the classifier fails, it must fall back to SQL (safe default)."""
        mock_router._client.chat.completions.create.side_effect = Exception("API timeout")

        mock_router._sql_agent = MagicMock()
        mock_router._sql_agent.query.return_value = {
            "answer": "Fallback answer.",
            "sql": "SELECT 1",
            "rows": [],
        }

        response = mock_router.route("some query", user_se)
        assert response.route == "sql"


class TestAccessDenial:
    def test_unauthorized_region_is_denied(self, mock_router, user_se):
        """Query mentioning a region outside user permissions must be denied without LLM call."""
        response = mock_router.route("How many jobs in the West Coast?", user_se)

        assert response.route == "denied"
        assert "Access denied" in response.answer
        assert "West Coast" in response.answer
        assert "Southeast" in response.answer  # permitted region shown
        mock_router._client.chat.completions.create.assert_not_called()

    def test_authorized_region_is_not_denied(self, mock_router, user_se):
        """Query about a permitted region must NOT trigger access denial."""
        mock_router._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SQL"))]
        )
        mock_router._sql_agent = MagicMock()
        mock_router._sql_agent.query.return_value = {
            "answer": "23 open jobs.", "sql": "SELECT 1", "rows": [],
        }

        response = mock_router.route("How many jobs in the Southeast?", user_se)
        assert response.route != "denied"

    def test_denial_is_not_cached(self, mock_router, user_se):
        """Denied responses must not be stored in the cache."""
        mock_router.route("How many jobs in the Northeast?", user_se)
        mock_router._cache.set.assert_not_called()
