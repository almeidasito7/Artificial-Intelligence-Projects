"""Unit tests for the semantic matcher (HuggingFace / OpenAI) using mocks."""
from __future__ import annotations
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from app.matchers.semantic import semantic_match
from app.models import CandidateLink
from tests.conftest import unit_vec, similar_vec


DIMS = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_embed(query_vec, *candidate_vecs):
    """Returns an AsyncMock that yields [query_vec, *candidate_vecs]."""
    return AsyncMock(return_value=[query_vec, *candidate_vecs])


# ---------------------------------------------------------------------------
# Happy path — high similarity match
# ---------------------------------------------------------------------------

class TestSemanticMatchHappyPath:
    @pytest.mark.asyncio
    async def test_returns_best_match(self, link_member_portal, link_listings):
        q = unit_vec(DIMS, seed=1)
        mp_vec = similar_vec(q, similarity=0.90, seed=2)   # Member Portal — close
        lh_vec = similar_vec(q, similarity=0.20, seed=3)   # Listings — far

        with patch("app.matchers.semantic._embed_hf", make_mock_embed(q, mp_vec, lh_vec)):
            matched, score, strategy, llm_calls, fallback = await semantic_match(
                "reset membership password",
                [link_member_portal, link_listings],
                threshold=0.40,
            )

        assert matched is not None
        assert matched.label == "Member Portal"
        assert score > 0.40
        assert strategy == "semantic"
        assert llm_calls == 1
        assert fallback == ""

    @pytest.mark.asyncio
    async def test_score_reflects_cosine(self, link_member_portal, link_listings):
        """Higher similarity input must produce higher cosine score than lower similarity input."""
        q = unit_vec(DIMS, seed=10)
        high_vec = similar_vec(q, similarity=0.90, seed=11)  # closer to query
        low_vec  = similar_vec(q, similarity=0.20, seed=12)  # farther from query

        with patch("app.matchers.semantic._embed_hf", make_mock_embed(q, high_vec, low_vec)):
            matched, score, *_ = await semantic_match(
                "reset password",
                [link_member_portal, link_listings],
                threshold=0.0,
            )

        assert matched is not None
        assert matched.label == "Member Portal"  # high_vec candidate wins
        assert score > 0.5  # comfortably above any reasonable threshold


# ---------------------------------------------------------------------------
# Below threshold — no match
# ---------------------------------------------------------------------------

class TestSemanticMatchBelowThreshold:
    @pytest.mark.asyncio
    async def test_below_threshold_returns_none(self, link_member_portal):
        q = unit_vec(DIMS, seed=20)
        mp_vec = similar_vec(q, similarity=0.30, seed=21)  # below 0.45

        with patch("app.matchers.semantic._embed_hf", make_mock_embed(q, mp_vec)):
            matched, score, strategy, llm_calls, fallback = await semantic_match(
                "pool hours", [link_member_portal], threshold=0.45
            )

        assert matched is None
        assert score < 0.45
        assert strategy == "semantic"
        assert llm_calls == 1


# ---------------------------------------------------------------------------
# Empty candidates — no API call
# ---------------------------------------------------------------------------

class TestSemanticMatchEmptyCandidates:
    @pytest.mark.asyncio
    async def test_empty_list_no_api_call(self):
        mock_hf = AsyncMock()
        with patch("app.matchers.semantic._embed_hf", mock_hf):
            matched, score, strategy, llm_calls, fallback = await semantic_match(
                "reset password", [], threshold=0.45
            )

        mock_hf.assert_not_called()
        assert matched is None
        assert score == 0.0
        assert llm_calls == 0


# ---------------------------------------------------------------------------
# Fallback on embedding API error
# ---------------------------------------------------------------------------

class TestSemanticMatchFallback:
    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_keyword(self, link_member_portal):
        import httpx
        mock_hf = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("app.matchers.semantic._embed_hf", mock_hf):
            matched, score, strategy, llm_calls, fallback = await semantic_match(
                "How do I reset my membership password?",
                [link_member_portal],
                threshold=0.05,
            )

        assert strategy == "keyword_fallback"
        assert llm_calls == 0
        assert "TimeoutException" in fallback or "timed out" in fallback.lower() or "fell back" in fallback

    @pytest.mark.asyncio
    async def test_connection_error_falls_back(self, link_member_portal):
        import httpx
        mock_hf = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with patch("app.matchers.semantic._embed_hf", mock_hf):
            _, _, strategy, llm_calls, fallback = await semantic_match(
                "reset password", [link_member_portal], threshold=0.05
            )

        assert strategy == "keyword_fallback"
        assert "fell back to keyword" in fallback

    @pytest.mark.asyncio
    async def test_fallback_does_not_raise(self, link_member_portal):
        """The endpoint must never propagate an embedding exception."""
        mock_hf = AsyncMock(side_effect=Exception("unexpected boom"))

        with patch("app.matchers.semantic._embed_hf", mock_hf):
            result = await semantic_match("reset password", [link_member_portal])

        assert result is not None  # returned a tuple, did not raise


# ---------------------------------------------------------------------------
# Provider switching — OpenAI path
# ---------------------------------------------------------------------------

class TestSemanticMatchOpenAI:
    @pytest.mark.asyncio
    async def test_openai_provider_called(self, link_member_portal):
        q = unit_vec(DIMS, seed=30)
        mp_vec = similar_vec(q, similarity=0.85, seed=31)

        mock_openai = AsyncMock(return_value=[q, mp_vec])
        with (
            patch("app.matchers.semantic._embed_openai", mock_openai),
            patch("app.matchers.semantic.config.LLM_PROVIDER", "openai"),
        ):
            matched, score, strategy, llm_calls, _ = await semantic_match(
                "reset password", [link_member_portal], threshold=0.40
            )

        mock_openai.assert_called_once()
        assert matched is not None
        assert strategy == "semantic"
        assert llm_calls == 1


# ---------------------------------------------------------------------------
# Disambiguation — semantic picks the right one when keywords are ambiguous
# ---------------------------------------------------------------------------

class TestSemanticDisambiguation:
    @pytest.mark.asyncio
    async def test_semantic_disambiguates_ambiguous_keyword(
        self, link_member_portal, link_billing
    ):
        """Both candidates share 'account'. Semantics must pick Billing Help."""
        q = unit_vec(DIMS, seed=40)
        mp_vec = similar_vec(q, similarity=0.45, seed=41)   # Member Portal — weaker
        billing_vec = similar_vec(q, similarity=0.85, seed=42)  # Billing — stronger

        with patch(
            "app.matchers.semantic._embed_hf",
            make_mock_embed(q, mp_vec, billing_vec),
        ):
            matched, score, *_ = await semantic_match(
                "I need help with my invoice and billing account",
                [link_member_portal, link_billing],
                threshold=0.40,
            )

        assert matched is not None
        assert matched.label == "Billing Help"
