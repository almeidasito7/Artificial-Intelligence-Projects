"""Unit tests for the Semantic Cache."""

from __future__ import annotations

import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache

_VEC_A = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
_VEC_A /= np.linalg.norm(_VEC_A)

_VEC_A2 = np.array([1.0, 0.15, 0.0, 0.0], dtype=np.float32)
_VEC_A2 /= np.linalg.norm(_VEC_A2)

_VEC_B = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

assert np.dot(_VEC_A, _VEC_A2) > 0.92, "Test setup: similar vecs must be > threshold"
assert np.dot(_VEC_A, _VEC_B) < 0.10, "Test setup: different vecs must be near 0"


def _make_encode(mapping: dict[str, np.ndarray]):
    """
    Return a mock encode function that maps specific query strings to fixed
    vectors, falling back to _VEC_B for anything not in the mapping.
    """
    def _encode(text: str, normalize_embeddings: bool = True) -> np.ndarray:
        return mapping.get(text, _VEC_B).copy()
    return _encode

@pytest.fixture
def cache() -> Iterator[SemanticCache]:
    """Cache with a mocked embedding model — no GPU/download needed."""
    with patch("src.cache.semantic_cache._get_embedding_model") as mock_factory:
        model = MagicMock()
        model.encode.side_effect = lambda text, normalize_embeddings=True: _VEC_A.copy()
        mock_factory.return_value = model
        yield SemanticCache(
            similarity_threshold=0.92,
            ttl_seconds=300,
            max_entries_per_user=10,
        )


REGIONS_SE = ["Southeast"]
DIVISIONS_IT = ["IT"]
REGIONS_WC = ["West Coast"]

_RESPONSE = {"answer": "There are 23 open jobs.", "source": None, "route": "sql"}


class TestCacheSetAndGet:
    def test_exact_query_hit(self, cache):
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        result = cache.get("how many open jobs", REGIONS_SE, DIVISIONS_IT)
        assert result is not None
        response, score = result
        assert response["answer"] == _RESPONSE["answer"]
        assert score >= 0.92

    def test_similar_query_hit(self, cache):
        """Semantically similar queries must produce a cache hit."""
        encode_calls = []

        def _side_effect(text, normalize_embeddings=True):
            encode_calls.append(text)
            if len(encode_calls) == 1:
                return _VEC_A.copy()
            return _VEC_A2.copy()

        with patch("src.cache.semantic_cache._get_embedding_model") as mock_factory:
            model = MagicMock()
            model.encode.side_effect = _side_effect
            mock_factory.return_value = model

            fresh_cache = SemanticCache(
                similarity_threshold=0.92,
                ttl_seconds=300,
                max_entries_per_user=10,
            )
            fresh_cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
            result = fresh_cache.get("count open positions", REGIONS_SE, DIVISIONS_IT)

        assert result is not None, "Similar query should result in a cache hit"
        _, score = result
        assert score >= 0.92

    def test_different_query_miss(self, cache):
        """Orthogonal query must produce a cache miss."""
        with patch("src.cache.semantic_cache._get_embedding_model") as mock_factory:
            model = MagicMock()
            call_count = [0]

            def _side_effect(text, normalize_embeddings=True):
                call_count[0] += 1
                if call_count[0] == 1:
                    return _VEC_A.copy()
                return _VEC_B.copy()

            model.encode.side_effect = _side_effect
            mock_factory.return_value = model

            fresh_cache = SemanticCache(
                similarity_threshold=0.92,
                ttl_seconds=300,
                max_entries_per_user=10,
            )
            fresh_cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
            result = fresh_cache.get("background check policy", REGIONS_SE, DIVISIONS_IT)

        assert result is None, "Different query should not hit the cache"

    def test_empty_cache_returns_none(self, cache):
        result = cache.get("anything", REGIONS_SE, DIVISIONS_IT)
        assert result is None


class TestCacheIsolation:
    def test_different_permissions_no_crossover(self, cache):
        """User with different regions must NOT receive another user's cache."""
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        result = cache.get("how many open jobs", REGIONS_WC, DIVISIONS_IT)
        assert result is None

    def test_same_permissions_different_username_shares_cache(self, cache):
        """
        Two users with identical permissions share the cache namespace.
        The namespace key is based on the permission hash, not the username.
        """
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        result = cache.get("how many open jobs", REGIONS_SE, DIVISIONS_IT)
        assert result is not None

    def test_superset_permissions_isolated(self, cache):
        """Admin with more regions must not receive a restricted user's cache."""
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        result = cache.get(
            "how many open jobs",
            ["Southeast", "West Coast"],
            DIVISIONS_IT,
        )
        assert result is None


class TestCacheTTL:
    def test_expired_entry_not_returned(self, cache):
        """Expired entries must be ignored during lookup."""
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)

        key = cache._cache_key(REGIONS_SE, DIVISIONS_IT)
        for entry in cache._store[key].values():
            entry.created_at = time.time() - 99999

        result = cache.get("how many open jobs", REGIONS_SE, DIVISIONS_IT)
        assert result is None

    def test_non_expired_entry_returned(self, cache):
        cache.set("how many open jobs", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        result = cache.get("how many open jobs", REGIONS_SE, DIVISIONS_IT)
        assert result is not None


class TestCacheLRU:
    def test_max_entries_respected(self, cache):
        """Bucket size must never exceed max_entries_per_user."""
        for i in range(15):
            cache.set(
                f"unique query {i}",
                REGIONS_SE, DIVISIONS_IT,
                {"answer": f"answer {i}", "source": None, "route": "sql"},
            )
        key = cache._cache_key(REGIONS_SE, DIVISIONS_IT)
        assert len(cache._store[key]) <= 10


class TestCacheUtils:
    def test_stats_returns_expected_keys(self, cache):
        stats = cache.stats()
        for key in ("namespaces", "total_entries", "active_entries", "threshold", "ttl_seconds"):
            assert key in stats

    def test_clear_removes_all_entries(self, cache):
        cache.set("q1", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        cache.set("q2", REGIONS_WC, DIVISIONS_IT, _RESPONSE)
        cache.clear()
        assert cache.stats()["total_entries"] == 0

    def test_invalidate_user_removes_namespace(self, cache):
        cache.set("q1", REGIONS_SE, DIVISIONS_IT, _RESPONSE)
        count = cache.invalidate_user(REGIONS_SE, DIVISIONS_IT)
        assert count >= 1
        assert cache.get("q1", REGIONS_SE, DIVISIONS_IT) is None
