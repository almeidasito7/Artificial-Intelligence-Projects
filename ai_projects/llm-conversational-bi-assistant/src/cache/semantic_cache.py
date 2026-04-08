"""
Semantic Cache

Cache in memory that detects semantically similar queries and avoids
unnecessary LLM calls.

Properties:
  - Similarity via cosine embeddings (same model as the RAG)
  - TTL configurable per entry
  - User isolation: the cache key includes hash of permissions,
    ensuring that User A never receives a cached response from User B
  - Limit of entries per user (LRU via OrderedDict)
  - Thread-safe for use with FastAPI (asyncio single-thread + GIL)

Design decision: cache in memory (not Redis/disk) because:
  1. Simplicity — zero extra dependencies for the assessment
  2. The default TTL is short (5 min), so persistence has limited value
  3. For production, swap the backend to Redis maintaining the same interface
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import get_logger, hash_permissions

logger = get_logger("cache")

_EMBEDDING_MODEL: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model for cache...")
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


@dataclass
class CacheEntry:
    """A cache entry."""

    query: str
    embedding: np.ndarray
    response: Any                    # AgentResponse serialized as dict
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300


    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticCache:
    """
    Semantic cache with user isolation and TTL.

    Usage:
        cache = SemanticCache()

        hit = cache.get(
            query="open jobs?",
            regions=["Southeast"],
            divisions=["IT"],
        )
        if hit:
            return hit

        response = llm_call(...)
        cache.set(
            query="open jobs?",
            regions=["Southeast"],
            divisions=["IT"],
            response=response,
        )
    """

    def __init__(
        self,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_entries_per_user: int | None = None,
    ) -> None:
        from src.config import settings

        self._threshold = similarity_threshold or settings.cache_similarity_threshold
        self._ttl = ttl_seconds or settings.cache_ttl_seconds
        self._max_per_user = max_entries_per_user or settings.cache_max_entries_per_user

        self._store: dict[str, OrderedDict[int, CacheEntry]] = {}
        self._counter = 0

        logger.info(
            f"SemanticCache initialized: "
            f"threshold={self._threshold}, ttl={self._ttl}s, "
            f"max_per_user={self._max_per_user}"
        )

    def _cache_key(self, regions: list[str], divisions: list[str]) -> str:
        """
        Generates the cache namespace key for a set of permissions.

        Two users with the same permissions share cache;
        users with different permissions never share cache.
        """
        return hash_permissions(regions, divisions)

    def get(
        self,
        query: str,
        regions: list[str],
        divisions: list[str],
    ) -> tuple[Any, float] | None:
        """
        Searches for a semantically similar cached response.

        Args:
            query:     Query from the user.
            regions:   Regions of the user (for cache namespace).
            divisions: Divisions of the user (for cache namespace).

        Returns:
            Tuple (response, similarity_score) if cache hit, None otherwise.
        """
        key = self._cache_key(regions, divisions)
        bucket = self._store.get(key)

        if not bucket:
            logger.debug("Cache miss: namespace empty")
            return None

        model = _get_embedding_model()
        query_emb = np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)

        best_score = 0.0
        best_entry: CacheEntry | None = None
        expired_ids: list[int] = []

        for entry_id, entry in bucket.items():
            if entry.is_expired:
                expired_ids.append(entry_id)
                continue

            score = float(np.dot(query_emb, entry.embedding))

            if score > best_score:
                best_score = score
                best_entry = entry

        for eid in expired_ids:
            del bucket[eid]
            logger.debug(f"Cache: expired entry removed (id={eid})")

        if best_entry and best_score >= self._threshold:
            logger.info(
                f"Cache HIT (similarity={best_score:.4f}, "
                f"threshold={self._threshold}, user_ns={key[:8]})"
            )
            return best_entry.response, best_score

        logger.debug(
            f"Cache MISS (best_score={best_score:.4f} < {self._threshold})"
        )
        return None

    def set(
        self,
        query: str,
        regions: list[str],
        divisions: list[str],
        response: Any,
    ) -> None:
        """
        Stores a response in the cache.

        Args:
            query:     Original query from the user.
            regions:   Regions of the user.
            divisions: Divisions of the user.
            response:  Response from the agent to be cached.
        """
        key = self._cache_key(regions, divisions)

        if key not in self._store:
            self._store[key] = OrderedDict()

        bucket = self._store[key]

        while len(bucket) >= self._max_per_user:
            oldest_id, _ = next(iter(bucket.items()))
            del bucket[oldest_id]
            logger.debug(f"Cache: LRU eviction (id={oldest_id}, user_ns={key[:8]})")

        model = _get_embedding_model()
        embedding = np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)

        self._counter += 1
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            ttl_seconds=self._ttl,
        )
        bucket[self._counter] = entry

        logger.debug(
            f"Cache SET: entry_id={self._counter}, "
            f"user_ns={key[:8]}, bucket_size={len(bucket)}"
        )

    def invalidate_user(self, regions: list[str], divisions: list[str]) -> int:
        """Removes all entries from a user namespace. Returns count."""
        key = self._cache_key(regions, divisions)
        bucket: OrderedDict[int, CacheEntry] = self._store.pop(key, OrderedDict())
        count = len(bucket)
        logger.info(f"Cache invalidated: {count} entries removed (ns={key[:8]})")
        return count

    def clear(self) -> None:
        """Clears the entire cache."""
        total = sum(len(b) for b in self._store.values())
        self._store.clear()
        logger.info(f"Cache cleared: {total} entries removed")

    def stats(self) -> dict:
        """Returns cache statistics for the /health endpoint."""
        namespaces = len(self._store)
        total_entries = sum(len(b) for b in self._store.values())
        active_entries = sum(
            sum(1 for e in b.values() if not e.is_expired)
            for b in self._store.values()
        )
        return {
            "namespaces": namespaces,
            "total_entries": total_entries,
            "active_entries": active_entries,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
        }

_cache_instance: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Returns the singleton instance of the cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
