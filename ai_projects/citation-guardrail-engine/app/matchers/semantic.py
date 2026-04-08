from __future__ import annotations
import math
import httpx
from typing import Optional, Tuple

from app.models import CandidateLink
from app import config
from app.matchers.keyword import keyword_match


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _candidate_text(candidate: CandidateLink) -> str:
    return f"{candidate.label}. {candidate.description}. {' '.join(candidate.keywords)}"


def _mean_pool(token_embeddings: list[list[float]]) -> list[float]:
    if not token_embeddings:
        return []
    dims = len(token_embeddings[0])
    pooled = [0.0] * dims
    for tok in token_embeddings:
        for i, v in enumerate(tok):
            pooled[i] += float(v)
    n = float(len(token_embeddings))
    return [v / n for v in pooled]


def _coerce_embedding(obj) -> list[float]:
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return _mean_pool(obj)
    if isinstance(obj, list):
        return [float(x) for x in obj]
    raise TypeError(f"unexpected embedding payload type: {type(obj).__name__}")



async def _embed_hf(texts: list[str]) -> list[list[float]]:
    headers = {}
    if config.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_API_TOKEN}"

    async with httpx.AsyncClient(timeout=config.EMBEDDING_TIMEOUT_S) as client:
        resp = await client.post(
            config.HF_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(payload["error"])

        if isinstance(payload, list) and len(payload) == len(texts):
            return [_coerce_embedding(item) for item in payload]

        return [_coerce_embedding(payload)]


async def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    response = await client.embeddings.create(
        model=config.OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def semantic_match(
    query: str,
    candidates: list[CandidateLink],
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> Tuple[Optional[CandidateLink], float, str, int, str]:
    """
    Returns: (best_candidate, score, strategy_used, llm_calls, fallback_reason)
    strategy_used: "semantic" | "keyword_fallback"
    fallback_reason: empty string if no fallback occurred
    """
    if not candidates:
        return None, 0.0, "semantic", 0, ""

    texts = [query] + [_candidate_text(c) for c in candidates]

    try:
        if config.LLM_PROVIDER == "openai":
            vectors = await _embed_openai(texts)
        else:
            vectors = await _embed_hf(texts)

        query_vec = vectors[0]
        candidate_vecs = vectors[1:]

        best_link: Optional[CandidateLink] = None
        best_score: float = 0.0

        for candidate, vec in zip(candidates, candidate_vecs):
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best_link = candidate

        if best_score >= threshold:
            return best_link, best_score, "semantic", 1, ""
        return None, best_score, "semantic", 1, ""

    except Exception as exc:
        # Fallback to keyword on any embedding error
        fallback_reason = f"embedding_api_error ({type(exc).__name__}: {exc}), fell back to keyword"
        best_link, best_score = keyword_match(query, candidates, threshold=threshold)
        return best_link, best_score, "keyword_fallback", 0, fallback_reason
