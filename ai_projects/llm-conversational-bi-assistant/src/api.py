"""
FastAPI — Web application interface.

Endpoints:
  POST /chat    — Send a question and receive the agent's response
  GET  /health  — System status (database, vector store, cache)
  GET  /users   — Lists the available users
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.router import AgentResponse, AgentRouter
from src.cache.semantic_cache import get_cache
from src.config import settings
from src.security.rls import list_users, load_user
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger("api")

app = FastAPI(
    title="Conversational BI Assistant",
    description="Text-to-SQL + RAG + RLS + Semantic Cache",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_router: AgentRouter | None = None


def get_router() -> AgentRouter:
    global _router
    if _router is None:
        _router = AgentRouter()
    return _router


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question in natural language")
    username: str = Field(..., min_length=1, description="User name (must exist in user_permissions.json)")


class ChatResponse(BaseModel):
    answer: str
    source: str | None
    cache_hit: bool
    similarity_score: float | None
    route: str | None
    username: str
    regions: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    router: Annotated[AgentRouter, Depends(get_router)],
) -> ChatResponse:
    """Processes a question and returns the agent's response."""
    try:
        user = load_user(request.username)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        response: AgentResponse = router.route(query=request.query, user=user)
    except Exception as exc:
        logger.error(f"Error processing query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        ) from exc

    return ChatResponse(
        answer=response.answer,
        source=response.source,
        cache_hit=response.cache_hit,
        similarity_score=response.similarity_score,
        route=response.route,
        username=user.username,
        regions=user.regions,
    )


@app.get("/health")
def health() -> dict:
    """Returns the status of the system."""
    db_ok = False
    db_jobs_count = 0
    try:
        conn = sqlite3.connect(str(settings.db_path))
        db_jobs_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        db_ok = True
    except Exception as exc:
        logger.warning(f"Health check — DB error: {exc}")

    cache_stats = get_cache().stats()

    rag_chunks = None
    if _router and _router._rag_agent:
        rag_chunks = _router._rag_agent.document_count()

    return {
        "status": "ok" if db_ok else "degraded",
        "database": {
            "ok": db_ok,
            "jobs_count": db_jobs_count,
            "path": str(settings.db_path),
        },
        "vector_store": {
            "chunks_ingested": rag_chunks,
            "path": str(settings.chroma_persist_path),
        },
        "cache": cache_stats,
        "model": settings.openrouter_model,
    }


@app.get("/users")
def users() -> dict:
    """Lists the available users in the system."""
    return {"users": list_users()}
