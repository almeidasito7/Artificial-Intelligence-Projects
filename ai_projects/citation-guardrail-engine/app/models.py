from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Grounding(BaseModel):
    is_grounded: bool
    kb_grounded: bool


class CandidateLink(BaseModel):
    label: str
    url: str
    keywords: List[str]
    description: str


class GuardrailRequest(BaseModel):
    query: str
    llm_answer: str
    grounding: Grounding
    is_chitchat: bool
    candidate_links: List[CandidateLink]


class CitationDecision(BaseModel):
    status: str
    matched_label: Optional[str] = None
    strategy_used: Optional[str] = None
    similarity_score: Optional[float] = None
    reason: str


class Metrics(BaseModel):
    latency_ms: float
    llm_calls: int


class GuardrailResponse(BaseModel):
    final_answer: str
    citation_decision: CitationDecision
    metrics: Metrics
