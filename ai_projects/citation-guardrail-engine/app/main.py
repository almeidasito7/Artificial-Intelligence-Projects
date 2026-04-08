from __future__ import annotations
import time
from collections import defaultdict

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
from app.models import GuardrailRequest, GuardrailResponse, Metrics
from app.rules import check_early_rules, apply_post_match_rules
from app.matchers.keyword import keyword_match
from app.matchers.semantic import semantic_match
from app import config

app = FastAPI(title="Citation Guardrail Engine", version="1.0.0")

# In-memory counters for /health
_counters: dict[str, int] = defaultdict(int)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "counters": dict(_counters)}


@app.post("/guardrail", response_model=GuardrailResponse)
async def guardrail(req: GuardrailRequest) -> GuardrailResponse:
    start = time.monotonic()
    llm_calls = 0

    # R1 / R2 / empty list — no embedding needed
    early = check_early_rules(req)
    if early is not None:
        _counters[early.status] += 1
        return GuardrailResponse(
            final_answer=req.llm_answer,
            citation_decision=early,
            metrics=Metrics(
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                llm_calls=0,
            ),
        )

    # Matching step
    if config.STRATEGY == "keyword":
        matched, score = keyword_match(
            req.query, req.candidate_links, threshold=config.SIMILARITY_THRESHOLD
        )
        strategy_used = "keyword"
        fallback_reason = ""
    elif config.STRATEGY == "hybrid":
        # Semantic first; keyword score used as tie-breaker / boost
        matched, score, strategy_used, llm_calls, fallback_reason = await semantic_match(
            req.query, req.candidate_links
        )
        if matched is None:
            kw_match, kw_score = keyword_match(
                req.query, req.candidate_links, threshold=config.SIMILARITY_THRESHOLD
            )
            if kw_match is not None:
                matched, score = kw_match, kw_score
                strategy_used = "hybrid_keyword_boost"
    else:
        # Default: semantic
        matched, score, strategy_used, llm_calls, fallback_reason = await semantic_match(
            req.query, req.candidate_links
        )

    # R3 / R4 / R5
    decision, final_answer = apply_post_match_rules(
        req, matched, score, strategy_used, fallback_reason
    )

    _counters[decision.status] += 1

    return GuardrailResponse(
        final_answer=final_answer,
        citation_decision=decision,
        metrics=Metrics(
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            llm_calls=llm_calls,
        ),
    )
