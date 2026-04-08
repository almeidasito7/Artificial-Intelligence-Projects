"""Shared fixtures for the Citation Guardrail test suite."""
from __future__ import annotations
import math, random
import pytest
from app.models import CandidateLink, GuardrailRequest, Grounding


# ---------------------------------------------------------------------------
# Candidate link fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def link_member_portal() -> CandidateLink:
    return CandidateLink(
        label="Member Portal",
        url="https://example.com/members",
        keywords=["membership", "password", "login"],
        description="Member account management and password reset.",
    )


@pytest.fixture
def link_listings() -> CandidateLink:
    return CandidateLink(
        label="Listings Help",
        url="https://example.com/listings",
        keywords=["listing", "property", "create listing"],
        description="Help center for property listings.",
    )


@pytest.fixture
def link_billing() -> CandidateLink:
    return CandidateLink(
        label="Billing Help",
        url="https://example.com/billing",
        keywords=["billing", "invoice", "payment", "account"],
        description="View invoices, update payment methods and manage billing.",
    )


# ---------------------------------------------------------------------------
# Request builder helpers
# ---------------------------------------------------------------------------

def make_request(
    query: str = "How do I reset my password?",
    llm_answer: str = "Go to the member portal and click Forgot Password.",
    kb_grounded: bool = True,
    is_grounded: bool = True,
    is_chitchat: bool = False,
    candidate_links: list[CandidateLink] | None = None,
) -> GuardrailRequest:
    return GuardrailRequest(
        query=query,
        llm_answer=llm_answer,
        grounding=Grounding(is_grounded=is_grounded, kb_grounded=kb_grounded),
        is_chitchat=is_chitchat,
        candidate_links=candidate_links or [],
    )


# ---------------------------------------------------------------------------
# Embedding mock helpers
# ---------------------------------------------------------------------------

def unit_vec(dims: int = 384, seed: int = 0) -> list[float]:
    """Return a deterministic unit vector."""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def similar_vec(base: list[float], similarity: float, seed: int = 99) -> list[float]:
    """Return a unit vector with approximately `similarity` cosine to base."""
    noise = unit_vec(len(base), seed)
    blended = [base[i] * similarity + noise[i] * (1 - similarity) for i in range(len(base))]
    norm = math.sqrt(sum(x * x for x in blended))
    return [x / norm for x in blended]
