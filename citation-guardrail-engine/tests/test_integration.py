"""Integration tests — full request/response cycle via FastAPI TestClient."""
from __future__ import annotations
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app, _counters
from tests.conftest import unit_vec, similar_vec

DIMS = 384
client = TestClient(app)


def high_sim_mock(*labels_and_sims):
    """
    Build a mock embedding response where:
      - vectors[0] is the query
      - vectors[1..n] correspond to candidates in order, each with given similarity
    """
    q = unit_vec(DIMS, seed=99)
    vecs = [q] + [similar_vec(q, sim, seed=i + 10) for i, (_, sim) in enumerate(labels_and_sims)]
    return AsyncMock(return_value=vecs)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "counters" in data

    def test_counters_are_dict(self):
        resp = client.get("/health")
        assert isinstance(resp.json()["counters"], dict)


# ---------------------------------------------------------------------------
# POST /guardrail — R1 chitchat
# ---------------------------------------------------------------------------

class TestGuardrailChitchat:
    def test_chitchat_returns_skipped(self):
        payload = {
            "query": "hello how are you",
            "llm_answer": "Hi! How can I help?",
            "grounding": {"is_grounded": True, "kb_grounded": False},
            "is_chitchat": True,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["membership"], "description": "Member portal"}
            ],
        }
        resp = client.post("/guardrail", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["citation_decision"]["status"] == "skipped_chitchat"
        assert data["citation_decision"]["matched_label"] is None
        assert data["final_answer"] == payload["llm_answer"]

    def test_chitchat_metrics_zero_llm_calls(self):
        payload = {
            "query": "hey",
            "llm_answer": "Hey!",
            "grounding": {"is_grounded": False, "kb_grounded": False},
            "is_chitchat": True,
            "candidate_links": [],
        }
        resp = client.post("/guardrail", json=payload)
        assert resp.json()["metrics"]["llm_calls"] == 0


# ---------------------------------------------------------------------------
# POST /guardrail — R2 ungrounded
# ---------------------------------------------------------------------------

class TestGuardrailUngrounded:
    def test_ungrounded_returns_skipped(self):
        payload = {
            "query": "What is the capital of France?",
            "llm_answer": "Paris.",
            "grounding": {"is_grounded": False, "kb_grounded": False},
            "is_chitchat": False,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["membership"], "description": "Member portal"}
            ],
        }
        resp = client.post("/guardrail", json=payload)
        assert resp.status_code == 200
        assert resp.json()["citation_decision"]["status"] == "skipped_ungrounded"


# ---------------------------------------------------------------------------
# POST /guardrail — R3 already present
# ---------------------------------------------------------------------------

class TestGuardrailAlreadyPresent:
    def test_markdown_url_already_present(self):
        payload = {
            "query": "Where is the member portal?",
            "llm_answer": "See [Member Portal](https://example.com/members) for help.",
            "grounding": {"is_grounded": True, "kb_grounded": True},
            "is_chitchat": False,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["membership", "portal"], "description": "Member account management"}
            ],
        }
        mock = high_sim_mock(("Member Portal", 0.85))
        with patch("app.matchers.semantic._embed_hf", mock):
            resp = client.post("/guardrail", json=payload)

        data = resp.json()
        assert data["citation_decision"]["status"] == "already_present"
        assert data["final_answer"] == payload["llm_answer"]
        assert data["final_answer"].count("https://example.com/members") == 1


# ---------------------------------------------------------------------------
# POST /guardrail — R4 inject
# ---------------------------------------------------------------------------

class TestGuardrailInject:
    def test_citation_injected_in_answer(self):
        payload = {
            "query": "How do I reset my membership password?",
            "llm_answer": "Go to the member portal and click Forgot Password.",
            "grounding": {"is_grounded": True, "kb_grounded": True},
            "is_chitchat": False,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["membership", "password", "login"], "description": "Member account management"},
                {"label": "Listings Help", "url": "https://example.com/listings",
                 "keywords": ["listing", "property"], "description": "Listings help"},
            ],
        }
        mock = high_sim_mock(("Member Portal", 0.85), ("Listings Help", 0.20))
        with patch("app.matchers.semantic._embed_hf", mock):
            resp = client.post("/guardrail", json=payload)

        data = resp.json()
        assert data["citation_decision"]["status"] == "injected"
        assert data["citation_decision"]["matched_label"] == "Member Portal"
        assert "[Member Portal](https://example.com/members)" in data["final_answer"]
        assert data["final_answer"].startswith(payload["llm_answer"])

    def test_inject_records_latency(self):
        payload = {
            "query": "reset password",
            "llm_answer": "Click Forgot Password.",
            "grounding": {"is_grounded": True, "kb_grounded": True},
            "is_chitchat": False,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["password"], "description": "Member portal"}
            ],
        }
        mock = high_sim_mock(("Member Portal", 0.80))
        with patch("app.matchers.semantic._embed_hf", mock):
            resp = client.post("/guardrail", json=payload)

        assert resp.json()["metrics"]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# POST /guardrail — R5 no match
# ---------------------------------------------------------------------------

class TestGuardrailNoMatch:
    def test_below_threshold_skipped(self):
        payload = {
            "query": "What time does the pool open?",
            "llm_answer": "The pool opens at 7am.",
            "grounding": {"is_grounded": True, "kb_grounded": True},
            "is_chitchat": False,
            "candidate_links": [
                {"label": "Member Portal", "url": "https://example.com/members",
                 "keywords": ["membership", "portal"], "description": "Member account management"}
            ],
        }
        q = unit_vec(DIMS, seed=50)
        mp_vec = similar_vec(q, 0.20, seed=51)  # well below threshold
        mock = AsyncMock(return_value=[q, mp_vec])
        with patch("app.matchers.semantic._embed_hf", mock):
            resp = client.post("/guardrail", json=payload)

        data = resp.json()
        assert data["citation_decision"]["status"] == "skipped_no_match"
        assert data["final_answer"] == payload["llm_answer"]

    def test_empty_candidates_skipped_no_match(self):
        payload = {
            "query": "reset password",
            "llm_answer": "Click Forgot Password.",
            "grounding": {"is_grounded": True, "kb_grounded": True},
            "is_chitchat": False,
            "candidate_links": [],
        }
        resp = client.post("/guardrail", json=payload)
        assert resp.json()["citation_decision"]["status"] == "skipped_no_match"


# ---------------------------------------------------------------------------
# Health counters increment
# ---------------------------------------------------------------------------

class TestHealthCounters:
    def test_counters_increment_on_request(self):
        _counters.clear()
        payload = {
            "query": "hi",
            "llm_answer": "Hello!",
            "grounding": {"is_grounded": False, "kb_grounded": False},
            "is_chitchat": True,
            "candidate_links": [],
        }
        client.post("/guardrail", json=payload)
        health = client.get("/health").json()
        assert health["counters"].get("skipped_chitchat", 0) >= 1


# ---------------------------------------------------------------------------
# Response contract shape
# ---------------------------------------------------------------------------

class TestResponseContract:
    def test_response_has_required_fields(self):
        payload = {
            "query": "hi",
            "llm_answer": "Hello!",
            "grounding": {"is_grounded": False, "kb_grounded": False},
            "is_chitchat": True,
            "candidate_links": [],
        }
        resp = client.post("/guardrail", json=payload)
        data = resp.json()
        assert "final_answer" in data
        assert "citation_decision" in data
        assert "metrics" in data
        cd = data["citation_decision"]
        assert "status" in cd
        assert "matched_label" in cd
        assert "reason" in cd
        m = data["metrics"]
        assert "latency_ms" in m
        assert "llm_calls" in m
