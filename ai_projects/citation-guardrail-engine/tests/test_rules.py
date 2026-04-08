"""Unit tests for guardrail rules R1–R5 (pure logic, no I/O)."""
from __future__ import annotations
import pytest
from app.rules import check_early_rules, apply_post_match_rules, url_already_present
from app.models import CandidateLink
from tests.conftest import make_request


# ---------------------------------------------------------------------------
# url_already_present
# ---------------------------------------------------------------------------

class TestUrlAlreadyPresent:
    def test_markdown_link(self):
        assert url_already_present(
            "https://example.com/members",
            "See [Member Portal](https://example.com/members) for details.",
        )

    def test_plain_url(self):
        assert url_already_present(
            "https://example.com/members",
            "Visit https://example.com/members directly.",
        )

    def test_not_present(self):
        assert not url_already_present(
            "https://example.com/members",
            "Go to the portal and click Forgot Password.",
        )

    def test_partial_url_does_not_match(self):
        assert not url_already_present(
            "https://example.com/members",
            "Visit https://example.com",
        )


# ---------------------------------------------------------------------------
# R1 — chitchat
# ---------------------------------------------------------------------------

class TestR1Chitchat:
    def test_chitchat_skipped(self, link_member_portal):
        req = make_request(
            query="hello, how are you?",
            llm_answer="Hi! How can I help?",
            is_chitchat=True,
            candidate_links=[link_member_portal],
        )
        decision = check_early_rules(req)
        assert decision is not None
        assert decision.status == "skipped_chitchat"
        assert decision.matched_label is None

    def test_chitchat_overrides_kb_grounded(self, link_member_portal):
        """R1 must fire even when kb_grounded=True."""
        req = make_request(
            is_chitchat=True,
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        decision = check_early_rules(req)
        assert decision.status == "skipped_chitchat"


# ---------------------------------------------------------------------------
# R2 — not grounded
# ---------------------------------------------------------------------------

class TestR2Ungrounded:
    def test_ungrounded_skipped(self, link_member_portal):
        req = make_request(
            query="What's the weather in Tokyo?",
            kb_grounded=False,
            is_grounded=False,
            candidate_links=[link_member_portal],
        )
        decision = check_early_rules(req)
        assert decision is not None
        assert decision.status == "skipped_ungrounded"
        assert decision.matched_label is None

    def test_grounded_passes_through(self, link_member_portal):
        """kb_grounded=True must return None so matching continues."""
        req = make_request(
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        assert check_early_rules(req) is None


# ---------------------------------------------------------------------------
# Empty candidate list
# ---------------------------------------------------------------------------

class TestEmptyCandidates:
    def test_empty_list_returns_no_match(self):
        req = make_request(kb_grounded=True, candidate_links=[])
        decision = check_early_rules(req)
        assert decision is not None
        assert decision.status == "skipped_no_match"

    def test_empty_list_does_not_call_matcher(self):
        """check_early_rules must short-circuit before any I/O."""
        req = make_request(kb_grounded=True, candidate_links=[])
        decision = check_early_rules(req)
        assert decision.matched_label is None


# ---------------------------------------------------------------------------
# R3 — already present
# ---------------------------------------------------------------------------

class TestR3AlreadyPresent:
    def test_markdown_link_already_present(self, link_member_portal):
        req = make_request(
            llm_answer="See [Member Portal](https://example.com/members) for help.",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        decision, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.80, "semantic", ""
        )
        assert decision.status == "already_present"
        assert decision.matched_label == "Member Portal"
        assert final_answer == req.llm_answer  # answer must not be modified

    def test_plain_url_already_present(self, link_member_portal):
        req = make_request(
            llm_answer="Visit https://example.com/members directly.",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        decision, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.75, "semantic", ""
        )
        assert decision.status == "already_present"
        assert final_answer == req.llm_answer


# ---------------------------------------------------------------------------
# R4 — inject
# ---------------------------------------------------------------------------

class TestR4Inject:
    def test_citation_injected(self, link_member_portal):
        req = make_request(
            llm_answer="Go to the portal and click Forgot Password.",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        decision, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.78, "semantic", ""
        )
        assert decision.status == "injected"
        assert decision.matched_label == "Member Portal"
        assert "[Member Portal](https://example.com/members)" in final_answer

    def test_citation_appended_at_end(self, link_member_portal):
        req = make_request(
            llm_answer="Reset your password via the portal.",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        _, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.78, "semantic", ""
        )
        assert final_answer.startswith(req.llm_answer)

    def test_citation_appears_exactly_once(self, link_member_portal):
        req = make_request(
            llm_answer="Reset your password via the portal.",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        _, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.78, "semantic", ""
        )
        assert final_answer.count("https://example.com/members") == 1

    def test_llm_answer_semantics_unchanged(self, link_member_portal):
        original = "Go to the portal and click Forgot Password."
        req = make_request(llm_answer=original, kb_grounded=True, candidate_links=[link_member_portal])
        _, final_answer = apply_post_match_rules(
            req, link_member_portal, 0.78, "semantic", ""
        )
        assert final_answer.startswith(original)


# ---------------------------------------------------------------------------
# R5 — no match
# ---------------------------------------------------------------------------

class TestR5NoMatch:
    def test_no_match_skipped(self, link_member_portal):
        req = make_request(
            query="What time does the pool open?",
            kb_grounded=True,
            candidate_links=[link_member_portal],
        )
        decision, final_answer = apply_post_match_rules(
            req, matched_link=None, score=0.12, strategy_used="semantic", fallback_reason=""
        )
        assert decision.status == "skipped_no_match"
        assert decision.matched_label is None
        assert final_answer == req.llm_answer  # answer must not be modified

    def test_no_match_records_best_score(self, link_member_portal):
        req = make_request(kb_grounded=True, candidate_links=[link_member_portal])
        decision, _ = apply_post_match_rules(
            req, matched_link=None, score=0.22, strategy_used="semantic", fallback_reason=""
        )
        assert decision.similarity_score == pytest.approx(0.22, abs=1e-4)


# ---------------------------------------------------------------------------
# Fallback reason propagation
# ---------------------------------------------------------------------------

class TestFallbackReason:
    def test_fallback_reason_in_injected(self, link_member_portal):
        req = make_request(kb_grounded=True, candidate_links=[link_member_portal])
        reason = "embedding_api_error (Timeout), fell back to keyword"
        decision, _ = apply_post_match_rules(
            req, link_member_portal, 0.50, "keyword_fallback", reason
        )
        assert reason in decision.reason

    def test_no_fallback_reason_clean(self, link_member_portal):
        req = make_request(kb_grounded=True, candidate_links=[link_member_portal])
        decision, _ = apply_post_match_rules(
            req, link_member_portal, 0.78, "semantic", ""
        )
        assert "fell back" not in decision.reason
