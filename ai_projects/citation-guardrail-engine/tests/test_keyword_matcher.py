"""Unit tests for the keyword (lexical) matcher."""
from __future__ import annotations
import pytest
from app.matchers.keyword import keyword_match, _tokenize
from app.models import CandidateLink
from tests.conftest import make_request


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert "Password" not in _tokenize("Reset Password")
        assert "password" in _tokenize("Reset Password")

    def test_splits_on_space(self):
        tokens = _tokenize("member portal login")
        assert tokens == {"member", "portal", "login"}

    def test_empty_string(self):
        assert _tokenize("") == set()


# ---------------------------------------------------------------------------
# keyword_match — basic behaviour
# ---------------------------------------------------------------------------

class TestKeywordMatch:
    def test_returns_best_match(self, link_member_portal, link_listings):
        matched, score = keyword_match(
            "How do I reset my membership password?",
            [link_member_portal, link_listings],
            threshold=0.0,
        )
        assert matched is not None
        assert matched.label == "Member Portal"
        assert score > 0.0

    def test_no_overlap_returns_none_when_threshold_positive(self, link_member_portal):
        matched, score = keyword_match(
            "What time does the pool open?",
            [link_member_portal],
            threshold=0.10,
        )
        assert matched is None

    def test_zero_threshold_always_returns_something(self, link_member_portal):
        """threshold=0.0 must return the best candidate even with zero overlap."""
        matched, score = keyword_match(
            "completely unrelated query xyz",
            [link_member_portal],
            threshold=0.0,
        )
        assert matched is not None

    def test_empty_candidates_returns_none(self):
        matched, score = keyword_match("reset password", [], threshold=0.0)
        assert matched is None
        assert score == 0.0

    def test_score_is_normalised(self, link_member_portal):
        _, score = keyword_match("reset password", [link_member_portal], threshold=0.0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# keyword_match — disambiguation
# ---------------------------------------------------------------------------

class TestKeywordDisambiguation:
    def test_picks_higher_overlap_candidate(self, link_member_portal, link_listings):
        """Query about listings should pick Listings Help, not Member Portal."""
        matched, score = keyword_match(
            "How do I create a new property listing?",
            [link_member_portal, link_listings],
            threshold=0.0,
        )
        assert matched is not None
        assert matched.label == "Listings Help"

    def test_threshold_filters_weak_match(self, link_member_portal):
        """A match below threshold must return None even if it's the best."""
        matched_low, score_low = keyword_match(
            "pool hours",
            [link_member_portal],
            threshold=0.0,
        )
        matched_high, _ = keyword_match(
            "pool hours",
            [link_member_portal],
            threshold=score_low + 0.01,  # just above the actual score
        )
        assert matched_high is None

    def test_description_words_count(self):
        """Words in the description field should contribute to matching.
        Note: the matcher uses exact token overlap (no stemming), so the query
        must share exact tokens with the description or keywords to score > 0.
        """
        link = CandidateLink(
            label="Billing Help",
            url="https://example.com/billing",
            keywords=["payment"],
            description="View billing history and manage payments",
        )
        matched, score = keyword_match(
            "I need to check my billing",  # 'billing' appears in description
            [link],
            threshold=0.0,
        )
        assert matched is not None
        assert score > 0.0
