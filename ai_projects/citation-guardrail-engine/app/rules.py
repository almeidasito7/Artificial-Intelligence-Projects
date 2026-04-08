from __future__ import annotations
from app.models import GuardrailRequest, CitationDecision, CandidateLink


def url_already_present(url: str, text: str) -> bool:
    return url in text


def build_decision(
    status: str,
    reason: str,
    matched_link: CandidateLink | None = None,
    strategy_used: str | None = None,
    similarity_score: float | None = None,
) -> CitationDecision:
    return CitationDecision(
        status=status,
        matched_label=matched_link.label if matched_link else None,
        strategy_used=strategy_used,
        similarity_score=round(similarity_score, 4) if similarity_score is not None else None,
        reason=reason,
    )


def check_early_rules(req: GuardrailRequest) -> CitationDecision | None:
    """
    Evaluate R1 and R2 synchronously before any async work.
    Returns a CitationDecision if a rule fires, else None.
    """
    # R1 — chitchat
    if req.is_chitchat:
        return build_decision("skipped_chitchat", "is_chitchat=true, citation must never appear")

    # R2 — not grounded
    if not req.grounding.kb_grounded:
        return build_decision("skipped_ungrounded", "kb_grounded=false, answer not grounded in KB")

    # Empty candidate list — no LLM call needed
    if not req.candidate_links:
        return build_decision("skipped_no_match", "candidate_links is empty")

    return None


def apply_post_match_rules(
    req: GuardrailRequest,
    matched_link: CandidateLink | None,
    score: float,
    strategy_used: str,
    fallback_reason: str,
) -> tuple[CitationDecision, str]:
    """
    Evaluate R3, R4, R5 after the matching step.
    Returns (CitationDecision, final_answer).
    """
    base_reason_suffix = f"; {fallback_reason}" if fallback_reason else ""

    # R5 — no candidate matched above threshold
    if matched_link is None:
        return (
            build_decision(
                "skipped_no_match",
                f"kb_grounded=true, no candidate above threshold (best score={round(score, 4)}){base_reason_suffix}",
                strategy_used=strategy_used,
                similarity_score=score,
            ),
            req.llm_answer,
        )

    # R3 — URL already in the answer
    if url_already_present(matched_link.url, req.llm_answer):
        return (
            build_decision(
                "already_present",
                f"kb_grounded=true, citation URL already present in answer{base_reason_suffix}",
                matched_link=matched_link,
                strategy_used=strategy_used,
                similarity_score=score,
            ),
            req.llm_answer,
        )

    # R4 — inject citation
    final_answer = (
        f"{req.llm_answer}\n\n"
        f"For more information, see [{matched_link.label}]({matched_link.url})."
    )
    return (
        build_decision(
            "injected",
            f"kb_grounded=true, no citation present, {strategy_used} match above threshold{base_reason_suffix}",
            matched_link=matched_link,
            strategy_used=strategy_used,
            similarity_score=score,
        ),
        final_answer,
    )
