from __future__ import annotations
import re
from typing import Optional, Tuple
from app.models import CandidateLink


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())

    def normalize(tok: str) -> str:
        if len(tok) > 4 and tok.endswith("ies"):
            return tok[:-3] + "y"
        if len(tok) > 3 and tok.endswith("es"):
            return tok[:-2]
        if len(tok) > 3 and tok.endswith("s"):
            return tok[:-1]
        return tok

    return {normalize(t) for t in raw}


def keyword_match(
    query: str,
    candidates: list[CandidateLink],
    threshold: float = 0.0,
) -> Tuple[Optional[CandidateLink], float]:
    """
    Lexical overlap between query tokens and each candidate's keywords + description.
    Returns the best candidate and its normalised score, or (None, 0.0) if no match
    exceeds the threshold.
    """
    if not candidates:
        return None, 0.0

    query_tokens = _tokenize(query)
    best_link: Optional[CandidateLink] = None
    best_score: float = 0.0

    for candidate in candidates:
        keyword_tokens = _tokenize(" ".join(candidate.keywords))
        label_tokens = _tokenize(candidate.label)
        all_tokens = keyword_tokens | label_tokens

        intersection_keywords = query_tokens & keyword_tokens
        intersection_all = query_tokens & all_tokens

        score_keywords = (len(intersection_keywords) / len(keyword_tokens)) if keyword_tokens else 0.0
        score_all = (len(intersection_all) / len(all_tokens)) if all_tokens else 0.0
        score = max(score_keywords, score_all)

        if score >= best_score:
            best_score = score
            best_link = candidate

    if best_link is not None and best_score >= threshold:
        if best_score == 0.0 and threshold > 0.0:
            return None, best_score
        return best_link, best_score
    return None, best_score
