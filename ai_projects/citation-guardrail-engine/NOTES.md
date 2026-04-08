# NOTES.md

## Chosen strategy — Semantic (default) with keyword fallback

**Why semantic?**  
Keyword matching is brittle: a query like "I can't log in" shares no tokens with keywords like "membership, portal, password", yet is clearly about the member portal. Embeddings capture this intent.

`sentence-transformers/all-MiniLM-L6-v2` was chosen because:
- It is explicitly suggested in the exercise brief
- It runs free on the HuggingFace Inference API (no paid tier)
- It is fast (384-dim vectors, ~22M params) and good enough for short texts like queries and link descriptions

**Strategy flag**  
Set via `STRATEGY` env var: `semantic` (default) | `keyword` | `hybrid`.  
`hybrid` runs semantic first and promotes keyword winner when semantic yields no match.

---

## Trade-off accepted

**Threshold tuned empirically, not learned.**  
The similarity threshold (`SIMILARITY_THRESHOLD=0.45`) was set by running the seed cases manually. A production system would learn the threshold from labelled data using a held-out validation set. The current value may need adjustment for different domains or embedding models.

---

## One limitation

**The HuggingFace free Inference API has cold-start latency.**  
The first request after model inactivity can take 10–20 seconds while the model loads. Subsequent requests are fast. In production this would be addressed by using a dedicated endpoint or a self-hosted model. The fallback to keyword mode handles timeouts gracefully, so the endpoint never fails — but the first cold-start call may exceed `EMBEDDING_TIMEOUT_S` and silently degrade to keyword.

---

## Assumption on R3 (already_present)

The check is a simple `url in llm_answer` substring match. This covers both plain-text URLs (`https://example.com/members`) and markdown links (`[label](https://example.com/members)`). If a URL appears inside an HTML attribute or other encoded form, it would not be detected — accepted as an edge case outside scope.

---

## Edge case I added (edge_08_ambiguous_keyword_semantic_disambiguates)

Both candidates share the keyword "account". Keyword strategy would pick the one that appears first or has marginally higher overlap. Semantic strategy correctly picks "Billing Help" because the query talks about invoices. This validates the core reason for using embeddings over pure lexical matching.
