# Citation Guardrail Engine

A post-processing HTTP service that decides whether to inject, skip, or leave a citation in an LLM-generated answer, based on grounding context and semantic matching.

Built for the VoiceFlip Technologies Senior AI Engineer technical test.

---

## Stack

- **Python 3.11+**
- **FastAPI** — HTTP service
- **HuggingFace Inference API** — embeddings (default, free tier)
- **OpenAI** — embeddings (optional, via flag)
- **httpx** — async HTTP client

---

## How to run

### 1. Clone and install

```bash
git clone <repo-url>
cd citation-guardrail
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file (optional — all vars have defaults):

```env
# LLM provider: "hf" (default) or "openai"
LLM_PROVIDER=hf

# HuggingFace token (optional — increases rate limits)
HF_API_TOKEN=hf_...

# Required only if LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Matching strategy: "semantic" (default) | "keyword" | "hybrid"
STRATEGY=semantic

# Cosine similarity threshold (0.0–1.0)
SIMILARITY_THRESHOLD=0.45

# Embedding API timeout in seconds
EMBEDDING_TIMEOUT_S=5.0
```

### 3. Start the service

```bash
uvicorn app.main:app --reload
```

Service runs at `http://localhost:8000`.

---

## Switching between HuggingFace and OpenAI

```bash
# Use HuggingFace (default — no API key required)
LLM_PROVIDER=hf uvicorn app.main:app

# Use OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... uvicorn app.main:app
```

---

## HuggingFace model

`sentence-transformers/all-MiniLM-L6-v2`

- 384-dimensional embeddings
- Free via HuggingFace Inference API
- No account required for basic usage (token optional, increases rate limits)

**Note on cold starts:** The first request may take 10–20s while HuggingFace loads the model. Subsequent requests are fast. On timeout, the engine automatically falls back to keyword matching and records the reason in `citation_decision.reason`.

---

## API

### `POST /guardrail`

Run the guardrail over an LLM response.

**Request:**
```json
{
  "query": "How do I reset my membership password?",
  "llm_answer": "Go to the member portal and click 'Forgot Password'.",
  "grounding": {
    "is_grounded": true,
    "kb_grounded": true
  },
  "is_chitchat": false,
  "candidate_links": [
    {
      "label": "Member Portal",
      "url": "https://example.com/members",
      "keywords": ["membership", "password", "login"],
      "description": "Official member portal for account management."
    }
  ]
}
```

**Response:**
```json
{
  "final_answer": "Go to the member portal and click 'Forgot Password'.\n\nFor more information, see [Member Portal](https://example.com/members).",
  "citation_decision": {
    "status": "injected",
    "matched_label": "Member Portal",
    "strategy_used": "semantic",
    "similarity_score": 0.78,
    "reason": "kb_grounded=true, no citation present, semantic match above threshold"
  },
  "metrics": {
    "latency_ms": 234,
    "llm_calls": 1
  }
}
```

### `GET /health`

Returns service status and per-status counters.

```json
{
  "status": "ok",
  "counters": {
    "injected": 12,
    "skipped_chitchat": 3,
    "skipped_ungrounded": 1,
    "already_present": 2,
    "skipped_no_match": 4
  }
}
```

---

## Citation decision statuses

| Status | Condition |
|---|---|
| `injected` | kb_grounded=true, match above threshold, URL not already present |
| `already_present` | kb_grounded=true, match found, URL already in the answer |
| `skipped_chitchat` | is_chitchat=true (R1 — highest priority) |
| `skipped_ungrounded` | kb_grounded=false (R2) |
| `skipped_no_match` | No candidate above threshold, or empty candidate list |

---

## Running the eval

```bash
# Start the service first
uvicorn app.main:app &

# Run eval against all 12 golden cases
python eval.py

# Override strategy
python eval.py --strategy keyword

# Custom host
python eval.py --base-url http://localhost:8000
```

### Eval output

```
Citation Guardrail Engine — Eval
Endpoint : http://localhost:8000/guardrail
Strategy : semantic
Cases    : 12

ID                                            EXPECTED               ACTUAL                 RESULT
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
seed_01_grounded_inject                       injected / Member Port injected / Member Port PASS
seed_02_grounded_already_present              already_present / Memb already_present / Memb PASS
...

Results: 12/12 passed — accuracy: 100.0%
```

---

## Video walkthrough

[Link to video — to be added after recording]

---

## See also

- [NOTES.md](./NOTES.md) — strategy decisions, trade-offs, limitations
