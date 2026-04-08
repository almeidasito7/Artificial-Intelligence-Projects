# Technical Test — Senior AI Engineer

## Citation Guardrail Engine (RAG Post-Processing)

**Company:** VoiceFlip Technologies
**Role:** Senior AI Engineer
**Suggested time:** ~1h planning + ~2h implementation. Video and final commits can be pushed later the same day.
**Stack:** Free choice (Python, TypeScript, etc.) with a fixed JSON contract.
**LLM:** HuggingFace (free Inference models are fine) as default + a flag to switch to OpenAI (`--llm hf|openai`).

---

## Welcome

Hi, and thanks for taking the time to do this exercise.

This is a self-contained technical test. You do **not** need access to any private repository, everything you need is in this document. Build the solution from scratch, on your own machine.

A few ground rules we want to be explicit about:

- **AI assistants are expected and encouraged.** We are hiring a Senior AI Engineer; we evaluate your technical judgment, decisions and ability to reason about trade-offs — not your typing speed. If you use Claude / ChatGPT / Cursor / Copilot, that is perfectly fine. Just make sure you understand and can defend every line you ship.
- **Small and solid beats large and messy.** If you have to cut scope, cut features, not rigor.
- **We read the commit history.** A single monolithic "final" commit is a negative signal. Incremental commits that tell the story of how you approached the problem are a strong positive signal.
- **The video is where you earn the hardest points.** See the "Video" section below. Record it in English.

---

## Context

A RAG assistant answers user questions using a Knowledge Base. For certain topics, the organization has a **canonical URL** (citation) that should appear in the final response when the answer is grounded in the KB — but must never appear on chitchat or when the answer was not grounded.

We want to see how you design a small **Citation Guardrail Engine** that receives the raw LLM answer and decides what to do with the citation: inject it, leave it, skip it. Correctly, observably, and measurable.

---

## Goal

Build a small HTTP service with **one main endpoint** that takes an LLM answer + grounding context + a list of candidate citations, and returns the final answer along with the citation decision explained.

---

## Input/Output Contract (fixed JSON)

### `POST /guardrail`

**Request:**

```json
{
  "query": "How do I reset my membership password?",
  "llm_answer": "Go to the member portal and click 'Forgot Password'. You will receive an email.",
  "grounding": {
    "is_grounded": true,
    "kb_grounded": true
  },
  "is_chitchat": false,
  "candidate_links": [
    {
      "label": "Member Portal",
      "url": "https://example.com/members",
      "keywords": ["membership", "member portal", "password", "login"],
      "description": "Official member portal for account management and password reset."
    },
    {
      "label": "Listings Help",
      "url": "https://example.com/listings",
      "keywords": ["listing", "create listing", "edit listing"],
      "description": "Help center for property listings."
    }
  ]
}
```

**Response:**

```json
{
  "final_answer": "Go to the member portal and click 'Forgot Password'. You will receive an email.\n\nFor more information, see [Member Portal](https://example.com/members).",
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

---

## Guardrail Rules (must-enforce)

| # | Condition | Expected action | Status |
|---|-----------|-----------------|--------|
| R1 | `is_chitchat=true` | Never inject citation | `skipped_chitchat` |
| R2 | `kb_grounded=false` | Never inject citation | `skipped_ungrounded` |
| R3 | `kb_grounded=true` + the winning citation URL is already present in `llm_answer` | Do not duplicate | `already_present` |
| R4 | `kb_grounded=true` + no citation in answer + valid match | Inject citation at the end | `injected` |
| R5 | `kb_grounded=true` + no candidate link matches above threshold | Do not invent a citation | `skipped_no_match` |

Additional rules:

- The citation **must never** appear more than once in `final_answer`.
- The engine **must never modify the semantics** of `llm_answer`. It may only append the citation.

---

## Technical Requirements

### 1. Matching Strategy (Core)

Implement at least one strategy to pick the winning candidate link. You choose:

- **Keyword** — lexical overlap between query and link keywords
- **Semantic** — embeddings with cosine similarity (recommended)
- **Hybrid** — combination (bonus)

The strategy must be swappable: the engine should take a `strategy` parameter (CLI flag or env var).

### 2. LLM Integration

- **Default**: HuggingFace (use a free Inference model like `sentence-transformers/all-MiniLM-L6-v2` — no paid tier required).
- **Flag**: `--llm openai` (or equivalent env var) switches to OpenAI embeddings.
- The service must run successfully without any OpenAI key.
- Keys via environment variables only, never hardcoded.

### 3. Minimum API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/guardrail` | Run the guardrail over an LLM response |
| `GET` | `/health` | Basic counters per `citation_decision.status` |

### 4. Failure Handling

- **Embedding API timeout or error**: must fall back to keyword mode (or equivalent deterministic path) and record the reason in `citation_decision.reason`. Must not fail the endpoint.
- **Empty candidate_links**: do not call the LLM, respond with `skipped_no_match` (or R1/R2 if applicable).

---

## Golden Set — 5 Seed Cases (extend to at least 10)

You **must extend this set to at least 10 cases** by adding edge cases you consider important. We want to see which edge cases you come up with — it is part of the evaluation.

```json
[
  {
    "id": "seed_01_grounded_inject",
    "input": {
      "query": "How do I reset my membership password?",
      "llm_answer": "Go to the member portal and click Forgot Password. You will receive an email.",
      "grounding": {"is_grounded": true, "kb_grounded": true},
      "is_chitchat": false,
      "candidate_links": [
        {"label": "Member Portal", "url": "https://example.com/members", "keywords": ["membership", "password", "login"], "description": "Member account management"},
        {"label": "Listings Help", "url": "https://example.com/listings", "keywords": ["listing", "property"], "description": "Listings help"}
      ]
    },
    "expected": {"status": "injected", "matched_label": "Member Portal"}
  },
  {
    "id": "seed_02_grounded_already_present",
    "input": {
      "query": "Where can I find the member portal?",
      "llm_answer": "You can find it at [Member Portal](https://example.com/members).",
      "grounding": {"is_grounded": true, "kb_grounded": true},
      "is_chitchat": false,
      "candidate_links": [
        {"label": "Member Portal", "url": "https://example.com/members", "keywords": ["membership", "portal"], "description": "Member account management"}
      ]
    },
    "expected": {"status": "already_present", "matched_label": "Member Portal"}
  },
  {
    "id": "seed_03_chitchat_skip",
    "input": {
      "query": "hello, how are you?",
      "llm_answer": "Hi! I'm doing well, thanks for asking. How can I help you today?",
      "grounding": {"is_grounded": true, "kb_grounded": false},
      "is_chitchat": true,
      "candidate_links": [
        {"label": "Member Portal", "url": "https://example.com/members", "keywords": ["membership", "portal"], "description": "Member account management"}
      ]
    },
    "expected": {"status": "skipped_chitchat", "matched_label": null}
  },
  {
    "id": "seed_04_ungrounded_skip",
    "input": {
      "query": "What's the weather in Tokyo?",
      "llm_answer": "I haven't been trained on that one yet.",
      "grounding": {"is_grounded": false, "kb_grounded": false},
      "is_chitchat": false,
      "candidate_links": [
        {"label": "Member Portal", "url": "https://example.com/members", "keywords": ["membership"], "description": "Member portal"}
      ]
    },
    "expected": {"status": "skipped_ungrounded", "matched_label": null}
  },
  {
    "id": "seed_05_no_match",
    "input": {
      "query": "What time does the pool open?",
      "llm_answer": "The pool opens at 7am every day.",
      "grounding": {"is_grounded": true, "kb_grounded": true},
      "is_chitchat": false,
      "candidate_links": [
        {"label": "Member Portal", "url": "https://example.com/members", "keywords": ["membership", "portal"], "description": "Member account management"}
      ]
    },
    "expected": {"status": "skipped_no_match", "matched_label": null}
  }
]
```

### Guidance for extending to 10+ cases

Add at least one case for each of these categories:

- **Ambiguous keyword**: a common word matches two groups and semantics must disambiguate
- **Empty candidate list**: `candidate_links=[]` must not break the service
- **Match just below threshold**: the best candidate exists but is too weak → `skipped_no_match`
- **Two candidates with similar scores**: your margin logic decides
- **(Your choice)**: one edge case that YOU think matters — explain why in the golden set file

---

## Eval Script

A simple script that loads the golden set and prints:

- Per-case: `id`, `expected_status`, `actual_status`, `pass/fail`
- Aggregated: `correct_decision_rate` (accuracy over the whole set)

No web UI required. A CLI `python eval.py` (or equivalent) is enough.

---

## Deliverables

1. **Public Git repository** with incremental commits that reflect your plan. **Minimum 3 commits** showing progress (e.g., contract + skeleton → strategy → eval + polish). A single final commit is a negative signal.
2. **Functional code** that runs locally.
3. **`README.md`** with: how to run, how to switch HF/OpenAI, which HF model you picked, how to run the eval.
4. **`NOTES.md`** with: chosen strategy + why, one trade-off you accepted, one limitation.
5. **Extended golden set** as a JSON file (10+ cases).
6. **Eval output** (a text file or pasted into the README with the final numbers).
7. **Video walkthrough (in ENGLISH)** linked from the `README.md`. Loom, YouTube unlisted, Google Drive, or equivalent. Keep it as long as you need to clearly walk through your decisions and a live run of the eval — there is no upper limit, and there is no prize for being brief. That said, most strong submissions land somewhere between 5 and 15 minutes.

If time is tight, prioritize: functional code + passing the 5 seed cases + video.

## What we do NOT evaluate

- Frontend
- Database
- Authentication
- Deployment
- Model training / fine-tuning
- Performance benchmarks

---

## Final Note on Scope

We prefer a small but solid solution, with explained judgment, over a large one with no metrics. If you cannot finish everything, prioritize:

1. Rules R1–R5 correct
2. The 5 seed golden cases passing at 100%
3. The video explaining your main decisions

---

## Video — what we want to hear

Record the video in English. Screen-share your repo, run the eval live on camera, and then answer these three questions with conviction:

1. **"Why didn't you just ask the synthesis LLM to include the citation directly in its response?"** — There is a good answer rooted in reproducibility, observability and cost. If your answer boils down to *"because the exercise asks for it"*, that is a negative signal.
2. **"What is the most dangerous failure mode of your system and how would you detect it in production?"**
3. **"If you had one more week, what would you add first and why?"**

You can take as long as you need. We care about clarity of reasoning, not about hitting a time budget.

---

## How to Submit

When you are done, send us:

1. The **link to your public Git repository** (GitHub, GitLab, Bitbucket — any of them is fine).
2. The **video link** (make sure it is accessible without requiring us to request access).
3. A one-paragraph note with anything you want us to keep in mind when reviewing (optional, but appreciated).

Send everything to the same contact who shared this document with you.

---

## Questions?

If something in this document is genuinely ambiguous, make a reasonable assumption, write it down in `NOTES.md`, and move on. We would rather see how you resolve ambiguity than answer a long list of clarification requests.

Good luck!

---

**VoiceFlip Technologies — 2026**
