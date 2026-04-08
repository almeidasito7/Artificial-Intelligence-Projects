# Conversational BI Assistant (LLM)

Conversational assistant for a staffing/ops team that answers natural-language questions about:
- structured data (SQLite) via Text-to-SQL
- internal documents (Markdown) via RAG

This project implements four pillars:
- Text-to-SQL with SQL validation (SELECT-only) and read-only execution
- RAG with ChromaDB + sentence-transformers
- Row-Level Security (RLS) by region/division, applied before any database execution
- Semantic cache with permission-scope isolation and TTL

## Example Data

- Database: [data/staffing.db](data/staffing.db)
- Access profiles: [data/user_permissions.json](data/user_permissions.json)
- RAG documents: [data/documents/](data/documents/)

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## LLM Configuration

The code uses the `openai` SDK with an OpenAI-compatible endpoint. You can use OpenAI directly (recommended if your key starts with `sk-...`) or OpenRouter.

```bash
# OpenAI
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# OpenRouter
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-4o-mini"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Running

CLI (interactive):

```bash
python -m src.main --mode cli --user carol.chen
```

Web (FastAPI + Uvicorn):

```bash
python -m src.main --mode web --host 0.0.0.0 --port 8000
```

Useful endpoints:
- `GET /health`
- `GET /users`
- `POST /chat` with `{ "username": "...", "query": "..." }`

## Quality (lint / typecheck / tests)

```bash
python -m ruff check .
python -m mypy .
python -m pytest
```

## How It Works

- Flow: semantic cache → access check (regions/divisions) → query classification (SQL vs RAG) → agent execution → store response in cache.
- SQL path:
  - Generates a SELECT via the LLM using the live SQLite schema.
  - Validates the SQL (no DDL/DML or multi-statements) and injects Row‑Level Security filters by region/division before execution.
  - Executes read‑only and returns a concise, friendly answer.
- RAG path:
  - Ingests Markdown files from `data/documents/` into ChromaDB (idempotent).
  - Retrieves top‑K relevant chunks and generates an answer grounded on them.
  - Sources are returned separately; they are not embedded in the answer text.

## Row‑Level Security (RLS)

- Each user has a profile in `data/user_permissions.json` with allowed `regions` and `divisions`.
- RLS is enforced by injecting `WHERE` conditions into the generated SQL before any database access.
- The LLM never sees unauthorized rows; SQLite is opened with `?mode=ro` (read‑only).

## Semantic Cache

- User‑aware: the cache key is a hash of `{regions, divisions}` so users with different scopes never share cached results.
- Cosine similarity on sentence‑transformer embeddings; high default threshold to avoid false hits.
- TTL and simple LRU per user namespace.

## What You Can Ask

The assistant supports two main types of questions:

- Data questions (SQL): counts, averages, trends, “top N”, filters by region/division/status/date, etc.
- Policy/process questions (RAG): onboarding SOP, timesheets, contractor policies, data privacy, benefits FAQ, etc.

Examples (SQL):

```text
How many open jobs do we have?
What is the average bill rate for IT placements this quarter?
Show me the top 5 clients by number of placements in the last 30 days.
How many candidates were registered last month?
What is the average margin_pct by division?
```

Examples (RAG):

```text
What does the onboarding SOP say about background checks?
What is the policy for contractor time-off requests?
Summarize the data privacy policy in 3 bullets.
Do we offer benefits to contractors?
```

Access notes:

- If you ask for a region outside your profile scope, the request is denied before any LLM/DB call.
- Document access is not restricted (all users can query documents).

## CLI Usage Examples

```text
python -m src.main --mode cli --user admin
> How many open jobs do we have?
> What does our onboarding SOP say about background checks?
```

## API Usage Examples

curl (bash):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","query":"How many placements last quarter?"}'
```

PowerShell:

```powershell
Invoke-RestMethod -Method POST `
  -Uri http://localhost:8000/chat `
  -ContentType "application/json" `
  -Body '{"username":"admin","query":"How many placements last quarter?"}'
```

## Documents (RAG)

- Place Markdown files in `data/documents/` before first run.
- Ingestion is automatic the first time the RAG agent is initialized.
- To rebuild the vector store, stop the app and delete the `.chroma/` directory; it will be recreated on next start.

## Database

- A prebuilt SQLite file is included at `data/staffing.db`.
- To regenerate data:

```bash
python data/seed_database.py
```

## Troubleshooting

- Authentication 401:
  - If your key starts with `sk-`, prefer OpenAI with:
    - `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o-mini`, `OPENAI_BASE_URL=https://api.openai.com/v1`
  - For OpenRouter, use:
    - `OPENROUTER_API_KEY`, `OPENROUTER_MODEL=openai/gpt-4o-mini`, `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
  - Ensure the key is on the same line as the variable in `.env` (no wrapped lines).
- “Path not found”:
  - Run from the project root and verify `data/staffing.db` exists; regenerate with the seed script if needed.
- Port in use:
  - Change `--port` when starting the web server.

## Notes for Publishing

- Do not commit `.env` or any secrets. `.env` is ignored by default.
- Tests, lint and typecheck are configured and passing (see Quality section).
- Recommended initial push:
  - `git init && git add . && git commit -m "Initial public version"`
  - Create a GitHub repo and `git remote add origin <url>; git push -u origin main`
