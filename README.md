# DB Intelligence – Deterministic Postgres SQL Agent


## Requirements
- Python 3.9+
- PostgreSQL with `pg_trgm` (and optionally `vector`)
- Ollama running locally with a pulled chat model (default `deepseek-r1`) and optional embed model

## Install
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Configure (.env)
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
DB_SCHEMAS=public
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_CHAT=deepseek-r1
OLLAMA_MODEL_EMBED=nomic-embed-text
ROW_LIMIT_DEFAULT=10
ANSWER_PREVIEW_ROWS=20
SINGLE_TABLE_CONFIDENCE=0.75
DEBUG_RETRIEVAL=false
```

## Migrations
Idempotent setup (trgm, optional vector, docs/columns/values tables, indexes):
- `migrations/001_retrieval_setup.sql` (earlier setup)
- `migrations/xxxx_single_table_first.sql` (adds optional `vector`, ivfflat index, value hints)

Run them with psql:
```bash
psql "$DATABASE_URL" -f migrations/001_retrieval_setup.sql
psql "$DATABASE_URL" -f migrations/xxxx_single_table_first.sql
```

Build docs/columns for retrieval (creates structural docs and columns registry):
```bash
python -m src.cli build-docs
```

## Run
- API: `uvicorn src.server:app --host 127.0.0.1 --port 8000`
  - Swagger: http://127.0.0.1:8000/docs
- CLI:
```bash
python -m src.cli ask "last 5 created alarm rules"
```
- Streamlit app:
```bash
streamlit run streamlit_app.py
```

## HTTP API
POST `/query`
```json
{
  "question": "last 5 created alarm rules",
  "schemas": ["public"]
}
```
Response (success):
```json
{
  "sql": "SELECT \"name\", \"created_at\" FROM \"public\".\"alarm_rule\"\nORDER BY \"created_at\" DESC\nLIMIT 5",
  "columns": ["name", "created_at"],
  "rows": [["Rule A","2025-08-01T10:00:00Z"], ...],
  "row_count": 5,
  "answer": "1. Rule A, 2. Rule B, ..."
}
```
Response (no signal or empty):
```json
{ "sql": "", "columns": [], "rows": [], "row_count": 0, "answer": "No answer" }
```

## Behavior and Guarantees
- Numerical intent
  - Parses N from: “last/top/first/latest/newest N”, digits or words up to 100
  - Enforces LIMIT = N (and trims in-memory if needed)
  - Enforces ORDER BY direction: last/latest/top/newest → DESC; first/earliest/oldest → ASC
- Date/time intent
  - Handles: yesterday; last N days; last week; last month; on YYYY-MM-DD; between A and B
  - Chooses best timestamp: created_at/on, event_time, occurred_at, else typed ts/date with sensible names
  - If time intent but no usable timestamp → `No answer (missing timestamp)`
- SQL safety
  - Exactly one statement; SELECT only; rejects DML/DDL and multi-statements
- No answer (exactly): retrieval empty/low confidence, missing timestamp, invalid SQL, empty results, or date mismatch

## Debugging (Trace)
Enable `DEBUG_RETRIEVAL=true` and call `/query` or `SqlAgent.answer_question(...)` to get a trace in the response:
```json
{
  "retrieval": [
    {"table":"public.alarm_rule","score":1.87,"nameSim":0.74,"colHits":1.2,"synHits":0.6,"vec":0.3,"penalties":0.0}
  ],
  "chosenTable":"public.alarm_rule",
  "plan": {"columns":["name","created_at","id"],"filters":{},"order":["created_at","desc"],"limit":5},
  "intent": {"count":5, "order":"desc", "date_range":"last 5 days", "createdPreferred":true},
  "sql": "...",
  "reason": null
}
```
