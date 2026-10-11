# SQL Query AI Agent — FDSE Hiring Assignment

A task-focused LangGraph agent that translates natural language into SQL
against a provided database schema, with validation, optimization, debugging,
explanation, multi-turn context, and strict out-of-scope refusal.

This README is a **living record of finalized decisions**. Entries move here
only after they've been agreed on; nothing speculative lives in this file.

---

## 1. Problem statement (recap)

Build an AI agent that:

- translates natural language into read-only SQL on a provided schema
- validates every query (tables, columns, joins, syntax, read-only only)
- explains, optimizes, and debugs SQL
- carries conversation context so follow-ups refine the previous query
- refuses anything outside SQL / database scope
- never generates destructive SQL (`DELETE / UPDATE / INSERT / DROP / ALTER / TRUNCATE`)

**Timebox:** ~5 hours total — 2.5 h model, 1 h frontend, 0.5 h deploy on Render,
1 h docs + demo video.

---

## 2. Dataset — Chinook (SQLite)

- **File:** `data/Chinook_Sqlite.sqlite` (984 KB, SQLite 3)
- **Why:** real public dataset, SQLite-native (zero-setup, deploy-friendly),
  11 tables with real FK relationships, and it literally contains the columns
  the assignment examples name (`Employee.HireDate`, `Customer.State`).
- **Tables:** `Artist, Album, Genre, MediaType, Track, Employee, Customer,
  Invoice, InvoiceLine, Playlist, PlaylistTrack`
- **Relationships (11):**
  ```
  Album.ArtistId        -> Artist.ArtistId
  Track.AlbumId         -> Album.AlbumId
  Track.GenreId         -> Genre.GenreId
  Track.MediaTypeId     -> MediaType.MediaTypeId
  Employee.ReportsTo    -> Employee.EmployeeId   (self-join)
  Customer.SupportRepId -> Employee.EmployeeId
  Invoice.CustomerId    -> Customer.CustomerId
  InvoiceLine.InvoiceId -> Invoice.InvoiceId
  InvoiceLine.TrackId   -> Track.TrackId
  PlaylistTrack.PlaylistId -> Playlist.PlaylistId
  PlaylistTrack.TrackId    -> Track.TrackId
  ```
- **DDL + relationship map** live in [`app/schema.py`](app/schema.py) and are
  the single source of truth for the generator prompt and the validator.

### Demo caveat
Chinook's `Employee.HireDate` values are 2002–2004, so the brief's literal
example "hired after Jan 2024" is schema-correct but returns 0 rows. We may
bump 2–3 rows into 2024 via a seed script before the demo video.

---

## 3. Architecture — finalized decisions

### 3.1 Intent classifier (one LLM call, merged with scope detection)
Classifier produces strict JSON:
```json
{
  "intent": "generate_sql | optimize_sql | debug_sql | explain_sql | followup | out_of_scope",
  "scope_reason": "destructive | non_sql | ambiguous | null",
  "needs_clarification": true | false,
  "clarification_question": "string | null"
}
```
- "Intent Detection" + "Scope Validation" from the assignment diagram are
  **collapsed into this one node** — one LLM call, not two.
- `out_of_scope` short-circuits to a one-line refusal; no pipeline runs.
- If the model says `followup` but no `last_sql` exists, we rewrite it to
  `generate_sql` defensively.

### 3.2 Conversation context model
- **`last_sql`** — the most recent generated SQL. Follow-ups anchor on this.
- **`messages`** — trimmed rolling window (~4–6 turns). Gives the LLM
  linguistic context for refinements like "only those from California".
- No unbounded history. Follow-ups always target the newest query.

### 3.3 Injection guard (separate from scope)
Keeps the intent classifier simple. The guard is a cheap regex pre-filter +
short CHEAP-tier LLM call that only answers: *is this trying to jailbreak the
system prompt?* ("ignore your instructions", "reveal your prompt", etc.).

### 3.4 Schema delivery — full DDL, no RAG
Chinook is small enough (~900 tokens of DDL) to inject verbatim into every
generator / optimizer / debugger prompt. RAG would add latency, a vector
store, and failure modes for zero benefit at this size. For >100-table
schemas, this node would be swapped for retrieval; noted in future-work.

### 3.5 SQL validator (shared by every path)
Uses `sqlglot` to AST-parse the SQL and confirm:
- Statement is `SELECT` (or `WITH ... SELECT`) only — reject destructive verbs
- Every referenced table exists in `TABLES`
- Every referenced column exists on its table
- Every `JOIN ... ON` references a real FK pair in `RELATIONSHIPS`
- Parses cleanly in SQLite dialect

Reused by `generate_sql`, `optimize_sql`, `debug_sql`.

### 3.6 Optimizer — single pass + revalidate + fallback
- One LLM pass produces: optimized SQL, index suggestions, notes.
- Optimized SQL is re-validated once.
- **If re-validation fails, fall back to the original validated SQL** so a
  bad optimization never breaks the response. Not a loop — a safety gate.

### 3.7 Executor (optional, bonus)
- Runs the final SQL against the local SQLite DB in read-only mode.
- Uses `EXPLAIN QUERY PLAN` for a cheap cost estimate (bonus item).
- Results are capped (e.g. 100 rows) for the UI; CSV export covers full set.

### 3.8 Explainer
Reads the shared LangGraph state (generated SQL, validation result,
optimization notes, execution rows) and writes 1–3 lines of plain English.
No manual JSON plumbing — LangGraph's shared state handles the accumulation.

### 3.9 LangGraph routing
```
                       intent_classifier (CHEAP, merged scope)
                             |
          +------------------+------------------------+
          |                  |                        |
   out_of_scope        generate/followup       optimize/debug/explain
          |                  |                        |
      one-liner        injection_guard          injection_guard
                             |                        |
                       schema injected          schema injected
                             |                        |
                       sql_generator            (user-supplied SQL)
                             |                        |
                       sql_validator <-----+     validate user SQL
                             |             |           |
                       sql_optimizer       |      +----+----+
                             |             |      |         |
                       re-validate --------+   optimize   debug/explain
                       (fallback to orig if fail)
                             |                   |
                     (optional) executor <-------+
                             |
                         explainer
                             |
                      return response
```

### 3.10 LLM router — two tiers, three cohorts, key rotation, throttling

**Why:** demonstrate a reliability story (and keep running when a free tier
rate-limits). Scores on both the architecture and code-quality axes.

**Tiers**
- `STRONG` — SQL generation / optimize / debug / explain
- `CHEAP`  — intent classify / injection guard

**Cohort chain (per tier)**: Gemini → Groq → OpenRouter. On any provider
error, the next cohort is tried. Within Gemini, keys round-robin.

**Current slugs** (all free-tier at time of writing):

| Tier    | Gemini                     | Groq                      | OpenRouter                              |
|---------|----------------------------|---------------------------|------------------------------------------|
| STRONG  | `gemini-2.5-flash`         | `openai/gpt-oss-120b`     | `deepseek/deepseek-chat-v3-0324:free`    |
| CHEAP   | `gemini-2.5-flash-lite`    | `openai/gpt-oss-20b`      | `google/gemini-2.0-flash-exp:free`       |

**Throttling (per-cohort `min_interval` seconds between any two calls):**
- Gemini: **7 s** (free tier is 10 RPM per project; keys share quota)
- Groq: 0 s
- OpenRouter: 0 s

All configurable via env (`GEMINI_MIN_INTERVAL`, `*_STRONG_MODEL`,
`*_CHEAP_MODEL`, etc.). See [`.env.example`](.env.example).

Code: [`app/llm_router.py`](app/llm_router.py).

---

## 4. State shape (LangGraph)

```python
{
  "messages": [...],              # trimmed rolling window (~4-6)
  "user_input": str,
  "intent": str,                  # routing key
  "scope_reason": str | None,     # destructive / non_sql / ambiguous
  "needs_clarification": bool,
  "clarification_question": str | None,
  "last_sql": str | None,         # follow-up anchor
  "schema_ddl": str,              # full DDL
  "generated_sql": str | None,
  "validation": {"ok": bool, "errors": [...]},
  "optimization": {"sql": str, "indexes": [...], "notes": [...]} | None,
  "execution": {"rows": [...], "cost": ..., "error": ...} | None,
  "explanation": str | None,
  "final_response": str,
}
```

---

## 5. Project layout (current)

```
.
├── .env.example           committed template; copy to .env and fill in keys
├── .env                   gitignored, local only
├── requirements.txt
├── README.md              <-- this file
├── data/
│   └── Chinook_Sqlite.sqlite
├── app/
│   ├── __init__.py
│   ├── schema.py          CHINOOK_DDL + TABLES + RELATIONSHIPS
│   ├── llm_router.py      two-tier cohort fallback + Gemini key rotation + throttling
│   └── nodes.py           graph nodes (currently: classify_intent)
└── notebooks/
    └── 01_build_agent.ipynb   builds and tests each node, then wires LangGraph
```

---

## 6. Progress log

| Step | Status | Notes |
|------|--------|-------|
| Chinook DB downloaded + schema introspected | ✅ | 11 tables, 11 FKs |
| `app/schema.py` (DDL + relationship map) | ✅ | single source of truth |
| `app/llm_router.py` (2 tiers × 3 cohorts) | ✅ | throttling + key rotation |
| Node 1 — intent classifier (`classify_intent`) | ✅ | 10/10 on test matrix |
| Node 2 — injection guard (`scan_injection`) | ✅ | 14/14 on test matrix, regex-only |
| Node 3 — SQL generator (`generate_sql`) | ✅ | 6/6, handles follow-ups via `last_sql` |
| Node 4 — SQL validator (`validate_sql`) | ✅ | 13/13, sqlglot AST walk + CTE support |
| Node 5 — optimizer (`optimize_sql`) | ✅ | 4/4, single pass + revalidate + fallback |
| Node 6 — executor (`execute_sql`) | ✅ | read-only URI + `EXPLAIN QUERY PLAN` + row cap |
| Node 7 — explainer (`explain_sql`) | ✅ | 1-3 sentence summary |
| Node 8 — debugger (`debug_sql`) | ✅ | revalidate + safe fallback |
| LangGraph wiring (`app/graph.py`) | ✅ | 9/9 scenarios green, state typed |
| Router: 60s cohort cooldown on 429 | ✅ | kills log flood from exhausted quotas |
| Prompt: Chinook data conventions (state codes etc.) | ✅ | so "California" → `'CA'` |
| FastAPI backend | ⏭ next | |
| Frontend (chat + SQL panel + history + dark mode) | ⏭ | |
| Dockerfile + Render deploy | ⏭ | |
| Demo video (3–5 min) | ⏭ | |

---

## 7. Scope — what we're building

**Required (all in):**
NL→SQL · validation · explanation · optimization · debugging · multi-turn
context · out-of-scope refusal · guardrails · chat UI · SQL panel ·
explanation panel · history · loading indicator · error messages.

**Bonus (planned):**
- SQL execution against sample DB (SQLite)
- Follow-up refinement (already required)
- Prompt-injection protection
- SQL syntax highlighting (frontend)
- CSV export of query results
- Query cost estimation (`EXPLAIN QUERY PLAN`)
- Dockerized deployment
- Lightweight unit/integration tests

**Explicitly skipped:**
- Multi-dialect support (SQLite / PG / MySQL) — out of scope for the timebox.
- RAG schema retrieval — unneeded at Chinook's size.

---

## 8. Setup (local)

```bash
# 1. Clone + enter the folder
# 2. Copy env template and fill in keys
cp .env.example .env
# edit .env: at minimum, GEMINI_API_KEYS=key1,key2,...

# 3. Install deps
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/01_build_agent.ipynb
# Run cell 1 to verify the router sees your cohorts; cell 2 pings each tier.
```

Backend + frontend run instructions will be added once those pieces exist.
