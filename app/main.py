"""FastAPI backend for the SQL agent.

Endpoints:
    POST  /api/chat          -> run the agent on one turn
    GET   /api/schema        -> return the DDL + table/column map
    POST  /api/export-csv    -> execute a validated SELECT and return CSV
    GET   /                  -> serve the single-page frontend
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from dataclasses import is_dataclass, asdict
from typing import Any, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph import build_graph, run_agent
from app.nodes import execute_sql, validate_sql
from app.schema import CHINOOK_DDL, RELATIONSHIPS, TABLES


log = logging.getLogger("app.main")
logging.getLogger("app.llm_router").setLevel(logging.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = FastAPI(title="SQL Query AI Agent", version="0.1.0")

# CORS: during dev we run frontend + backend on the same origin, but a
# relaxed CORS lets you point other dev frontends at it if useful.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses + Pydantic things to JSON-safe forms."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Fallback: stringify anything else.
    return str(obj)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    user_input: str
    last_sql: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)


class ExportCsvRequest(BaseModel):
    sql: str
    row_cap: int = Field(default=1000, ge=1, le=10000)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/schema")
def get_schema() -> dict:
    return {
        "ddl": CHINOOK_DDL,
        "tables": {t: sorted(list(cols)) for t, cols in TABLES.items()},
        "relationships": [
            {"from_table": ft, "from_col": fc, "to_table": tt, "to_col": tc}
            for (ft, fc), (tt, tc) in RELATIONSHIPS.items()
        ],
    }


# ---------------------------------------------------------------------------
# Streaming chat (SSE)
# ---------------------------------------------------------------------------

# Human-friendly labels + icons for each graph node so the UI can show
# live progress without the frontend needing to know node internals.
_STEP_LABELS: dict[str, str] = {
    "classify":          "Understanding your request",
    "inject":            "Checking for injection",
    "refuse_oos":        "Refusing (out of scope)",
    "refuse_injection":  "Refusing (injection attempt)",
    "clarify":           "Need a bit more detail",
    "extract_opt":       "Extracting your SQL",
    "extract_explain":   "Extracting your SQL",
    "generate":          "Writing SQL",
    "validate":          "Validating SQL against schema",
    "optimize":          "Optimizing",
    "debug":             "Fixing your SQL",
    "execute":           "Running on the database",
    "explain":           "Writing explanation",
    "validation_failed": "Validation failed",
    "debug_fallback":    "Could not safely correct",
}

_compiled_graph = None

def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE endpoint. Emits:
      event: step   -> {node, label, update}         (one per node)
      event: text   -> {t}                           (word-by-word final response)
      event: done   -> {<full final state>}         (terminator)
      event: error  -> {message}
    """
    def gen() -> Iterator[str]:
        state = {
            "user_input": req.user_input,
            "last_sql": req.last_sql,
            "messages": [m.model_dump() for m in req.messages],
        }
        # Prime the UI immediately so the first word shows up without delay.
        yield _sse("open", {"ts": time.time()})

        final_state: dict = {}
        try:
            for chunk in _get_graph().stream(state):
                # LangGraph yields {node_name: partial_state_update}
                for node_name, update in chunk.items():
                    final_state.update(update or {})
                    yield _sse("step", {
                        "node": node_name,
                        "label": _STEP_LABELS.get(node_name, node_name),
                        "update": _to_jsonable(update or {}),
                    })
        except Exception as e:  # noqa: BLE001
            log.exception("stream run failed")
            yield _sse("error", {"message": f"agent error: {e}"})
            return

        # Convenience fields the frontend uses up-front
        final_payload = _to_jsonable(final_state)
        final_payload["sql"] = (
            final_state.get("final_sql")
            or final_state.get("generated_sql")
            or ""
        )
        final_payload["response"] = final_state.get("response", "")
        final_payload["status"] = final_state.get("status", "ok")

        # Stream the final response text word-by-word for a typing effect.
        resp = final_payload["response"] or ""
        if resp:
            words = resp.split(" ")
            for i, w in enumerate(words):
                piece = w + (" " if i < len(words) - 1 else "")
                yield _sse("text", {"t": piece})
                time.sleep(0.02)

        yield _sse("done", final_payload)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx, render)
        },
    )


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    try:
        state = run_agent(
            req.user_input,
            last_sql=req.last_sql,
            messages=[m.model_dump() for m in req.messages],
        )
    except Exception as e:  # noqa: BLE001
        log.exception("agent run failed")
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    # The state carries dataclasses (IntentResult, ValidationResult, ...);
    # make them JSON-safe before returning.
    payload = _to_jsonable(state)

    # Convenience fields for the frontend.
    payload["sql"] = state.get("final_sql") or state.get("generated_sql") or ""
    payload["response"] = state.get("response", "")
    payload["status"] = state.get("status", "ok")
    return payload


@app.post("/api/export-csv")
def export_csv(req: ExportCsvRequest) -> PlainTextResponse:
    # Hard guardrail: never execute anything the validator rejects.
    v = validate_sql(req.sql)
    if not v.ok:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "SQL failed validation; refusing to execute.",
                "errors": [e.to_dict() for e in v.errors],
            },
        )
    e = execute_sql(req.sql, row_cap=req.row_cap)
    if not e.ok:
        raise HTTPException(status_code=400, detail=f"execution error: {e.error}")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(e.columns)
    for row in e.rows:
        w.writerow(row)
    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="results.csv"'},
    )


# ---------------------------------------------------------------------------
# Frontend (served as static)
# ---------------------------------------------------------------------------

if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Any other static assets (css/js) live next to index.html.
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
