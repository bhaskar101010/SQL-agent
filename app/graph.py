"""LangGraph wiring for the SQL agent.

Shared state flows through conditional edges keyed off `intent`. Each branch
reuses the pure functions from `app.nodes`.
"""
from __future__ import annotations

import re
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.llm_router import LLMRouter
from app.nodes import (
    DebugResult,
    ExecutionResult,
    GenerationResult,
    IntentResult,
    OptimizationResult,
    ValidationResult,
    classify_intent,
    debug_sql,
    execute_sql,
    explain_sql,
    generate_sql,
    optimize_sql,
    scan_injection,
    validate_sql,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Inputs
    user_input: str
    messages: list[dict]          # trimmed rolling window
    last_sql: Optional[str]

    # Classification + guard
    intent: IntentResult
    injection_safe: bool
    injection_reason: Optional[str]

    # User-supplied SQL (for optimize/debug/explain intents)
    user_sql: Optional[str]

    # Pipeline results
    generated_sql: Optional[str]
    validation: Optional[ValidationResult]
    optimization: Optional[OptimizationResult]
    debug: Optional[DebugResult]
    execution: Optional[ExecutionResult]
    explanation: Optional[str]

    # Final
    final_sql: Optional[str]        # the SQL we chose to run / return
    response: str                   # user-facing message
    status: str                     # ok | refused | clarify | error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQL_BLOCK_RE = re.compile(
    r"((?:WITH\s|SELECT\s|SELECT\n|SELCT\s|SELEC\s)"       # tolerate a typo
    r".*?;)",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_SQL_RE = re.compile(
    r"```(?:sql|sqlite)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_sql(text: str) -> str:
    """Pull an SQL block out of a free-form user message."""
    if not text:
        return ""
    m = _FENCE_SQL_RE.search(text)
    if m:
        inner = m.group(1).strip()
        return inner if inner.endswith(";") else inner + ";"
    m = _SQL_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: hand the whole message to downstream — the LLM can still work.
    return text.strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_router_singleton: Optional[LLMRouter] = None

def _get_router() -> LLMRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = LLMRouter()
    return _router_singleton


def n_classify(state: AgentState) -> dict:
    r = classify_intent(
        state["user_input"],
        last_sql=state.get("last_sql"),
        messages=state.get("messages"),
        router=_get_router(),
    )
    return {"intent": r}


def n_inject(state: AgentState) -> dict:
    r = scan_injection(state["user_input"])
    return {"injection_safe": r.safe, "injection_reason": r.reason}


def n_refuse_oos(state: AgentState) -> dict:
    intent = state.get("intent")
    reason = intent.scope_reason if intent else "non_sql"
    msg = {
        "destructive": "I can only generate read-only SQL. I won't run DELETE, UPDATE, INSERT, DROP, ALTER, or TRUNCATE statements.",
        "non_sql":     "I'm designed to assist only with SQL and database-related tasks. Please ask a question related to the provided schema.",
        "ambiguous":   intent.clarification_question if intent and intent.clarification_question
                        else "I'm not sure what you're asking. Could you rephrase as a database question?",
    }.get(reason or "non_sql", "I can only help with SQL and database questions.")
    return {"status": "refused", "response": msg}


def n_refuse_injection(state: AgentState) -> dict:
    reason = state.get("injection_reason") or "possible prompt injection"
    return {
        "status": "refused",
        "response": f"That request looks like a prompt-injection attempt ({reason}). I only answer SQL and database questions grounded in the provided schema.",
    }


_SCHEMA_LEAK_RE = re.compile(
    r"\b(schema|tables?|columns?|which (?:data|db|database)|"
    r"what (?:tables?|columns?|data)|provide .* (?:schema|tables?|columns?))\b",
    re.IGNORECASE,
)


def n_clarify(state: AgentState) -> dict:
    intent = state.get("intent")
    q = (intent.clarification_question if intent else None) or ""
    # Safety net: never let the agent ask the user for schema details -- it has them.
    if not q.strip() or _SCHEMA_LEAK_RE.search(q):
        q = "Could you be a bit more specific about what you'd like to see?"
    return {"status": "clarify", "response": q}


def n_extract_user_sql(state: AgentState) -> dict:
    return {"user_sql": _extract_sql(state["user_input"])}


def n_generate(state: AgentState) -> dict:
    intent = state["intent"].intent
    r: GenerationResult = generate_sql(
        state["user_input"],
        intent=intent,
        last_sql=state.get("last_sql"),
        messages=state.get("messages"),
        router=_get_router(),
    )
    return {"generated_sql": r.sql}


def n_validate(state: AgentState) -> dict:
    sql = state.get("generated_sql") or state.get("user_sql") or ""
    return {"validation": validate_sql(sql)}


def n_optimize(state: AgentState) -> dict:
    sql = state.get("generated_sql") or state.get("user_sql") or ""
    r = optimize_sql(sql, router=_get_router())
    return {"optimization": r, "final_sql": r.sql}


def n_debug(state: AgentState) -> dict:
    r = debug_sql(
        state.get("user_sql") or state["user_input"],
        router=_get_router(),
    )
    return {"debug": r, "final_sql": r.corrected_sql if not r.used_fallback else ""}


def n_execute(state: AgentState) -> dict:
    sql = state.get("final_sql") or state.get("generated_sql") or state.get("user_sql") or ""
    if not sql:
        return {"execution": None}
    return {"execution": execute_sql(sql)}


def n_explain(state: AgentState) -> dict:
    sql = state.get("final_sql") or state.get("generated_sql") or state.get("user_sql") or ""
    opt = state.get("optimization")
    exe = state.get("execution")
    notes = opt.notes if opt and not opt.used_fallback else None
    row_count = exe.row_count if exe and exe.ok else None
    executed = bool(exe and exe.ok)
    r = explain_sql(
        sql,
        optimization_notes=notes,
        executed=executed,
        row_count=row_count,
        router=_get_router(),
    )
    return {"explanation": r.text, "status": "ok", "response": r.text}


def n_handle_validation_failure(state: AgentState) -> dict:
    v = state.get("validation")
    errs = ", ".join(e.message for e in v.errors) if v else "unknown error"
    sql = state.get("generated_sql") or state.get("user_sql") or "(none)"
    return {
        "status": "error",
        "response": f"The SQL failed validation: {errs}",
        "final_sql": sql,
    }


def n_handle_debug_fallback(state: AgentState) -> dict:
    d = state.get("debug")
    reason = d.fallback_reason if d else "unknown"
    issue = d.issue if d else ""
    return {
        "status": "error",
        "response": f"I identified the issue ({issue}) but could not produce a safe corrected query ({reason}).",
    }


# ---------------------------------------------------------------------------
# Routers (conditional edges)
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> str:
    intent = state["intent"]
    if intent.intent == "out_of_scope":
        return "refuse_oos"
    if intent.needs_clarification:
        return "clarify"
    return "inject"


def route_after_inject(state: AgentState) -> str:
    if not state.get("injection_safe", True):
        return "refuse_injection"
    i = state["intent"].intent
    if i in ("generate_sql", "followup"):
        return "generate"
    if i == "optimize_sql":
        return "extract_opt"
    if i == "debug_sql":
        return "debug"
    if i == "explain_sql":
        return "extract_explain"
    return "refuse_oos"


def route_after_validate(state: AgentState) -> str:
    v = state.get("validation")
    if v and v.ok:
        # If this came from the user, skip the optimizer (user wanted it as-is);
        # if generated, run the optimizer.
        if state.get("user_sql") and not state.get("generated_sql"):
            # explain path: execute then explain, no optimize
            return "execute"
        return "optimize"
    return "validation_failed"


def route_after_debug(state: AgentState) -> str:
    d = state.get("debug")
    if not d or d.used_fallback or not d.corrected_sql:
        return "debug_fallback"
    return "execute"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("classify", n_classify)
    g.add_node("inject", n_inject)
    g.add_node("refuse_oos", n_refuse_oos)
    g.add_node("refuse_injection", n_refuse_injection)
    g.add_node("clarify", n_clarify)

    g.add_node("extract_opt", n_extract_user_sql)
    g.add_node("extract_explain", n_extract_user_sql)

    g.add_node("generate", n_generate)
    g.add_node("validate", n_validate)
    g.add_node("optimize", n_optimize)
    g.add_node("debug", n_debug)
    g.add_node("execute", n_execute)
    g.add_node("explain", n_explain)

    g.add_node("validation_failed", n_handle_validation_failure)
    g.add_node("debug_fallback", n_handle_debug_fallback)

    # Entry
    g.set_entry_point("classify")

    # After classify
    g.add_conditional_edges("classify", route_after_classify, {
        "refuse_oos": "refuse_oos",
        "clarify": "clarify",
        "inject": "inject",
    })

    # After inject
    g.add_conditional_edges("inject", route_after_inject, {
        "refuse_injection": "refuse_injection",
        "generate": "generate",
        "extract_opt": "extract_opt",
        "debug": "debug",
        "extract_explain": "extract_explain",
        "refuse_oos": "refuse_oos",
    })

    # Generate -> validate -> optimize -> execute -> explain
    g.add_edge("generate", "validate")
    g.add_edge("extract_opt", "validate")
    g.add_edge("extract_explain", "validate")

    g.add_conditional_edges("validate", route_after_validate, {
        "optimize": "optimize",
        "execute": "execute",
        "validation_failed": "validation_failed",
    })

    g.add_edge("optimize", "execute")

    # Debug path
    g.add_conditional_edges("debug", route_after_debug, {
        "execute": "execute",
        "debug_fallback": "debug_fallback",
    })

    g.add_edge("execute", "explain")

    # Terminal
    for n in ("explain", "refuse_oos", "refuse_injection", "clarify",
              "validation_failed", "debug_fallback"):
        g.add_edge(n, END)

    return g.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

_compiled = None

def run_agent(user_input: str, *, last_sql: Optional[str] = None,
              messages: Optional[list[dict]] = None) -> AgentState:
    """One-shot invocation. Returns the full final state."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    state: AgentState = {
        "user_input": user_input,
        "last_sql": last_sql,
        "messages": messages or [],
    }
    return _compiled.invoke(state)
