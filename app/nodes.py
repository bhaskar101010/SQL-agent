"""Graph nodes for the SQL agent. Each node is a plain function so it can be
unit-tested in a notebook before wiring into LangGraph.

Current nodes:
    - classify_intent(user_input, last_sql, messages) -> IntentResult
    - scan_injection(user_input) -> InjectionResult
    - generate_sql(user_input, intent, last_sql, messages) -> GenerationResult
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Literal, Optional

from app.llm_router import LLMRouter, Tier
from app.schema import (
    CHINOOK_DDL,
    COLUMNS_LC,
    RELATIONSHIPS,
    TABLE_NAMES_LC,
    TABLES,
)

Intent = Literal[
    "generate_sql", "optimize_sql", "debug_sql",
    "explain_sql", "followup", "out_of_scope",
]
ScopeReason = Literal["destructive", "non_sql", "ambiguous"]


@dataclass
class IntentResult:
    intent: Intent
    scope_reason: Optional[ScopeReason] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    raw: Optional[str] = None  # debug: the model's raw text

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "raw"}


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """\
You classify user requests sent to an SQL assistant. Return STRICT JSON only -
no prose, no markdown fences.

Schema:
{
  "intent": "generate_sql" | "optimize_sql" | "debug_sql" | "explain_sql" | "followup" | "out_of_scope",
  "scope_reason": "destructive" | "non_sql" | "ambiguous" | null,
  "needs_clarification": true | false,
  "clarification_question": string | null
}

Intent rules:
- generate_sql: user asks a question in natural language that should be answered with a NEW SQL query.
- optimize_sql: user supplies SQL and asks to improve, speed up, add indexes, clean up, or rewrite it.
- debug_sql: user supplies SQL they say is broken, failing, or returning wrong results, and wants it fixed.
- explain_sql: user supplies SQL and wants it explained in plain English.
- followup: user's message refines the previous query (e.g. "only those from California", "sort by date", "add a count"). Only use this when `last_sql` is non-empty AND the message clearly refines it instead of starting a new topic.
- out_of_scope: anything not SQL / database related, OR requests for destructive statements (DELETE, UPDATE, INSERT, DROP, ALTER, TRUNCATE), OR requests so vague they cannot be classified.

scope_reason (only when intent == out_of_scope, else null):
- destructive: the user asked for a write/delete/schema-change statement.
- non_sql: the question is unrelated to SQL or the database (sports, politics, general knowledge, unrelated code, creative writing).
- ambiguous: the message is in-scope but too vague to act on.

needs_clarification: use VERY SPARINGLY. Set true ONLY when the user's message is so vague that NO reasonable SQL could be written for it (e.g. "give me data", "something about customers", "idk"). Do NOT set true just because:
- you don't know the schema (you don't need to — a downstream node has it),
- the question mentions tables/columns you aren't sure exist (assume they do; a downstream validator will catch real mismatches),
- the question is complex or involves multiple joins (that's fine, let the generator handle it),
- the user didn't specify every filter detail (sensible defaults are fine).
NEVER ask the user "what tables / columns are available". NEVER ask them to pick a schema. The downstream SQL generator has the full schema and will handle naming.
When you do set it true, write a short follow-up question (one sentence).

Return JSON only. No code fences. No commentary.
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_json(text: str) -> str:
    """Pull JSON out of a response that may be fenced or have prose around it."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Grab the first balanced-looking {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _format_history(messages: list[dict] | None, max_turns: int = 4) -> str:
    if not messages:
        return "(no prior turns)"
    tail = messages[-max_turns:]
    lines = []
    for m in tail:
        role = m.get("role", "?")
        content = (m.get("content", "") or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classify_intent(
    user_input: str,
    *,
    last_sql: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    router: Optional[LLMRouter] = None,
) -> IntentResult:
    """Classify user intent. Uses the CHEAP tier."""
    router = router or LLMRouter()
    user_block = (
        f"last_sql:\n{last_sql or '(none)'}\n\n"
        f"recent_turns:\n{_format_history(messages)}\n\n"
        f"user_message:\n{user_input}"
    )
    raw = router.chat(
        Tier.CHEAP,
        system=_INTENT_SYSTEM,
        user=user_block,
        temperature=0.0,
    )
    try:
        data = json.loads(_strip_json(raw))
    except (json.JSONDecodeError, ValueError):
        # Fail safe: if we can't parse, refuse the request as ambiguous.
        return IntentResult(
            intent="out_of_scope",
            scope_reason="ambiguous",
            needs_clarification=True,
            clarification_question="Sorry, I couldn't parse that. Can you rephrase?",
            raw=raw,
        )

    intent = data.get("intent", "out_of_scope")
    if intent not in {
        "generate_sql", "optimize_sql", "debug_sql",
        "explain_sql", "followup", "out_of_scope",
    }:
        intent = "out_of_scope"

    scope_reason = data.get("scope_reason")
    if intent != "out_of_scope":
        scope_reason = None
    elif scope_reason not in {"destructive", "non_sql", "ambiguous", None}:
        scope_reason = "non_sql"

    # Guard: if the model said `followup` but there's no prior SQL, treat as generate.
    if intent == "followup" and not last_sql:
        intent = "generate_sql"

    return IntentResult(
        intent=intent,  # type: ignore[arg-type]
        scope_reason=scope_reason,  # type: ignore[arg-type]
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=data.get("clarification_question") or None,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Injection guard (regex-only, no LLM call)
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    safe: bool
    reason: Optional[str] = None    # human-readable category
    matched: Optional[str] = None   # the pattern name that triggered

    def to_dict(self) -> dict:
        return asdict(self)


# (pattern_name, compiled regex, human reason)
# Case-insensitive. Patterns target the common prompt-injection attack surface:
#   - instruction override ("ignore previous instructions")
#   - role / persona takeover ("you are now DAN", "act as")
#   - system-prompt exfiltration ("reveal your prompt")
#   - rule-set replacement ("new rules:", "from now on")
#   - encoded payloads (base64 blobs, obvious payload markers)
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("instruction_override",
     re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
                r"(previous|prior|above|earlier|all|your|the)\s+"
                r"(instruction|instructions|rules|prompt|system|context|message)",
                re.IGNORECASE),
     "attempts to override prior instructions"),

    ("role_takeover",
     re.compile(r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as|"
                r"from\s+now\s+on\s+you\s+are)\b",
                re.IGNORECASE),
     "attempts to change the assistant's role"),

    ("jailbreak_persona",
     re.compile(r"\b(DAN|do\s+anything\s+now|developer\s+mode|jailbreak|"
                r"unfiltered\s+mode|without\s+restrictions|no\s+restrictions|"
                r"no\s+filter|evil\s+mode)\b",
                re.IGNORECASE),
     "known jailbreak persona or mode"),

    ("prompt_exfiltration",
     re.compile(r"\b(reveal|show|print|display|output|repeat|leak|expose)\b"
                r"[^.\n]{0,30}\b(system\s+prompt|your\s+prompt|initial\s+prompt|"
                r"instructions|system\s+message|hidden\s+instructions|rules)",
                re.IGNORECASE),
     "attempts to extract the system prompt"),

    ("rule_replacement",
     re.compile(r"\b(new\s+(rules|instructions|system|prompt)|"
                r"here\s+are\s+your\s+new|from\s+now\s+on,?\s+you)\b",
                re.IGNORECASE),
     "attempts to install new rules"),

    ("delimiter_injection",
     re.compile(r"(</?\s*(system|instructions?|prompt|assistant)\s*>|"
                r"\[/?\s*(system|instructions?|inst)\s*\]|"
                r"###\s*(system|instructions?|new\s+rules))",
                re.IGNORECASE),
     "fake system/instruction delimiter"),

    ("base64_payload",
     # Long contiguous base64 is almost never a real SQL question.
     re.compile(r"(?:[A-Za-z0-9+/]{80,}={0,2})"),
     "suspiciously long base64-like payload"),
]


def scan_injection(user_input: str) -> InjectionResult:
    """Fast regex scan for prompt-injection attempts. No LLM call.

    Returns InjectionResult(safe=False, ...) if a pattern matches.
    """
    text = user_input or ""
    for name, rx, reason in _INJECTION_PATTERNS:
        if rx.search(text):
            return InjectionResult(safe=False, reason=reason, matched=name)
    return InjectionResult(safe=True)


# ---------------------------------------------------------------------------
# SQL generator (STRONG tier). Handles `generate_sql` AND `followup`.
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    sql: str
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        return {"sql": self.sql}


_GENERATOR_SYSTEM = """\
You are an expert SQL assistant for the Chinook SQLite database.

ABSOLUTE RULES:
1. Output ONLY a single SQL statement. No prose, no commentary, no markdown code fences, no leading/trailing text.
2. Produce ONLY read-only SQL: a SELECT (optionally with a leading WITH clause). NEVER generate DELETE, UPDATE, INSERT, DROP, ALTER, TRUNCATE, CREATE, REPLACE, MERGE, GRANT, REVOKE, or any PRAGMA / ATTACH statement.
3. Use the SQLite dialect.
4. Use ONLY the tables and columns defined in the schema below. Do not invent tables or columns. If something the user asks for does not exist in the schema, pick the closest real columns and return a valid query - never fabricate identifiers.
5. Quote identifiers that collide with SQL keywords; otherwise use their exact spelling and casing from the schema.
6. Terminate the statement with a single semicolon.
7. For FOLLOW-UP requests, start from the previous SQL and modify it (add WHERE clauses, change columns, add JOINs, etc.) rather than writing a new query from scratch.

DATA CONVENTIONS (important when writing WHERE clauses):
- Customer.State, Employee.State, Invoice.BillingState are stored as 2-letter codes for US states (e.g. 'CA' for California, 'NY' for New York, 'TX' for Texas). When the user names a US state in full, translate it to its 2-letter code.
- Country columns are stored as full names (e.g. 'USA', 'Canada', 'Germany', 'Brazil').
- Dates are ISO-ish strings ('YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD').
- Compare strings with exact matches (case-sensitive). Use LIKE with % wildcards for partial matches.

SCHEMA:
```
{schema}
```
"""


def _format_history_short(messages: list[dict] | None, max_turns: int = 4) -> str:
    if not messages:
        return "(none)"
    tail = messages[-max_turns:]
    out = []
    for m in tail:
        role = m.get("role", "?")
        c = (m.get("content", "") or "").strip().replace("\n", " ")
        if len(c) > 240:
            c = c[:240] + "..."
        out.append(f"{role}: {c}")
    return "\n".join(out)


_SQL_FENCE_RE = re.compile(
    r"```(?:sql|sqlite)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _clean_sql(text: str) -> str:
    """Strip fences, leading labels, and trailing prose. Return a single statement."""
    s = text.strip()
    m = _SQL_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    # Remove a leading label like "SQL:" or "Query:" on the first line.
    s = re.sub(r"^(sql|query|answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    # Keep only up through the first terminating semicolon on its own.
    # (Multi-statement responses are collapsed to the first.)
    first_semi = s.find(";")
    if first_semi != -1:
        s = s[: first_semi + 1]

    # Ensure trailing semicolon.
    if not s.rstrip().endswith(";"):
        s = s.rstrip() + ";"
    return s.strip()


def generate_sql(
    user_input: str,
    *,
    intent: str = "generate_sql",
    last_sql: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    schema_ddl: str = CHINOOK_DDL,
    router: Optional[LLMRouter] = None,
) -> GenerationResult:
    """Produce a SELECT statement answering the user's request.

    `intent` must be 'generate_sql' or 'followup'. For 'followup', `last_sql`
    is injected as the baseline the model should modify.
    """
    router = router or LLMRouter()
    system = _GENERATOR_SYSTEM.format(schema=schema_ddl.strip())

    if intent == "followup" and last_sql:
        task_block = (
            "This is a FOLLOW-UP. Modify the previous SQL to satisfy the new request.\n\n"
            f"PREVIOUS SQL:\n{last_sql.strip()}\n\n"
            f"NEW REQUEST:\n{user_input.strip()}\n\n"
            f"RECENT CONVERSATION:\n{_format_history_short(messages)}\n\n"
            "Return ONLY the modified SQL statement."
        )
    else:
        task_block = (
            f"USER REQUEST:\n{user_input.strip()}\n\n"
            f"RECENT CONVERSATION:\n{_format_history_short(messages)}\n\n"
            "Return ONLY the SQL statement."
        )

    raw = router.chat(Tier.STRONG, system=system, user=task_block, temperature=0.0)
    sql = _clean_sql(raw)
    return GenerationResult(sql=sql, raw=raw)


# ---------------------------------------------------------------------------
# SQL validator (sqlglot AST walk; no LLM call)
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    code: str                       # e.g. "destructive", "unknown_table"
    message: str                    # human-readable
    detail: Optional[str] = None    # offending identifier, if any

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[ValidationError]
    # Normalized metadata the downstream nodes may want.
    tables_used: list[str]          # canonical-case table names
    is_select: bool

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [e.to_dict() for e in self.errors],
            "tables_used": self.tables_used,
            "is_select": self.is_select,
        }


_DESTRUCTIVE_KEYWORDS = {
    "DELETE", "UPDATE", "INSERT", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "MERGE", "GRANT",
    "REVOKE", "PRAGMA", "ATTACH", "DETACH", "VACUUM", "REINDEX",
}


def _destructive_prescan(sql: str) -> Optional[ValidationError]:
    """Catch destructive verbs even if sqlglot fails to parse. Cheap first line."""
    # Strip comments and string literals so keywords inside them don't trip us.
    cleaned = re.sub(r"--[^\n]*", " ", sql)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"'[^']*'", " ", cleaned)
    cleaned = re.sub(r'"[^"]*"', " ", cleaned)
    upper = cleaned.upper()
    for kw in _DESTRUCTIVE_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return ValidationError(
                code="destructive",
                message=f"destructive keyword `{kw}` is not allowed",
                detail=kw,
            )
    return None


def _collect_fk_pair_set() -> set[tuple[str, str, str, str]]:
    """Return bidirectional FK pairs as (t1, c1, t2, c2) with t1 < t2 (lexical).

    The validator accepts a join equality in either direction.
    """
    out: set[tuple[str, str, str, str]] = set()
    for (ft, fc), (tt, tc) in RELATIONSHIPS.items():
        pair = sorted([(ft, fc), (tt, tc)])
        out.add((pair[0][0], pair[0][1], pair[1][0], pair[1][1]))
    # Self-join on Employee (ReportsTo -> EmployeeId) is already in the map;
    # the sorted key handles the equal-table case fine.
    return out


_FK_PAIRS = _collect_fk_pair_set()


def _canon_table(name: str) -> Optional[str]:
    return TABLE_NAMES_LC.get((name or "").lower())


def _canon_column(table: str, column: str) -> Optional[str]:
    cols = COLUMNS_LC.get((table or "").lower())
    if not cols:
        return None
    return cols.get((column or "").lower())


def validate_sql(sql: str) -> ValidationResult:
    """Validate a generated SQL statement against the Chinook schema.

    Checks (in order): destructive-keyword prescan, parseability, SELECT-only,
    tables exist, columns exist (qualified or inferable), joins use real FKs.
    """
    import sqlglot
    from sqlglot import exp

    errors: list[ValidationError] = []
    tables_used: list[str] = []
    is_select = False

    # 1. Destructive prescan (works even on syntactically broken SQL).
    pre = _destructive_prescan(sql)
    if pre is not None:
        return ValidationResult(ok=False, errors=[pre],
                                tables_used=[], is_select=False)

    # 2. Parse.
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            ok=False,
            errors=[ValidationError("parse_error", f"could not parse SQL: {e}")],
            tables_used=[], is_select=False,
        )
    if tree is None:
        return ValidationResult(
            ok=False,
            errors=[ValidationError("parse_error", "empty parse result")],
            tables_used=[], is_select=False,
        )

    # 3. SELECT-only (allow top-level WITH ... SELECT).
    top = tree
    if isinstance(top, exp.With):
        top = top.this
    is_select = isinstance(top, (exp.Select, exp.Union, exp.Intersect, exp.Except))
    if not is_select:
        errors.append(ValidationError(
            "not_select",
            "only SELECT (optionally with a leading WITH) is permitted",
            detail=type(top).__name__,
        ))

    # 4a. Collect CTE names so they are treated as valid table references.
    cte_names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            cte_names.add(alias.lower())

    # 4b. Tables exist. Build alias -> canonical-table map while we scan.
    alias_to_table: dict[str, str] = {}   # lowercased alias/name -> canonical
    seen_tables: list[str] = []

    for t in tree.find_all(exp.Table):
        raw_name = t.name
        if raw_name and raw_name.lower() in cte_names:
            # CTE reference, not a base table.
            alias = (t.alias or raw_name).lower()
            # Mark the alias so column resolution won't trip. We don't know the
            # CTE's columns at validation time, so unqualified columns later
            # fall into a permissive branch (handled in 5).
            alias_to_table[alias] = f"__cte__:{raw_name.lower()}"
            continue
        canon = _canon_table(raw_name)
        if canon is None:
            errors.append(ValidationError(
                "unknown_table",
                f"unknown table: {raw_name}",
                detail=raw_name,
            ))
            continue
        if canon not in seen_tables:
            seen_tables.append(canon)
        alias = t.alias or raw_name
        alias_to_table[alias.lower()] = canon

    tables_used = seen_tables

    # 5. Columns exist. Qualified columns must be valid on their table.
    #    Unqualified columns must exist on at least one in-scope base table
    #    OR match a SELECT-list alias (legal in ORDER BY / GROUP BY / HAVING).
    #    CTE columns are not statically known -> skip resolution for them.
    base_in_scope = [t for t in alias_to_table.values() if not t.startswith("__cte__:")]
    base_in_scope = list(dict.fromkeys(base_in_scope))
    has_cte_in_scope = any(t.startswith("__cte__:") for t in alias_to_table.values())

    # Collect SELECT-list column aliases (and any aliased expressions).
    select_aliases: set[str] = set()
    for a in tree.find_all(exp.Alias):
        name = a.alias
        if name:
            select_aliases.add(name.lower())

    for c in tree.find_all(exp.Column):
        col_name = c.name
        if not col_name or col_name == "*":
            continue
        tbl_ref = c.table  # alias or table name, or ""
        if tbl_ref:
            canon_tbl = alias_to_table.get(tbl_ref.lower()) or _canon_table(tbl_ref)
            if canon_tbl is None:
                errors.append(ValidationError(
                    "unknown_table_qualifier",
                    f"column `{col_name}` references unknown table/alias `{tbl_ref}`",
                    detail=tbl_ref,
                ))
                continue
            if canon_tbl.startswith("__cte__:"):
                continue  # CTE columns unknown statically
            if _canon_column(canon_tbl, col_name) is None:
                errors.append(ValidationError(
                    "unknown_column",
                    f"column `{tbl_ref}.{col_name}` does not exist",
                    detail=f"{canon_tbl}.{col_name}",
                ))
        else:
            if has_cte_in_scope:
                continue  # permissive: unqualified column might be from CTE
            if col_name.lower() in select_aliases:
                continue  # ORDER BY / GROUP BY / HAVING reference to SELECT alias
            if not any(_canon_column(t, col_name) for t in base_in_scope):
                errors.append(ValidationError(
                    "unknown_column",
                    f"column `{col_name}` not found on any in-scope table",
                    detail=col_name,
                ))

    # 6. Joins use real FK relationships.
    #    For each JOIN ... ON <eq>, check that the equality's two sides
    #    together resolve to an FK pair in RELATIONSHIPS (either direction).
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if not on:
            # USING clause -> accept if the column matches a shared FK column
            using = j.args.get("using")
            if using:
                continue
            # cross join / no condition
            continue
        # Walk all equalities in the ON expression.
        eqs = list(on.find_all(exp.EQ))
        if not eqs:
            # Non-equi joins exist in valid SQL; we don't block them,
            # but we also can't verify them. Skip quietly.
            continue
        for eq in eqs:
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            lt = alias_to_table.get((left.table or "").lower()) or _canon_table(left.table or "")
            rt = alias_to_table.get((right.table or "").lower()) or _canon_table(right.table or "")
            # Skip FK checks involving CTE-sourced aliases.
            if (lt and lt.startswith("__cte__:")) or (rt and rt.startswith("__cte__:")):
                continue
            lc = _canon_column(lt, left.name) if lt else None
            rc = _canon_column(rt, right.name) if rt else None
            if not (lt and rt and lc and rc):
                # Column/table resolution already errored above; skip FK check.
                continue
            pair = tuple(sorted([(lt, lc), (rt, rc)]))
            key = (pair[0][0], pair[0][1], pair[1][0], pair[1][1])
            if key not in _FK_PAIRS:
                errors.append(ValidationError(
                    "invalid_join",
                    f"join `{lt}.{lc} = {rt}.{rc}` is not a declared FK relationship",
                    detail=f"{lt}.{lc} = {rt}.{rc}",
                ))

    ok = not errors
    return ValidationResult(
        ok=ok,
        errors=errors,
        tables_used=tables_used,
        is_select=is_select,
    )


# ---------------------------------------------------------------------------
# Optimizer (STRONG tier; single pass + revalidate + fallback)
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    sql: str                    # best SQL to use (optimized if valid, else original)
    original_sql: str
    optimized_sql: Optional[str]
    indexes: list[str]
    notes: list[str]
    used_fallback: bool         # True if optimized SQL failed revalidation
    fallback_reason: Optional[str] = None
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


_OPTIMIZER_SYSTEM = """\
You are an expert SQLite optimizer. You will receive a validated read-only SQL
query. Rewrite it for better readability and performance, and suggest helpful
indexes.

ABSOLUTE RULES:
1. Return STRICT JSON. No prose, no markdown fences.
2. Preserve the query's semantics exactly. Same result set, same columns, same ordering unless the user clearly wants it reordered.
3. Keep it read-only. No DELETE / UPDATE / INSERT / DROP / ALTER / TRUNCATE / PRAGMA.
4. Use only tables and columns from the provided schema. Never fabricate identifiers.
5. SQLite dialect. End the SQL with a single semicolon.

JSON schema:
{{
  "sql": "<optimized SQL ending with ;>",
  "indexes": ["CREATE INDEX ...", "..."],
  "notes": ["short note 1", "short note 2"]
}}

If the query is already optimal, return it unchanged with notes=["already optimal"].

SCHEMA:
```
{schema}
```
"""


def optimize_sql(
    original_sql: str,
    *,
    schema_ddl: str = CHINOOK_DDL,
    router: Optional[LLMRouter] = None,
) -> OptimizationResult:
    """Single-pass optimization with revalidate + fallback to original on failure."""
    router = router or LLMRouter()
    system = _OPTIMIZER_SYSTEM.format(schema=schema_ddl.strip())
    user = f"SQL to optimize:\n{original_sql.strip()}"

    raw = router.chat(Tier.STRONG, system=system, user=user, temperature=0.0)

    try:
        data = json.loads(_strip_json(raw))
    except (json.JSONDecodeError, ValueError):
        return OptimizationResult(
            sql=original_sql,
            original_sql=original_sql,
            optimized_sql=None,
            indexes=[],
            notes=[],
            used_fallback=True,
            fallback_reason="optimizer returned non-JSON output",
            raw=raw,
        )

    optimized = _clean_sql(data.get("sql", "") or "")
    indexes = [str(x) for x in (data.get("indexes") or []) if str(x).strip()]
    notes = [str(x) for x in (data.get("notes") or []) if str(x).strip()]

    if not optimized:
        return OptimizationResult(
            sql=original_sql,
            original_sql=original_sql,
            optimized_sql=None,
            indexes=indexes,
            notes=notes,
            used_fallback=True,
            fallback_reason="optimizer returned empty SQL",
            raw=raw,
        )

    # Revalidate the optimized SQL. If it fails, fall back to original.
    revalid = validate_sql(optimized)
    if not revalid.ok:
        reason = "optimized SQL failed revalidation: " + ", ".join(
            e.code for e in revalid.errors
        )
        return OptimizationResult(
            sql=original_sql,
            original_sql=original_sql,
            optimized_sql=optimized,
            indexes=indexes,
            notes=notes,
            used_fallback=True,
            fallback_reason=reason,
            raw=raw,
        )

    return OptimizationResult(
        sql=optimized,
        original_sql=original_sql,
        optimized_sql=optimized,
        indexes=indexes,
        notes=notes,
        used_fallback=False,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Executor (deterministic; SQLite with read-only guard + EXPLAIN QUERY PLAN)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    ok: bool
    columns: list[str]
    rows: list[list]            # JSON-safe
    row_count: int              # rows fetched (may equal row_cap)
    truncated: bool             # True if more rows exist beyond row_cap
    plan: list[str]             # EXPLAIN QUERY PLAN lines
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _json_safe(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return str(v)
    except Exception:  # noqa: BLE001
        return None


import os as _os_mod

_PROJECT_ROOT = _os_mod.path.dirname(_os_mod.path.dirname(_os_mod.path.realpath(__file__)))


def _resolve_db_path(path: str) -> str:
    """Make `path` absolute. Relative paths resolve against the project root
    (parent of `app/`), not the notebook / server CWD, so execution works
    the same whether called from a notebook, FastAPI, or Docker."""
    if _os_mod.path.isabs(path):
        return path
    return _os_mod.path.normpath(_os_mod.path.join(_PROJECT_ROOT, path))


def execute_sql(
    sql: str,
    *,
    db_path: Optional[str] = None,
    row_cap: int = 100,
) -> ExecutionResult:
    """Execute a validated SELECT against the Chinook SQLite DB.

    Uses a read-only URI connection as a defense-in-depth measure in addition
    to the validator. Returns up to `row_cap + 1` to detect truncation and
    then trims to `row_cap`.
    """
    import os as _os
    import sqlite3

    from urllib.parse import quote
    raw_path = db_path or _os.getenv("CHINOOK_DB_PATH", "data/Chinook_Sqlite.sqlite")
    abs_path = _resolve_db_path(raw_path)
    if not _os.path.exists(abs_path):
        return ExecutionResult(
            ok=False, columns=[], rows=[], row_count=0, truncated=False,
            plan=[], error=f"DB file not found at {abs_path}",
        )
    # SQLite URI needs special characters (spaces, etc.) percent-encoded.
    uri = f"file:{quote(abs_path, safe='/')}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        return ExecutionResult(
            ok=False, columns=[], rows=[], row_count=0,
            truncated=False, plan=[], error=f"could not open DB: {e}",
        )

    try:
        plan_rows: list[str] = []
        try:
            cur = con.execute(f"EXPLAIN QUERY PLAN {sql}")
            for r in cur.fetchall():
                # r is (id, parent, notused, detail) in SQLite
                plan_rows.append(str(r[-1]))
        except sqlite3.Error:
            plan_rows = []

        cur = con.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        raw = cur.fetchmany(row_cap + 1)
        truncated = len(raw) > row_cap
        raw = raw[:row_cap]
        rows = [[_json_safe(v) for v in r] for r in raw]
        return ExecutionResult(
            ok=True, columns=columns, rows=rows, row_count=len(rows),
            truncated=truncated, plan=plan_rows,
        )
    except sqlite3.Error as e:
        return ExecutionResult(
            ok=False, columns=[], rows=[], row_count=0,
            truncated=False, plan=[], error=str(e),
        )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Debugger (STRONG tier; for `debug_sql` intent)
# ---------------------------------------------------------------------------

@dataclass
class DebugResult:
    corrected_sql: str
    issue: str
    notes: list[str]
    used_fallback: bool
    fallback_reason: Optional[str] = None
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


_DEBUGGER_SYSTEM = """\
You are an expert SQL debugger for the Chinook SQLite database. The user
supplies a SQL query they say is broken. Identify the problem and return a
corrected, read-only, schema-correct version.

ABSOLUTE RULES:
1. Return STRICT JSON. No prose, no markdown fences.
2. The corrected SQL must be a read-only SELECT (optionally with WITH). No DELETE/UPDATE/INSERT/DROP/ALTER/TRUNCATE/PRAGMA.
3. Use only tables and columns from the provided schema. Never fabricate identifiers.
4. SQLite dialect. End the SQL with a single semicolon.
5. Keep the user's clear intent; do not silently redesign the query beyond what the fix requires.

JSON schema:
{{
  "issue": "<one short sentence describing what was wrong>",
  "corrected_sql": "<fixed SQL ending with ;>",
  "notes": ["bullet 1", "bullet 2"]
}}

SCHEMA:
```
{schema}
```
"""


def debug_sql(
    user_sql: str,
    *,
    user_hint: str = "",
    schema_ddl: str = CHINOOK_DDL,
    router: Optional[LLMRouter] = None,
) -> DebugResult:
    """Fix a user-supplied broken SQL query, with validation + safe fallback."""
    router = router or LLMRouter()
    system = _DEBUGGER_SYSTEM.format(schema=schema_ddl.strip())
    user = f"BROKEN SQL:\n{user_sql.strip()}\n\nUSER HINT:\n{user_hint.strip() or '(none)'}"

    raw = router.chat(Tier.STRONG, system=system, user=user, temperature=0.0)
    try:
        data = json.loads(_strip_json(raw))
    except (json.JSONDecodeError, ValueError):
        return DebugResult(
            corrected_sql="",
            issue="debugger returned non-JSON output",
            notes=[],
            used_fallback=True,
            fallback_reason="non-JSON",
            raw=raw,
        )

    corrected = _clean_sql(data.get("corrected_sql", "") or "")
    issue = str(data.get("issue", "")).strip() or "unspecified issue"
    notes = [str(x) for x in (data.get("notes") or []) if str(x).strip()]

    if not corrected:
        return DebugResult(
            corrected_sql="",
            issue=issue,
            notes=notes,
            used_fallback=True,
            fallback_reason="empty corrected SQL",
            raw=raw,
        )

    reval = validate_sql(corrected)
    if not reval.ok:
        reason = "corrected SQL failed validation: " + ", ".join(
            e.code for e in reval.errors
        )
        return DebugResult(
            corrected_sql=corrected,
            issue=issue,
            notes=notes,
            used_fallback=True,
            fallback_reason=reason,
            raw=raw,
        )

    return DebugResult(
        corrected_sql=corrected,
        issue=issue,
        notes=notes,
        used_fallback=False,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Explainer (STRONG tier; 1-3 plain-English lines; reads whatever we pass)
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    text: str
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        return {"text": self.text}


_EXPLAINER_SYSTEM = """\
You explain SQL queries to a non-technical business user in 1-3 short
sentences of plain English.

Rules:
- Describe what the query retrieves and any filters, joins, groupings, or
  ordering that materially change the result.
- Mention optimizations only if they would surprise the user (e.g. a redundant
  join was removed). Keep it brief.
- No code fences. No bullet points. Just prose.
- Do NOT restate the SQL text itself.
"""


def explain_sql(
    sql: str,
    *,
    optimization_notes: Optional[list[str]] = None,
    executed: bool = False,
    row_count: Optional[int] = None,
    router: Optional[LLMRouter] = None,
) -> ExplanationResult:
    """Return a 1-3 sentence plain-English explanation of `sql`."""
    router = router or LLMRouter()
    extras = []
    if optimization_notes:
        extras.append("Optimizer notes: " + " | ".join(optimization_notes))
    if executed and row_count is not None:
        extras.append(f"Executed against the database; returned {row_count} row(s).")
    extras_block = "\n".join(extras) if extras else "(none)"
    user = f"SQL:\n{sql.strip()}\n\nExtras:\n{extras_block}"
    raw = router.chat(Tier.STRONG, system=_EXPLAINER_SYSTEM, user=user,
                      temperature=0.0)
    return ExplanationResult(text=raw.strip(), raw=raw)

