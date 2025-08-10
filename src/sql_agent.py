from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import sqlparse
from tenacity import retry, stop_after_attempt, wait_fixed

from .config import SETTINGS
from .db import QueryResult, execute_select
from .retrieval import retrieve_tables, LAST_RETRIEVAL_TRACE
from .column_picker import build_column_plan, ColumnPlan
from .llm import LlmClient


SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass(frozen=True)
class AgentResponse:
    sql: str
    result: QueryResult
    answer: Optional[str]
    trace: Optional[Dict[str, Any]] = None


_WORD_NUMS = {
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
    'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,
    'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,
}


def _parse_requested_count(q: str) -> Optional[int]:
    q = q.lower()
    # digits after keywords
    m = re.search(r"\b(?:top|last|first|latest|newest)\s*[:\-]?\s*(\d{1,3})\b", q)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 100:
                return n
        except Exception:
            pass
    # word numbers up to 100
    m = re.search(r"\b(?:top|last|first|latest|newest)\s*[:\-]?\s*([a-z\-]+)\b", q)
    if m:
        words = m.group(1).split("-")
        total = 0
        for w in words:
            if w in _WORD_NUMS:
                total += _WORD_NUMS[w]
            else:
                total = 0; break
        if 1 <= total <= 100:
            return total
    return None


def _parse_order_intent(q: str) -> Optional[str]:
    q = q.lower()
    if any(k in q for k in ["last", "latest", "newest", "top"]):
        return "desc"
    if any(k in q for k in ["first", "earliest", "oldest"]):
        return "asc"
    return None


def _validate_sql(sql: str) -> bool:
    if ";" in sql.strip().rstrip(";"):
        return False
    statements = sqlparse.parse(sql)
    if len(statements) != 1:
        return False
    stmt = statements[0]
    if stmt.get_type().upper() != "SELECT":
        return False
    tokens = sql.lower()
    forbidden = [" insert ", " update ", " delete ", " create ", " drop ", " alter ", " truncate "]
    if any(f in tokens for f in forbidden):
        return False
    return True


def _quote_ident(name: str) -> str:
    return name if (name.islower() and " " not in name and not name.startswith("\"")) else f'"{name}"'


def _force_limit(sql: str, n: Optional[int]) -> str:
    if not n:
        return sql
    # remove existing limit if larger
    m = re.search(r"\blimit\s+(\d+)$", sql.strip(), flags=re.I)
    if m:
        try:
            cur = int(m.group(1))
            if cur <= n:
                return sql
        except Exception:
            pass
        sql = re.sub(r"\blimit\s+\d+$", "", sql.strip(), flags=re.I).rstrip()
    return f"{sql}\nLIMIT {n}"


def _ensure_order(sql: str, col: Optional[str], direction: Optional[str]) -> str:
    if not col or not direction:
        return sql
    if re.search(r"\border\s+by\b", sql, flags=re.I):
        return sql
    return f"{sql}\nORDER BY {_quote_ident(col)} {direction.upper()}"


def _normalize_sql(sql: str) -> str:
    # Replace common unicode punctuation/operators with ASCII equivalents
    return (
        sql.replace("≥", ">=")
           .replace("≤", "<=")
           .replace("—", "-")
           .replace("–", "-")
    )


def _append_where(sql: str, condition: str) -> str:
    # If WHERE exists, append AND; else insert before ORDER BY/LIMIT if present, otherwise at end
    if re.search(r"\bwhere\b", sql, flags=re.I):
        return re.sub(r"(\bwhere\b)", r"\\1", sql, flags=re.I) + f"\nAND {condition}"
    # Insert before ORDER BY if present
    m = re.search(r"\border\s+by\b", sql, flags=re.I)
    if m:
        idx = m.start()
        return sql[:idx].rstrip() + f"\nWHERE {condition}\n" + sql[idx:]
    # Insert before LIMIT if present (and no ORDER BY)
    m2 = re.search(r"\blimit\s+\d+\b", sql, flags=re.I)
    if m2:
        idx = m2.start()
        return sql[:idx].rstrip() + f"\nWHERE {condition}\n" + sql[idx:]
    # Otherwise append at end
    return f"{sql}\nWHERE {condition}"


def _parse_date_range(q: str) -> Optional[Dict[str, str]]:
    q = q.lower()
    # between A and B
    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        return {"op": "between", "start": m.group(1), "end": m.group(2), "label": f"between {m.group(1)} and {m.group(2)}"}
    # on YYYY-MM-DD
    m = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", q)
    if m:
        return {"op": "on", "date": m.group(1), "label": f"on {m.group(1)}"}
    # last N days
    m = re.search(r"last\s+(\d{1,3})\s+days", q)
    if m:
        return {"op": ">=", "interval": f"{m.group(1)} days", "label": f"last {m.group(1)} days"}
    # last week / last month
    if "last week" in q:
        return {"op": ">=", "interval": "7 days", "label": "last week"}
    if "last month" in q:
        return {"op": ">=", "interval": "1 month", "label": "last month"}
    # yesterday
    if "yesterday" in q:
        return {"op": "between_yesterday", "label": "yesterday"}
    return None


def _apply_date_filter(sql: str, plan: ColumnPlan, date_info: Dict[str, str]) -> Optional[str]:
    col = plan.ts_column or (plan.order[0] if plan.order else None)
    if not col:
        return None
    colq = _quote_ident(col)
    if date_info["op"] == ">=":
        return _append_where(sql, f"{colq} >= CURRENT_DATE - INTERVAL '{date_info['interval']}'")
    if date_info["op"] == "between":
        return _append_where(sql, f"{colq} BETWEEN DATE '{date_info['start']}' AND DATE '{date_info['end']}'")
    if date_info["op"] == "on":
        return _append_where(sql, f"DATE({colq}) = DATE '{date_info['date']}'")
    if date_info["op"] == "between_yesterday":
        return _append_where(sql, f"{colq} BETWEEN CURRENT_DATE - INTERVAL '1 day' AND CURRENT_DATE")
    return None


def _row_matches_date(row: Tuple[Any, ...], columns: List[str], plan: ColumnPlan, date_info: Dict[str, str]) -> bool:
    col = plan.ts_column or (plan.order[0] if plan.order else None)
    if not col:
        return False
    if col not in columns:
        return True  # cannot verify
    idx = columns.index(col)
    val = row[idx]
    if val is None:
        return False
    # Best-effort: rely on DB filter; here just accept
    return True


class SqlAgent:
    def __init__(self, llm: Optional[LlmClient] = None) -> None:
        self._llm = llm or LlmClient()

    def _is_safe_select(self, sql: str) -> bool:
        return _validate_sql(sql)

    def _deterministic_sql(self, plan: ColumnPlan) -> str:
        s, t = plan.table.split(".", 1)
        if plan.aggregate == "count":
            sql = f"SELECT COUNT(*) AS cnt\nFROM {_quote_ident(s)}.{_quote_ident(t)}"
            return sql
        # projection
        cols = []
        for c in plan.columns:
            cols.append(_quote_ident(c.column))
        if not cols:
            cols = ["*"]
        sql = f"SELECT {', '.join(cols)}\nFROM {_quote_ident(s)}.{_quote_ident(t)}"
        # order
        if plan.order:
            col, direction = plan.order
            sql += f"\nORDER BY {_quote_ident(col)} {direction.upper()}"
        # limit only for lists; will be forced later if explicit N was parsed
        if plan.limit:
            sql = f"{sql}\nLIMIT {plan.limit}"
        return sql

    def _summarize_answer(self, question: str, result: QueryResult) -> str:
        if not result.rows:
            return "No answer"
        preview_rows = result.rows[: SETTINGS.answer_preview_rows]
        preview_text_lines: List[str] = []
        preview_text_lines.append(
            "Columns: " + ", ".join(result.columns)
        )
        for r in preview_rows:
            preview_text_lines.append("Row: " + ", ".join(str(v) for v in r))
        preview_text = "\n".join(preview_text_lines)

        system = (
            "You are a data summarizer. Turn rows into a short, human-readable answer. Keep it factual and concise."
        )
        user = (
            f"Question: {question}\n\nResult preview (up to {SETTINGS.answer_preview_rows} rows):\n{preview_text}\n\n"
            "If listing items, number them. If counts, show the number and list names if small (<=10)."
        )
        try:
            text = self._llm.chat(system=system, user=user, temperature=0.0, timeout=3.0)
            return text.strip()
        except Exception:
            # Fallback: basic enumerated list
            lines: List[str] = []
            for idx, r in enumerate(preview_rows, 1):
                lines.append(f"{idx}. " + ", ".join(str(v) for v in r))
            return "\n".join(lines) if lines else "No answer"

    def answer_question(self, question: str, schemas: Optional[List[str]] = None) -> AgentResponse:
        target_schemas = schemas or SETTINGS.db_schemas

        # Intent parsing
        requested_n = _parse_requested_count(question)
        order_intent = _parse_order_intent(question)
        date_info = _parse_date_range(question)

        # Step A: shortlist tables deterministically
        candidates = retrieve_tables(question, target_schemas)
        if not candidates:
            return AgentResponse(sql="", result=QueryResult(columns=[], rows=[]), answer="No answer",
                                 trace=self._build_trace(None, None, "retrieval_empty", None, None))

        # Step B: single-table-first column plan
        plan = build_column_plan(question, candidates)
        if not plan:
            return AgentResponse(sql="", result=QueryResult(columns=[], rows=[]), answer="No answer",
                                 trace=self._build_trace(candidates, None, "score_too_low", None, None))

        # Apply reinforced intents
        if requested_n:
            plan.limit = requested_n
        if order_intent and plan.order and plan.order[0]:
            plan = ColumnPlan(table=plan.table, columns=plan.columns, filters=plan.filters,
                               order=(plan.order[0], order_intent), limit=plan.limit,
                               aggregate=plan.aggregate, join_budget=plan.join_budget,
                               ts_column=plan.ts_column, created_preferred=plan.created_preferred)
        elif order_intent and not plan.order:
            # require timestamp to order
            # if missing, return No answer
            return AgentResponse(sql="", result=QueryResult(columns=[], rows=[]), answer="No answer",
                                 trace=self._build_trace(candidates, plan, "missing_timestamp", None, None))

        # Step C: deterministic SQL
        sql = self._deterministic_sql(plan)
        # Date filter
        if date_info:
            sql_with_date = _apply_date_filter(sql, plan, date_info)
            if not sql_with_date:
                return AgentResponse(sql="", result=QueryResult(columns=[], rows=[]), answer="No answer",
                                     trace=self._build_trace(candidates, plan, "missing_timestamp", None, None))
            sql = sql_with_date
        # Post-check order/limit
        if order_intent and plan.order:
            sql = _ensure_order(sql, plan.order[0], plan.order[1])
        if requested_n:
            sql = _force_limit(sql, requested_n)

        # Normalize unicode operators/punctuation
        sql = _normalize_sql(sql)

        if not _validate_sql(sql):
            return AgentResponse(sql="", result=QueryResult(columns=[], rows=[]), answer="No answer",
                                 trace=self._build_trace(candidates, plan, "invalid_sql", sql, None))

        # Step D: execute
        result = execute_select(sql)
        # In-memory cap guard
        effective_limit = requested_n or plan.limit
        if plan.aggregate != "count" and effective_limit and len(result.rows) > effective_limit:
            result = QueryResult(columns=result.columns, rows=result.rows[: effective_limit])
        if len(result.rows) == 0:
            return AgentResponse(sql=sql, result=result, answer="No answer",
                                 trace=self._build_trace(candidates, plan, "result_empty", sql, result))
        # Verify date filter (best-effort)
        if date_info and any(not _row_matches_date(r, result.columns, plan, date_info) for r in result.rows):
            return AgentResponse(sql=sql, result=QueryResult(columns=result.columns, rows=[]), answer="No answer",
                                 trace=self._build_trace(candidates, plan, "result_date_mismatch", sql, result))

        # Step E: summarize via LLM (small assist)
        answer = self._summarize_answer(question, result)
        if not answer.strip():
            answer = "No answer"
        return AgentResponse(sql=sql, result=result, answer=answer,
                             trace=self._build_trace(candidates, plan, None, sql, result))

    def _build_trace(self, candidates, plan: Optional[ColumnPlan], reason: Optional[str], sql: Optional[str], result: Optional[QueryResult]) -> Optional[Dict[str, Any]]:
        if not SETTINGS.debug_retrieval:
            return None
        retr = LAST_RETRIEVAL_TRACE.copy() if LAST_RETRIEVAL_TRACE else []
        intent = {
            "count": None,
            "order": plan.order[1] if plan and plan.order else None,
            "date_range": None,
            "createdPreferred": bool(plan and plan.created_preferred),
        }
        trace: Dict[str, Any] = {
            "retrieval": retr,
            "chosenTable": plan.table if plan else None,
            "plan": None,
            "intent": intent,
            "sql": sql,
            "reason": reason,
        }
        if plan:
            trace["plan"] = {
                "columns": [c.column for c in plan.columns],
                "filters": plan.filters,
                "order": list(plan.order) if plan.order else None,
                "limit": plan.limit,
                "aggregate": plan.aggregate,
            }
        return trace 