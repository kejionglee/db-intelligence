from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import SETTINGS
from .db import fetch_columns, fetch_fk_relationships, infer_type_family
from .retrieval import TOKEN_RE
from .llm import LlmClient

DISPLAY_COLS = ["name", "title", "label", "description", "code"]

SYN: Dict[str, List[str]] = {
    "user":   ["user","usr","customer","account","member","employee","person"],
    "role":   ["role","group","permission"],
    "login":  ["login","signin","last_login","last_seen","last_sign_in_at"],
    "active": ["active","is_active","enabled","status"],
    "alarm":  ["alarm","event","incident","alert"],
    "access": ["access","access_control","door","reader","badge","credential"],
    "door":   ["door","entry","gate","turnstile"],
    "time":   ["time","timestamp","event_time","created_at","triggered_at","occurred_at"],
    "name":   ["name","title","label","description"],
    "count":  ["count","total","number"],
}


@dataclass
class ColumnRef:
    schema: str
    table: str
    column: str
    type: str
    is_pk: bool
    is_fk: bool
    comment: Optional[str] = None


@dataclass
class ColumnPlan:
    table: str  # fqtn
    columns: List[ColumnRef]
    filters: Dict[str, Any]
    order: Optional[Tuple[str, str]]  # (column, 'asc'|'desc')
    limit: Optional[int]
    aggregate: Optional[str]  # 'count' or None
    join_budget: int
    ts_column: Optional[str]
    created_preferred: bool


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _is_ts(col: ColumnRef) -> bool:
    if infer_type_family(col.type) in ("timestamp", "date"):
        return True
    name = col.column.lower()
    return any(k in name for k in ["occurred_at", "event_time", "created_at", "created_on", "triggered_at", "updated_at", "timestamp", "time", "date", "_at"])


def _is_label(col: ColumnRef) -> bool:
    return any(lbl in col.column.lower() for lbl in DISPLAY_COLS)


def _best_ts(cols: List[ColumnRef]) -> Optional[ColumnRef]:
    ts = [c for c in cols if _is_ts(c)]
    if not ts:
        return None
    # prefer created_at/on, event_time, occurred_at
    for pref in ["created_at", "created_on", "event_time", "occurred_at", "timestamp", "updated_at", "time", "date"]:
        for c in ts:
            if pref in c.column.lower():
                return c
    return ts[0]


def _best_label(cols: List[ColumnRef]) -> Optional[ColumnRef]:
    labels = [c for c in cols if _is_label(c) and infer_type_family(c.type) == "text"]
    if not labels:
        return None
    for pref in DISPLAY_COLS:
        for c in labels:
            if pref in c.column.lower():
                return c
    return labels[0]


def _best_key(cols: List[ColumnRef]) -> Optional[ColumnRef]:
    for c in cols:
        if c.is_pk or c.column.lower() in ("id", f"{c.table}_id"):
            return c
    return None


def build_column_plan(question: str, shortlist: List[Tuple[str, str, float]]) -> Optional[ColumnPlan]:
    q = question.lower()
    q_tokens = _tokens(question)
    need_time = any(t in q for t in ["last", "latest", "yesterday", "recent", "today", "this week", "past ", "first", "earliest", "oldest"])
    need_count = any(kw in q for kw in ["how many", "count", "number of"])
    need_names = any(kw in q for kw in ["who", "names", "list"])

    if not shortlist:
        return None

    # Pick the top-scoring table; ensure exceeds confidence threshold
    schema, table, score = shortlist[0]
    if score < SETTINGS.single_table_confidence:
        return None

    cols_raw = fetch_columns(schema, table)
    cols = [ColumnRef(schema=schema, table=table, column=c.name, type=c.type, is_pk=c.is_pk, is_fk=c.is_fk, comment=c.comment) for c in cols_raw]

    # Choose minimal set of columns
    pick: List[ColumnRef] = []

    ts_col = _best_ts(cols)
    if ts_col and need_time:
        pick.append(ts_col)

    label_col = _best_label(cols) if need_names or not need_count else None
    if label_col and all(label_col.column != c.column for c in pick):
        pick.append(label_col)

    key_col = _best_key(cols)
    if key_col and all(key_col.column != c.column for c in pick):
        pick.append(key_col)

    # If no columns selected, fallback to first few text columns
    if not pick:
        for c in cols:
            if infer_type_family(c.type) == "text":
                pick.append(c)
                if len(pick) >= 2:
                    break
        if not pick and cols:
            pick.append(cols[0])

    # Order and limit
    order: Optional[Tuple[str, str]] = None
    limit: Optional[int] = None
    if need_time and ts_col:
        if any(t in q for t in ["last", "latest", "recent", "newest", "top"]):
            order = (ts_col.column, "desc")
        elif any(t in q for t in ["first", "earliest", "oldest"]):
            order = (ts_col.column, "asc")
    if not need_count:
        limit = SETTINGS.row_limit_default

    aggregate: Optional[str] = None
    if need_count:
        aggregate = "count"
        # For counts, no limit/order
        order = None
        limit = None

    created_pref = bool(ts_col and ts_col.column.lower().startswith("created"))

    return ColumnPlan(
        table=f"{schema}.{table}",
        columns=pick,
        filters={},
        order=order,
        limit=limit,
        aggregate=aggregate,
        join_budget=0,
        ts_column=ts_col.column if ts_col else None,
        created_preferred=created_pref,
    ) 