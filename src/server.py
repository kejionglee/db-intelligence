from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException

from .sql_agent import SqlAgent


class QueryRequest(TypedDict, total=False):
    question: str
    schemas: Optional[List[str]]


class QueryResponse(TypedDict):
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    answer: str


app = FastAPI(title="DB Intelligence", version="0.1.0")
_agent: Optional[SqlAgent] = None


def _get_agent() -> SqlAgent:
    global _agent
    if _agent is None:
        _agent = SqlAgent()
    return _agent


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    question = (req.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required")

    schemas = req.get("schemas")
    try:
        agent = _get_agent()
        resp = agent.answer_question(question=question, schemas=schemas)
        return {
            "sql": resp.sql,
            "columns": resp.result.columns,
            "rows": [list(r) for r in resp.result.rows],
            "row_count": len(resp.result.rows),
            "answer": resp.answer or "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 