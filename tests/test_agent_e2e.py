from __future__ import annotations

import os
import pytest

from src.config import SETTINGS
from src.sql_agent import SqlAgent


@pytest.mark.skipif(not SETTINGS.database_url, reason="DATABASE_URL not configured")
@pytest.mark.parametrize("question", [
    "Give me the last alarms associated with access control within the last 5 days.",
])
def test_agent_runs(question: str):
    agent = SqlAgent()
    resp = agent.answer_question(question=question)
    assert isinstance(resp.sql, str) and resp.sql.lower().startswith("select"), "SQL should be a single SELECT"
    assert isinstance(resp.result.rows, list)
    assert isinstance(resp.result.columns, list)
    assert isinstance(resp.answer, str)
    # Ensure limit is enforced unless aggregate-only
    if " group by " not in resp.sql.lower() and "count(" not in resp.sql.lower():
        assert f"limit {SETTINGS.row_limit_default}" in resp.sql.lower() 