#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import List, Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sql_agent import SqlAgent
from src.config import SETTINGS


load_dotenv()

st.set_page_config(page_title="DB Intelligence - Query Tester", layout="wide")
st.title("DB Intelligence - Query Tester")

with st.sidebar:
    st.header("Settings")
    db_url = st.text_input("DATABASE_URL", value=SETTINGS.database_url, type="password")
    default_schemas = SETTINGS.db_schemas or ["public"]
    schema_text = st.text_input("Schemas (comma-separated)", value=", ".join(default_schemas))
    row_limit = st.number_input("Row limit", min_value=1, max_value=10000, value=SETTINGS.row_limit_default)
    run_button = st.button("Run query", type="primary")

question = st.text_area("Enter your question", height=120, placeholder="e.g., active users with role names")

st.divider()

if run_button:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Apply runtime overrides for this session
    if db_url and db_url != SETTINGS.database_url:
        os.environ["DATABASE_URL"] = db_url
    if row_limit != SETTINGS.row_limit_default:
        os.environ["ROW_LIMIT_DEFAULT"] = str(row_limit)

    # Parse schemas
    schemas: Optional[List[str]] = None
    if schema_text.strip():
        schemas = [s.strip() for s in schema_text.split(",") if s.strip()]

    try:
        agent = SqlAgent()
        resp = agent.answer_question(question=question, schemas=schemas)

        st.subheader("Generated SQL")
        st.code(resp.sql, language="sql")

        st.subheader(f"Result ({len(resp.result.rows)} rows)")
        if resp.result.columns:
            rows_as_dicts: List[Dict[str, Any]] = []
            for r in resp.result.rows:
                rows_as_dicts.append({col: val for col, val in zip(resp.result.columns, r)})
            st.dataframe(rows_as_dicts, use_container_width=True)
        else:
            st.info("Query returned no columns. Rendering raw rows:")
            st.write(resp.result.rows)

        if resp.answer:
            st.subheader("Answer")
            st.markdown(resp.answer)

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.caption("Tip: set DEBUG_RETRIEVAL=true before running to log table/column subscores in the app logs.") 