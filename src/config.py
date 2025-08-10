from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


load_dotenv()


def _get_env_list(name: str, default: str) -> List[str]:
    value = os.getenv(name, default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError:
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class Settings:
    database_url: str
    ollama_host: str
    ollama_model: str
    db_schemas: List[str]
    row_limit_default: int
    
    # Retrieval settings
    retr_top_vector: int
    retr_top_keyword: int
    retr_final_top_k: int
    retr_min_score: float
    
    # Scoring weights
    w_name: float
    w_col: float
    w_syn: float
    w_graph: float
    w_vec: float
    
    # Penalties
    p_no_token_hit: float
    p_noise: float
    
    # Debug
    debug_retrieval: bool

    # Column picking and formatting
    answer_preview_rows: int
    time_format: str
    ollama_model_chat: str
    ollama_model_embed: str
    col_score_min: float
    top_cols_per_table: int
    single_table_confidence: float

    @staticmethod
    def load() -> "Settings":
        database_url = os.getenv("DATABASE_URL", "").strip()
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
        ollama_model = os.getenv("OLLAMA_MODEL", os.getenv("OPENAI_MODEL", "llama3.1:8b")).strip()
        db_schemas = _get_env_list("DB_SCHEMAS", "public")
        row_limit_default = _get_env_int("ROW_LIMIT_DEFAULT", 10)

        # Retrieval settings
        retr_top_vector = _get_env_int("RETR_TOP_VECTOR", 40)
        retr_top_keyword = _get_env_int("RETR_TOP_KEYWORD", 40)
        retr_final_top_k = _get_env_int("RETR_FINAL_TOP_K", 6)
        retr_min_score = _get_env_float("RETR_MIN_SCORE", 0.6)
        
        # Scoring weights
        w_name = _get_env_float("W_NAME", 1.2)
        w_col = _get_env_float("W_COL", 0.6)
        w_syn = _get_env_float("W_SYN", 0.6)
        w_graph = _get_env_float("W_GRAPH", 0.3)
        w_vec = _get_env_float("W_VEC", 0.2)
        
        # Penalties
        p_no_token_hit = _get_env_float("P_NO_TOKEN_HIT", 2.0)
        p_noise = _get_env_float("P_NOISE", 0.5)
        
        # Debug
        debug_retrieval = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"

        # Column picking & formatting
        answer_preview_rows = _get_env_int("ANSWER_PREVIEW_ROWS", 20)
        time_format = os.getenv("TIME_FORMAT", "%d/%m/%y %-I:%M%p")
        ollama_model_chat = os.getenv("OLLAMA_MODEL_CHAT", "deepseek-r1")
        ollama_model_embed = os.getenv("OLLAMA_MODEL_EMBED", os.getenv("OLLAMA_MODEL", "nomic-embed-text"))
        col_score_min = _get_env_float("COL_SCORE_MIN", 0.8)
        top_cols_per_table = _get_env_int("TOP_COLS_PER_TABLE", 3)
        single_table_confidence = _get_env_float("SINGLE_TABLE_CONFIDENCE", 0.75)

        return Settings(
            database_url=database_url,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            db_schemas=db_schemas,
            row_limit_default=row_limit_default,
            retr_top_vector=retr_top_vector,
            retr_top_keyword=retr_top_keyword,
            retr_final_top_k=retr_final_top_k,
            retr_min_score=retr_min_score,
            w_name=w_name,
            w_col=w_col,
            w_syn=w_syn,
            w_graph=w_graph,
            w_vec=w_vec,
            p_no_token_hit=p_no_token_hit,
            p_noise=p_noise,
            debug_retrieval=debug_retrieval,
            answer_preview_rows=answer_preview_rows,
            time_format=time_format,
            ollama_model_chat=ollama_model_chat,
            ollama_model_embed=ollama_model_embed,
            col_score_min=col_score_min,
            top_cols_per_table=top_cols_per_table,
            single_table_confidence=single_table_confidence,
        )


SETTINGS = Settings.load() 