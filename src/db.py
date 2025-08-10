from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict

import psycopg
from psycopg_pool import ConnectionPool

from .config import SETTINGS


logger = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not SETTINGS.database_url:
            raise RuntimeError("DATABASE_URL is not set. Configure it in your environment or .env file.")
        _pool = ConnectionPool(SETTINGS.database_url, min_size=1, max_size=5, timeout=10)
    return _pool


@dataclass(frozen=True)
class QueryResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    comment: Optional[str]
    is_pk: bool
    is_fk: bool


def execute_select(sql: str, parameters: Optional[Sequence[Any]] = None) -> QueryResult:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, parameters or [])
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description] if cur.description else []
            return QueryResult(columns=columns, rows=rows)


def fetch_schema_summary(schemas: Optional[Iterable[str]] = None, max_columns_per_table: int = 50) -> str:
    target_schemas = list(schemas) if schemas else SETTINGS.db_schemas
    sql = """
    with cols as (
      select table_schema, table_name, column_name, data_type,
             ordinal_position,
             is_nullable,
             column_default
      from information_schema.columns
      where table_schema = any(%s)
    ), pks as (
      select
        kcu.table_schema,
        kcu.table_name,
        kcu.column_name
      from information_schema.table_constraints tc
      join information_schema.key_column_usage kcu
        on tc.constraint_name = kcu.constraint_name
       and tc.table_schema = kcu.table_schema
      where tc.constraint_type = 'PRIMARY KEY'
    ), fks as (
      select
        kcu.table_schema,
        kcu.table_name,
        kcu.column_name,
        ccu.table_schema as foreign_table_schema,
        ccu.table_name as foreign_table_name,
        ccu.column_name as foreign_column_name
      from information_schema.table_constraints tc
      join information_schema.key_column_usage kcu
        on tc.constraint_name = kcu.constraint_name
       and tc.table_schema = kcu.table_schema
      join information_schema.constraint_column_usage ccu
        on ccu.constraint_name = tc.constraint_name
       and ccu.table_schema = tc.table_schema
      where tc.constraint_type = 'FOREIGN KEY'
    )
    select c.table_schema,
           c.table_name,
           string_agg(
             c.column_name || ' ' || c.data_type ||
             case when pks.column_name is not null then ' PK' else '' end ||
             case when fks.column_name is not null then (' FK->' || fks.foreign_table_schema || '.' || fks.foreign_table_name || '(' || fks.foreign_column_name || ')') else '' end,
             ', ' order by c.ordinal_position
           ) as columns_summary
    from cols c
    left join pks on pks.table_schema = c.table_schema and pks.table_name = c.table_name and pks.column_name = c.column_name
    left join fks on fks.table_schema = c.table_schema and fks.table_name = c.table_name and fks.column_name = c.column_name
    group by c.table_schema, c.table_name
    order by c.table_schema, c.table_name
    """

    res = execute_select(sql, [target_schemas])

    lines: List[str] = []
    for schema, table, columns_summary in res.rows:
        column_parts = [p.strip() for p in str(columns_summary).split(',') if p.strip()]
        if len(column_parts) > max_columns_per_table:
            column_parts = column_parts[:max_columns_per_table] + [f"... (+{len(column_parts) - max_columns_per_table} more columns)"]
        columns_summary_trunc = ", ".join(column_parts)
        lines.append(f"{schema}.{table}({columns_summary_trunc})")

    prefix = f"Schemas: {', '.join(target_schemas)}\n"
    summary = prefix + "\n".join(lines)

    logger.info("fetch_schema_summary output:\n%s", summary)

    return summary


def fetch_columns(schema: str, table: str) -> List[ColumnInfo]:
    """Fetch column information for a specific table from dbi_table_columns."""
    sql = """
    SELECT column_name, column_type, column_comment, is_pk, is_fk
    FROM dbi_table_columns
    WHERE schema_name = %s AND table_name = %s
    ORDER BY column_name
    """
    
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (schema, table))
            rows = cur.fetchall()
            return [
                ColumnInfo(
                    name=row[0],
                    type=row[1],
                    comment=row[2],
                    is_pk=row[3],
                    is_fk=row[4]
                )
                for row in rows
            ]


def get_table_neighbors(schema: str, table: str) -> List[str]:
    """Get FK neighbors of a table (1-hop) using information_schema.
    Includes both outgoing FKs (this table -> other) and incoming FKs (other -> this table).
    """
    sql = """
    (
      SELECT DISTINCT ccu.table_schema AS neighbor_schema, ccu.table_name AS neighbor_table
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
       AND ccu.table_schema = tc.table_schema
      WHERE tc.constraint_type = 'FOREIGN KEY'
        AND kcu.table_schema = %s
        AND kcu.table_name  = %s
    )
    UNION
    (
      SELECT DISTINCT kcu.table_schema AS neighbor_schema, kcu.table_name AS neighbor_table
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
       AND ccu.table_schema = tc.table_schema
      WHERE tc.constraint_type = 'FOREIGN KEY'
        AND ccu.table_schema = %s
        AND ccu.table_name  = %s
    )
    """

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (schema, table, schema, table))
            rows = cur.fetchall()
            neighbors = {f"{r[0]}.{r[1]}" for r in rows if r and r[0] and r[1]}
            return sorted(neighbors)


def trigram_similarity(a: str, b: str) -> float:
    """Compute trigram similarity between two strings using pg_trgm."""
    sql = "SELECT greatest(similarity(%s, %s), similarity(%s, %s)) AS sim"
    
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (a.lower(), b.lower(), b.lower(), a.lower()))
            result = cur.fetchone()
            return float(result[0]) if result and result[0] is not None else 0.0


def search_tables_by_name(query: str, schemas: List[str], limit: int = 40) -> List[Tuple[str, str, float]]:
    """Search tables by name using trigram similarity."""
    sql = """
    SELECT schema_name, table_name, 
           greatest(similarity(lower(schema_name||'.'||table_name), %s),
                    similarity(%s, lower(schema_name||'.'||table_name))) AS name_sim
    FROM dbi_table_docs
    WHERE schema_name = ANY(%s)
    ORDER BY name_sim DESC
    LIMIT %s
    """
    
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query.lower(), query.lower(), schemas, limit))
            rows = cur.fetchall()
            return [(row[0], row[1], float(row[2])) for row in rows]


def search_tables_by_columns(query: str, schemas: List[str], limit: int = 40) -> List[Tuple[str, str, float]]:
    """Search tables by column names using trigram similarity."""
    sql = """
    SELECT DISTINCT t.schema_name, t.table_name,
           MAX(greatest(similarity(lower(c.column_name), %s),
                       similarity(%s, lower(c.column_name)))) AS col_sim
    FROM dbi_table_docs t
    JOIN dbi_table_columns c ON t.schema_name = c.schema_name AND t.table_name = c.table_name
    WHERE t.schema_name = ANY(%s)
    GROUP BY t.schema_name, t.table_name
    HAVING MAX(greatest(similarity(lower(c.column_name), %s),
                       similarity(%s, lower(c.column_name)))) > 0.3
    ORDER BY col_sim DESC
    LIMIT %s
    """
    
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query.lower(), query.lower(), schemas, query.lower(), query.lower(), limit))
            rows = cur.fetchall()
            return [(row[0], row[1], float(row[2])) for row in rows]


@contextlib.contextmanager
def connection_cursor():
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield conn, cur 


def fetch_fk_relationships(schema: str, table: str) -> List[Tuple[str, str, str, str]]:
    """Return list of (local_column, ref_schema, ref_table, ref_column) for outgoing FKs."""
    sql = """
    SELECT kcu.column_name AS local_column,
           ccu.table_schema AS ref_schema,
           ccu.table_name  AS ref_table,
           ccu.column_name AS ref_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND kcu.table_schema = %s
      AND kcu.table_name  = %s
    """
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (schema, table))
            rows = cur.fetchall()
            return [(r[0], r[1], r[2], r[3]) for r in rows]


def infer_type_family(type_str: str) -> str:
    t = (type_str or "").lower()
    if any(x in t for x in ["timestamp", "timestamptz", "date", "time"]):
        return "timestamp"
    if any(x in t for x in ["int", "bigint", "numeric", "decimal", "double", "float", "real"]):
        return "numeric"
    if any(x in t for x in ["char", "text", "uuid", "json", "name"]):
        return "text"
    return "other" 