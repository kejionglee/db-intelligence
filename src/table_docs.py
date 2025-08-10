"""
Table documentation builder for deterministic retrieval system.
Builds both dbi_table_docs (with embeddings) and dbi_table_columns registry.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine

from .config import SETTINGS
from .llm import embed

logger = logging.getLogger(__name__)


def _to_sqlalchemy_url(url: str) -> str:
    """Ensure SQLAlchemy uses the psycopg (psycopg3) driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _detect_embedding_meta(conn) -> Dict[str, Any]:
    """Detect embedding column existence, nullability, and expected dimensions."""
    meta = {"exists": False, "not_null": False, "dims": None}
    row = conn.execute(
        text(
            """
            SELECT a.attnotnull AS not_null,
                   format_type(a.atttypid, a.atttypmod) AS fmt
            FROM pg_attribute a
            WHERE a.attrelid = 'dbi_table_docs'::regclass
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
    ).fetchone()
    if row:
        meta["exists"] = True
        meta["not_null"] = bool(row.not_null)
        fmt = str(row.fmt or "")
        if "vector(" in fmt:
            try:
                dims = int(fmt.split("vector(", 1)[1].split(")", 1)[0])
                meta["dims"] = dims
            except Exception:
                pass
    return meta


def _coerce_dims(vec: List[float], target_dims: int) -> List[float]:
    """Coerce an embedding vector to target dimensions by truncating/padding."""
    if target_dims <= 0:
        return vec
    n = len(vec)
    if n == target_dims:
        return vec
    if n > target_dims:
        # Simple uniform downsample: take first target_dims elements
        # (quality is not critical; embeddings are low-weight tiebreaker)
        return vec[:target_dims]
    # Pad with zeros
    return vec + [0.0] * (target_dims - n)


def build_table_docs(schemas: Optional[List[str]] = None) -> None:
    """
    Build table documentation and column registry for the specified schemas.
    Creates/updates both dbi_table_docs and dbi_table_columns tables.
    """
    target_schemas = schemas or SETTINGS.db_schemas
    engine = create_engine(_to_sqlalchemy_url(SETTINGS.database_url))
    
    logger.info(f"Building table docs for schemas: {target_schemas}")
    
    # Build column registry first
    build_column_registry(engine, target_schemas)
    
    # Build table documentation with embeddings
    build_table_documentation(engine, target_schemas)
    
    logger.info("Table documentation build completed")


def build_column_registry(engine: Engine, schemas: List[str]) -> None:
    """Build the dbi_table_columns registry with column metadata."""
    logger.info("Building column registry...")
    
    with engine.begin() as conn:
        # Clear existing data for these schemas
        conn.execute(
            text("DELETE FROM dbi_table_columns WHERE schema_name = ANY(:schemas)"),
            {"schemas": schemas}
        )
        
        # Get table information from information_schema
        metadata = MetaData()
        inspect_engine = inspect(engine)
        
        for schema in schemas:
            try:
                tables = inspect_engine.get_table_names(schema=schema)
                logger.info(f"Processing {len(tables)} tables in schema {schema}")
                
                for table_name in tables:
                    columns = inspect_engine.get_columns(table_name, schema=schema)
                    primary_keys = inspect_engine.get_pk_constraint(table_name, schema=schema)
                    foreign_keys = inspect_engine.get_foreign_keys(table_name, schema=schema)
                    
                    pk_columns = set(primary_keys.get('constrained_columns', []))
                    fk_columns = {fk['constrained_columns'][0] for fk in foreign_keys}
                    
                    for column in columns:
                        is_pk = column['name'] in pk_columns
                        is_fk = column['name'] in fk_columns
                        
                        # Insert column info
                        conn.execute(
                            text("""
                                INSERT INTO dbi_table_columns 
                                (schema_name, table_name, column_name, column_type, column_comment, is_pk, is_fk)
                                VALUES (:schema, :table, :column, :type, :comment, :is_pk, :is_fk)
                            """),
                            {
                                "schema": schema,
                                "table": table_name,
                                "column": column['name'],
                                "type": str(column['type']),
                                "comment": column.get('comment'),
                                "is_pk": is_pk,
                                "is_fk": is_fk
                            }
                        )
                        
            except Exception as e:
                logger.warning(f"Error processing schema {schema}: {e}")
                continue
        
        logger.info("Column registry build completed")


def build_table_documentation(engine: Engine, schemas: List[str]) -> None:
    """Build table documentation with embeddings."""
    logger.info("Building table documentation...")
    
    with engine.begin() as conn:
        # Clear existing docs for these schemas
        conn.execute(
            text("DELETE FROM dbi_table_docs WHERE schema_name = ANY(:schemas)"),
            {"schemas": schemas}
        )
        
        # Detect embedding column meta
        emb_meta = _detect_embedding_meta(conn)
        logger.info(f"Embedding column: exists={emb_meta['exists']}, not_null={emb_meta['not_null']}, dims={emb_meta['dims']}")
        
        for schema in schemas:
            try:
                # Get tables and their columns
                result = conn.execute(
                    text("""
                        SELECT table_name, 
                               string_agg(column_name || ' ' || column_type || 
                                         CASE WHEN is_pk THEN ' PK' ELSE '' END ||
                                         CASE WHEN is_fk THEN ' FK' ELSE '' END, 
                                         ', ' ORDER BY column_name) as columns_summary
                        FROM dbi_table_columns 
                        WHERE schema_name = :schema
                        GROUP BY table_name
                    """),
                    {"schema": schema}
                )
                
                for table_name, columns_summary in result:
                    # Build table documentation
                    doc = build_table_doc(schema, table_name, columns_summary)
                    
                    # Generate embedding
                    embedding: Optional[List[float]] = None
                    try:
                        raw_vec = embed(doc)
                        if emb_meta["dims"]:
                            embedding = _coerce_dims(raw_vec, emb_meta["dims"])
                        else:
                            embedding = raw_vec
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for {schema}.{table_name}: {e}")
                        embedding = None
                    
                    # Insert/update table doc using savepoints to avoid aborting the transaction
                    if embedding is not None:
                        try:
                            with conn.begin_nested():
                                conn.execute(
                                    text("""
                                        INSERT INTO dbi_table_docs (schema_name, table_name, doc, embedding)
                                        VALUES (:schema, :table, :doc, :embedding)
                                    """),
                                    {
                                        "schema": schema,
                                        "table": table_name,
                                        "doc": doc,
                                        "embedding": embedding
                                    }
                                )
                        except Exception as insert_err:
                            logger.warning(f"Embedding insert failed for {schema}.{table_name}, inserting without embedding: {insert_err}")
                            with conn.begin_nested():
                                if emb_meta["exists"] and emb_meta["not_null"] and emb_meta["dims"]:
                                    zero_vec = [0.0] * emb_meta["dims"]
                                    conn.execute(
                                        text("""
                                            INSERT INTO dbi_table_docs (schema_name, table_name, doc, embedding)
                                            VALUES (:schema, :table, :doc, :embedding)
                                        """),
                                        {
                                            "schema": schema,
                                            "table": table_name,
                                            "doc": doc,
                                            "embedding": zero_vec
                                        }
                                    )
                                else:
                                    conn.execute(
                                        text("""
                                            INSERT INTO dbi_table_docs (schema_name, table_name, doc)
                                            VALUES (:schema, :table, :doc)
                                        """),
                                        {
                                            "schema": schema,
                                            "table": table_name,
                                            "doc": doc
                                        }
                                    )
                    else:
                        with conn.begin_nested():
                            if emb_meta["exists"] and emb_meta["not_null"] and emb_meta["dims"]:
                                zero_vec = [0.0] * emb_meta["dims"]
                                conn.execute(
                                    text("""
                                        INSERT INTO dbi_table_docs (schema_name, table_name, doc, embedding)
                                        VALUES (:schema, :table, :doc, :embedding)
                                    """),
                                    {
                                        "schema": schema,
                                        "table": table_name,
                                        "doc": doc,
                                        "embedding": zero_vec
                                    }
                                )
                            else:
                                conn.execute(
                                    text("""
                                        INSERT INTO dbi_table_docs (schema_name, table_name, doc)
                                        VALUES (:schema, :table, :doc)
                                    """),
                                    {
                                        "schema": schema,
                                        "table": table_name,
                                        "doc": doc
                                    }
                                )
                        
            except Exception as e:
                logger.warning(f"Error processing schema {schema}: {e}")
                continue
        
        logger.info("Table documentation build completed")


def build_table_doc(schema: str, table_name: str, columns_summary: str) -> str:
    """Build a structured table documentation string."""
    doc_lines = [
        f"Table: {schema}.{table_name}",
        f"Columns: {columns_summary}",
        f"Purpose: Stores {table_name.replace('_', ' ')} data",
        f"Neighbors: OUT->; IN->"  # Placeholder for FK relationships
    ]
    
    return "\n".join(doc_lines)


def get_table_doc(schema: str, table_name: str) -> Optional[str]:
    """Retrieve table documentation for a specific table."""
    engine = create_engine(_to_sqlalchemy_url(SETTINGS.database_url))
    
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT doc FROM dbi_table_docs WHERE schema_name = :schema AND table_name = :table"),
            {"schema": schema, "table": table_name}
        )
        row = result.fetchone()
        return row[0] if row else None


def list_table_docs(schemas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """List all table documentation entries."""
    target_schemas = schemas or SETTINGS.db_schemas
    engine = create_engine(_to_sqlalchemy_url(SETTINGS.database_url))
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT schema_name, table_name, doc, 
                       CASE WHEN embedding IS NOT NULL THEN true ELSE false END as has_embedding,
                       created_at
                FROM dbi_table_docs 
                WHERE schema_name = ANY(:schemas)
                ORDER BY schema_name, table_name
            """),
            {"schemas": target_schemas}
        )
        
        return [
            {
                "schema": row[0],
                "table": row[1],
                "doc": row[2],
                "has_embedding": row[3],
                "created_at": row[4]
            }
            for row in result
        ]


if __name__ == "__main__":
    # CLI interface for building docs
    import argparse
    
    parser = argparse.ArgumentParser(description="Build table documentation and column registry")
    parser.add_argument("--schemas", type=str, help="Comma-separated list of schemas")
    parser.add_argument("--list", action="store_true", help="List existing table docs")
    
    args = parser.parse_args()
    
    if args.list:
        docs = list_table_docs()
        for doc in docs:
            print(f"{doc['schema']}.{doc['table']}: {'✓' if doc['has_embedding'] else '✗'}")
    else:
        schemas = None
        if args.schemas:
            schemas = [s.strip() for s in args.schemas.split(",")]
        
        build_table_docs(schemas) 