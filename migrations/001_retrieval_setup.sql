-- Migration: 001_retrieval_setup.sql
-- Enable pg_trgm extension for trigram similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Columns registry to support keyword/column scoring
CREATE TABLE IF NOT EXISTS dbi_table_columns (
  schema_name text NOT NULL,
  table_name  text NOT NULL,
  column_name text NOT NULL,
  column_type text NOT NULL,
  column_comment text,
  is_pk boolean default false,
  is_fk boolean default false,
  PRIMARY KEY (schema_name, table_name, column_name)
);

-- Ensure dbi_table_docs exists (for backward compatibility)
CREATE TABLE IF NOT EXISTS dbi_table_docs (
  schema_name text NOT NULL,
  table_name  text NOT NULL,
  doc text NOT NULL,
  embedding vector(1536), -- OpenAI embedding dimension
  created_at timestamp with time zone DEFAULT now(),
  PRIMARY KEY (schema_name, table_name)
);

-- Trigram indexes for fast keyword lookups
CREATE INDEX IF NOT EXISTS dbi_tn_gin ON dbi_table_docs USING GIN ((lower(schema_name||'.'||table_name)) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS dbi_cols_name_gin ON dbi_table_columns USING GIN ((lower(column_name)) gin_trgm_ops);

-- Index on schema_name for faster filtering
CREATE INDEX IF NOT EXISTS dbi_schema_idx ON dbi_table_docs(schema_name);
CREATE INDEX IF NOT EXISTS dbi_cols_schema_idx ON dbi_table_columns(schema_name);

-- Index on table_name for FK lookups
CREATE INDEX IF NOT EXISTS dbi_cols_table_idx ON dbi_table_columns(table_name);

-- Comments for documentation
COMMENT ON TABLE dbi_table_columns IS 'Registry of table columns with metadata for deterministic retrieval';
COMMENT ON TABLE dbi_table_docs IS 'Table documentation and embeddings for hybrid retrieval system';
COMMENT ON COLUMN dbi_table_columns.column_type IS 'PostgreSQL data type (e.g., text, integer, timestamp)';
COMMENT ON COLUMN dbi_table_columns.column_comment IS 'Optional column description or business meaning';
COMMENT ON COLUMN dbi_table_columns.is_pk IS 'Whether this column is part of the primary key';
COMMENT ON COLUMN dbi_table_columns.is_fk IS 'Whether this column is a foreign key reference'; 