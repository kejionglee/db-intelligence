# Deterministic-First Table Retrieval System

This document describes the new deterministic-first table retrieval system that replaces the pure RAG approach with a hybrid system using keywords, columns, synonyms, FK relationships, and embeddings as a minor tiebreaker.

## Overview

The new system prioritizes deterministic matching over vector similarity:

1. **Keyword candidates** using pg_trgm on table + column names
2. **Column-aware scoring** with synonyms and type hints
3. **Penalties** for junk tables (e.g., _log, _tmp, _audit)
4. **FK expansion** (1-hop) after reranking
5. **Optional vector similarity** with low weight as tiebreaker

## Configuration

All retrieval parameters are configurable via environment variables:

```bash
# Retrieval limits
RETR_TOP_VECTOR=40          # Vector pre-candidates
RETR_TOP_KEYWORD=40         # Keyword pre-candidates  
RETR_FINAL_TOP_K=6          # Final candidates before FK expansion
RETR_MIN_SCORE=0.6          # Minimum confidence threshold

# Scoring weights
W_NAME=1.2                  # Table name similarity weight
W_COL=0.6                   # Column hits weight
W_SYN=0.6                   # Synonym hits weight
W_GRAPH=0.3                 # FK neighbor boost weight
W_VEC=0.2                   # Vector similarity weight (tiebreaker)

# Penalties
P_NO_TOKEN_HIT=2.0          # Penalty for no name/column matches
P_NOISE=0.5                 # Penalty for noise tables

# Debug
DEBUG_RETRIEVAL=false        # Enable detailed scoring logs
```

## Database Setup

### 1. Run the migration

```sql
-- Run migrations/001_retrieval_setup.sql
-- This creates:
-- - pg_trgm extension
-- - dbi_table_columns table
-- - dbi_table_docs table  
-- - Required indexes
```

### 2. Build table documentation

```bash
# Build docs for all schemas in DB_SCHEMAS
python -m src.cli build-docs

# Build docs for specific schemas
python -m src.cli build-docs --schemas public,analytics

# List existing docs
python -m src.cli list-docs
```

## Usage

### Basic usage

```python
from src.retrieval import retrieve_tables

# Get relevant tables for a question
tables = retrieve_tables("active users with role names", ["public"])
# Returns: [("public", "users", 2.1), ("public", "roles", 1.8), ...]
```

### In SQL Agent

The SQL agent automatically uses the new retrieval system:

```python
from src.sql_agent import SqlAgent

agent = SqlAgent()
response = agent.answer_question("Show me active users and their roles")
# Only relevant tables are included in the schema summary
```

## Scoring Breakdown

Each table gets scored on multiple dimensions:

```
Final Score = 
  W_VEC * vector_similarity +
  W_NAME * name_trigram_similarity +
  W_COL * column_hits +
  W_SYN * synonym_hits +
  W_GRAPH * fk_neighbor_boost -
  penalties
```

### Column Hits

- **Direct matches**: Token appears in column name
- **Synonyms**: "customer" matches "user" columns
- **Type hints**: "date" prefers timestamp columns (+0.8)
- **PK/FK bonuses**: +0.2 each

### Penalties

- **No token hit**: -2.0 if no tokens match table/column names
- **Noise tables**: -0.5 for _log/_tmp/_audit unless mentioned

## Debug Mode

Enable detailed logging to see scoring breakdowns:

```bash
DEBUG_RETRIEVAL=true python -m src.cli ask "active users"
```

Example output:
```
Table public.users: vec=0.123, name=0.856, col=1.200, syn=0.600, graph=0.300, penalties=0.000, final=2.679
Table public.roles: vec=0.098, name=0.234, col=0.800, syn=0.600, graph=0.300, penalties=0.000, final=1.832
```

## Testing

Run the test script to verify the system:

```bash
python test_retrieval.py
```

## Migration from Old System

1. **No code changes needed** - the SQL agent automatically uses the new system
2. **Old table docs** are preserved and can still use embeddings
3. **New column registry** provides better column-aware scoring
4. **Fallback behavior** - if retrieval fails, falls back to full schema summary

## Performance

- **pg_trgm indexes** provide fast keyword search
- **Column registry** enables efficient column-based filtering
- **Vector search** is optional and only used if embeddings exist
- **FK expansion** is limited to 1-hop from top candidates

## Troubleshooting

### Common Issues

1. **No tables found**: Check if `build-docs` was run
2. **Low scores**: Verify pg_trgm extension is enabled
3. **Missing embeddings**: Vector search will be skipped automatically
4. **FK errors**: Check table_columns registry for proper PK/FK flags

### Debug Commands

```bash
# Check database setup
python -c "from src.db import get_pool; print('DB connection OK')"

# Verify table docs
python -m src.cli list-docs

# Test specific query
python -m src.cli ask "test query"
``` 