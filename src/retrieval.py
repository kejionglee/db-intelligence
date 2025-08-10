"""
Deterministic-first table retrieval system with embeddings as minor tiebreaker.
Implements hybrid retrieval using keywords, columns, synonyms, FK relationships, and vector similarity.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine, text

from .config import SETTINGS
from .db import (
    fetch_columns, get_table_neighbors, trigram_similarity,
    search_tables_by_name, search_tables_by_columns
)
from .llm import embed

logger = logging.getLogger(__name__)

# Noise table pattern for penalties
NOISE_PAT = re.compile(r"(?:_|^)(log|audit|tmp|backup|bak|old|stg|staging)s?$", re.I)

# Token extraction regex
TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Domain synonyms for improved matching
SYNONYMS: Dict[str, List[str]] = {
    "user": ["user", "usr", "account", "customer", "member", "employee"],
    "role": ["role", "group", "permission"],
    "login": ["login", "signin", "last_login", "last_seen", "last_sign_in_at"],
    "active": ["active", "is_active", "enabled", "status"],
    "order": ["order", "txn", "transaction", "sale"],
    "product": ["product", "item", "goods", "merchandise"],
    "category": ["category", "type", "class", "group"],
    "date": ["date", "created", "updated", "timestamp", "time"],
    "count": ["count", "total", "sum", "amount", "quantity"],
    "name": ["name", "title", "label", "description"],
}

# Type hints for column scoring
TYPE_HINTS = {
    "date": ["timestamp", "date", "time"],
    "count": ["integer", "bigint", "numeric", "decimal"],
    "name": ["text", "varchar", "character varying"],
    "login": ["timestamp", "date", "time"],
}


def _to_sqlalchemy_url(url: str) -> str:
    """Ensure SQLAlchemy uses the psycopg (psycopg3) driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


# Cache a flag to permanently disable vector prefilter if it errors once
_VECTOR_PREFILTER_AVAILABLE: Optional[bool] = None

# Last retrieval trace for debug
LAST_RETRIEVAL_TRACE: List[Dict[str, float]] = []


@dataclass
class TableScore:
    """Detailed scoring breakdown for a table."""
    schema: str
    table: str
    vec_sim: float = 0.0
    name_sim: float = 0.0
    col_hits: float = 0.0
    syn_hits: float = 0.0
    graph_boost: float = 0.0
    penalties: float = 0.0
    final_score: float = 0.0
    
    @property
    def fqtn(self) -> str:
        return f"{self.schema}.{self.table}"


def _tokens(s: str) -> List[str]:
    """Extract lowercase tokens from text."""
    return TOKEN_RE.findall(s.lower())


def _is_noise_table(name: str) -> bool:
    """Check if table name matches noise patterns."""
    base = name.split(".")[-1]
    return NOISE_PAT.search(base) is not None


def _synonym_hits(text: str, q_tokens: List[str]) -> float:
    """Count synonym matches in text."""
    text = text.lower()
    hits = 0.0
    
    for base, syns in SYNONYMS.items():
        if base in q_tokens and any(s in text for s in syns):
            hits += 1.0
    
    return hits


def _column_hits(columns: List, q_tokens: List[str]) -> float:
    """Score column matches based on names, types, and synonyms."""
    if not columns:
        return 0.0
    
    hits = 0.0
    
    for col in columns:
        col_name = col.name.lower()
        col_type = col.type.lower()
        
        # Direct token matches
        for token in q_tokens:
            if token in col_name:
                hits += 1.0
                break
        
        # Synonym matches
        hits += _synonym_hits(col_name, q_tokens)
        
        # Type hints
        for hint_type, type_patterns in TYPE_HINTS.items():
            if hint_type in q_tokens:
                if any(pattern in col_type for pattern in type_patterns):
                    if hint_type == "date":
                        hits += 0.8
                    elif hint_type == "count":
                        hits += 0.5
                    elif hint_type == "name":
                        hits += 0.3
                    elif hint_type == "login":
                        hits += 0.8
        
        # PK/FK bonuses
        if col.is_pk:
            hits += 0.2
        if col.is_fk:
            hits += 0.2
    
    return hits


def _compute_table_score(
    table_score: TableScore,
    q_tokens: List[str],
    ask_text: str,
    neighbors_map: Dict[str, List[str]],
    vec_sim_cache: Dict[str, float]
) -> None:
    """Compute detailed scoring for a table."""
    # Get columns for this table
    columns = fetch_columns(table_score.schema, table_score.table)
    
    # Column hits
    table_score.col_hits = _column_hits(columns, q_tokens)
    
    # Synonym hits on table name
    table_score.syn_hits = _synonym_hits(table_score.fqtn, q_tokens)
    
    # Graph boost: FK neighbors with good scores
    graph_boost = 0.0
    for neighbor in neighbors_map.get(table_score.fqtn.lower(), []):
        if neighbor in vec_sim_cache and vec_sim_cache[neighbor] >= 0.6:
            graph_boost += 0.3
    table_score.graph_boost = graph_boost
    
    # Penalties
    penalties = 0.0
    
    # No token hit penalty
    has_name_hit = any(t in table_score.fqtn.lower() for t in q_tokens)
    has_col_hit = table_score.col_hits > 0
    if not (has_name_hit or has_col_hit):
        penalties += SETTINGS.p_no_token_hit
    
    # Noise table penalty
    if _is_noise_table(table_score.fqtn) and not any(t in q_tokens for t in ["log", "audit", "tmp"]):
        penalties += SETTINGS.p_noise
    
    table_score.penalties = penalties
    
    # Final score calculation
    table_score.final_score = (
        SETTINGS.w_vec * table_score.vec_sim +
        SETTINGS.w_name * table_score.name_sim +
        SETTINGS.w_col * table_score.col_hits +
        SETTINGS.w_syn * table_score.syn_hits +
        SETTINGS.w_graph * table_score.graph_boost -
        table_score.penalties
    )


def retrieve_tables(question: str, schemas: List[str]) -> List[Tuple[str, str, float]]:
    """
    Retrieve relevant tables using deterministic-first approach with embeddings as tiebreaker.
    
    Returns:
        List of (schema, table, score) tuples sorted by relevance score.
    """
    global _VECTOR_PREFILTER_AVAILABLE

    logger.info(f"Retrieving tables for question: {question}")
    
    engine = create_engine(_to_sqlalchemy_url(SETTINGS.database_url))
    q_tokens = _tokens(question)
    
    logger.debug(f"Extracted tokens: {q_tokens}")
    
    with engine.begin() as conn:
        # --- Vector prefilter (optional) ---
        vector_candidates = []
        vec_sim_cache: Dict[str, float] = {}

        do_vector = (
            SETTINGS.w_vec > 0 and (_VECTOR_PREFILTER_AVAILABLE is None or _VECTOR_PREFILTER_AVAILABLE)  # not known bad
        )
        if do_vector:
            # Check if we have embeddings available
            has_embeddings = conn.execute(
                text("SELECT COUNT(*) FROM dbi_table_docs WHERE embedding IS NOT NULL AND schema_name = ANY(:schemas)"),
                {"schemas": schemas}
            ).scalar() > 0
        else:
            has_embeddings = False
        
        if do_vector and has_embeddings:
            try:
                q_vec = embed(question)
                # Attempt vector search; if it errors (e.g., operator/adapter missing), disable for remainder
                vec_rows = conn.execute(
                    text(
                        """
                        SELECT schema_name, table_name, doc,
                               1 - (embedding <=> :q) AS sim
                        FROM dbi_table_docs
                        WHERE schema_name = ANY(:schemas) AND embedding IS NOT NULL
                        ORDER BY embedding <=> :q
                        LIMIT :lim
                        """
                    ),
                    {"q": q_vec, "schemas": schemas, "lim": SETTINGS.retr_top_vector}
                ).fetchall()
                
                for row in vec_rows:
                    fq = f"{row.schema_name}.{row.table_name}".lower()
                    vec_sim_cache[fq] = float(row.sim)
                    vector_candidates.append(TableScore(
                        schema=row.schema_name,
                        table=row.table_name,
                        vec_sim=float(row.sim)
                    ))
                
                logger.debug(f"Vector prefilter found {len(vector_candidates)} candidates")
                if _VECTOR_PREFILTER_AVAILABLE is None:
                    _VECTOR_PREFILTER_AVAILABLE = True
            except Exception as e:
                # Disable vector prefilter going forward; log once
                if _VECTOR_PREFILTER_AVAILABLE is not False:
                    logger.warning("Vector prefilter unavailable, disabling for this process: %s", e)
                _VECTOR_PREFILTER_AVAILABLE = False
        
        # --- Keyword prefilter ---
        # Name-based search
        name_candidates = search_tables_by_name(question, schemas, SETTINGS.retr_top_keyword)
        
        # Column-based search
        col_candidates = search_tables_by_columns(question, schemas, SETTINGS.retr_top_keyword)
        
        # Combine keyword candidates
        keyword_candidates = []
        for schema, table, name_sim in name_candidates:
            keyword_candidates.append(TableScore(
                schema=schema,
                table=table,
                name_sim=name_sim
            ))
        
        for schema, table, col_sim in col_candidates:
            # Check if already added
            existing = next((c for c in keyword_candidates if c.schema == schema and c.table == table), None)
            if existing:
                existing.name_sim = max(existing.name_sim, col_sim)
            else:
                keyword_candidates.append(TableScore(
                    schema=schema,
                    table=table,
                    name_sim=col_sim
                ))
        
        logger.debug(f"Keyword prefilter found {len(keyword_candidates)} candidates")
        
        # --- Union and dedupe candidates ---
        all_candidates: Dict[str, TableScore] = {}
        
        # Add vector candidates
        for candidate in vector_candidates:
            key = candidate.fqtn.lower()
            all_candidates[key] = candidate
        
        # Add keyword candidates, merging if exists
        for candidate in keyword_candidates:
            key = candidate.fqtn.lower()
            if key in all_candidates:
                # Merge scores
                existing = all_candidates[key]
                existing.name_sim = max(existing.name_sim, candidate.name_sim)
                if candidate.vec_sim > 0:
                    existing.vec_sim = max(existing.vec_sim, candidate.vec_sim)
            else:
                all_candidates[key] = candidate
        
        logger.debug(f"Combined candidates: {len(all_candidates)}")
        
        # --- Precompute neighbors for graph boost ---
        neighbors_map: Dict[str, List[str]] = {}
        for candidate in all_candidates.values():
            neighbors = get_table_neighbors(candidate.schema, candidate.table)
            neighbors_map[candidate.fqtn.lower()] = neighbors
        
        # --- Score all candidates ---
        scored_candidates = []
        # reset debug trace
        try:
            LAST_RETRIEVAL_TRACE.clear()
        except Exception:
            pass
        for candidate in all_candidates.values():
            # Ensure name similarity is computed
            if candidate.name_sim == 0:
                candidate.name_sim = trigram_similarity(candidate.fqtn, question)
            
            # Compute detailed scoring
            _compute_table_score(candidate, q_tokens, question, neighbors_map, vec_sim_cache)
            scored_candidates.append(candidate)
            
            # Debug logging
            if SETTINGS.debug_retrieval:
                logger.info(
                    f"Table {candidate.fqtn}: "
                    f"vec={candidate.vec_sim:.3f}, "
                    f"name={candidate.name_sim:.3f}, "
                    f"col={candidate.col_hits:.3f}, "
                    f"syn={candidate.syn_hits:.3f}, "
                    f"graph={candidate.graph_boost:.3f}, "
                    f"penalties={candidate.penalties:.3f}, "
                    f"final={candidate.final_score:.3f}"
                )
                # store in trace
                try:
                    LAST_RETRIEVAL_TRACE.append({
                        "table": candidate.fqtn,
                        "score": round(candidate.final_score, 3),
                        "nameSim": round(candidate.name_sim, 3),
                        "colHits": round(candidate.col_hits, 3),
                        "synHits": round(candidate.syn_hits, 3),
                        "vec": round(candidate.vec_sim, 3),
                        "penalties": round(candidate.penalties, 3),
                    })
                except Exception:
                    pass
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
        top_candidates = scored_candidates[:SETTINGS.retr_final_top_k]
        
        # Check minimum score threshold
        if top_candidates and top_candidates[0].final_score < SETTINGS.retr_min_score:
            logger.warning(f"Top score {top_candidates[0].final_score:.3f} below threshold {SETTINGS.retr_min_score}")
        
        # --- FK expansion (1-hop from final shortlist) ---
        selected_tables = {c.fqtn.lower() for c in top_candidates}
        expanded_candidates = list(top_candidates)
        
        for candidate in top_candidates:
            for neighbor in neighbors_map.get(candidate.fqtn.lower(), []):
                if neighbor not in selected_tables and "." in neighbor:
                    ns, nt = neighbor.split(".", 1)
                    
                    # Check if neighbor exists in our candidates
                    neighbor_key = neighbor.lower()
                    if neighbor_key in all_candidates:
                        neighbor_candidate = all_candidates[neighbor_key]
                        # Give small boost as neighbor
                        neighbor_candidate.final_score = 0.2 + neighbor_candidate.vec_sim
                        expanded_candidates.append(neighbor_candidate)
                        selected_tables.add(neighbor_key)
                    else:
                        # Create minimal neighbor candidate
                        neighbor_candidate = TableScore(
                            schema=ns,
                            table=nt,
                            vec_sim=vec_sim_cache.get(neighbor_key, 0.0),
                            final_score=0.2 + vec_sim_cache.get(neighbor_key, 0.0)
                        )
                        expanded_candidates.append(neighbor_candidate)
                        selected_tables.add(neighbor_key)
        
        # Final deduplication and sorting
        final_candidates = {}
        for candidate in expanded_candidates:
            key = candidate.fqtn.lower()
            if key not in final_candidates or candidate.final_score > final_candidates[key].final_score:
                final_candidates[key] = candidate
        
        final_results = sorted(final_candidates.values(), key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"Retrieved {len(final_results)} tables with scores: {[(c.schema, c.table, round(c.final_score, 3)) for c in final_results[:5]]}")
        
        return [(c.schema, c.table, round(c.final_score, 3)) for c in final_results]
