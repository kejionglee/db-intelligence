#!/usr/bin/env python3
"""
Simple test script for the new deterministic retrieval system.
Run this after setting up the database and building docs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.retrieval import retrieve_tables
from src.config import SETTINGS


def test_retrieval():
    """Test the retrieval system with sample queries."""
    
    # Test queries
    test_queries = [
        "active users with role names",
        "orders with totals by month",
        "customer login history",
        "product categories and counts",
        "user audit logs",
        "temporary backup data"
    ]
    
    print("Testing deterministic retrieval system...")
    print(f"Database: {SETTINGS.database_url}")
    print(f"Schemas: {SETTINGS.db_schemas}")
    print(f"Debug mode: {SETTINGS.debug_retrieval}")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            results = retrieve_tables(query, SETTINGS.db_schemas)
            
            if results:
                print(f"  Found {len(results)} relevant tables:")
                for i, (schema, table, score) in enumerate(results[:5]):  # Top 5
                    print(f"    {i+1}. {schema}.{table} (score: {score:.3f})")
                if len(results) > 5:
                    print(f"    ... and {len(results) - 5} more")
            else:
                print("  No relevant tables found")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        print()


if __name__ == "__main__":
    test_retrieval() 