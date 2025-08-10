from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .sql_agent import SqlAgent
from .table_docs import build_table_docs, list_table_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="DB Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question to your Postgres database")
    ask_parser.add_argument("question", type=str, help="Natural language question")
    ask_parser.add_argument(
        "--schemas",
        type=str,
        default="",
        help="Comma-separated schemas to consider (default from env)",
    )
    
    # Build-docs command
    build_parser = subparsers.add_parser("build-docs", help="Build table documentation and column registry")
    build_parser.add_argument(
        "--schemas",
        type=str,
        default="",
        help="Comma-separated list of schemas to process",
    )
    
    # List-docs command
    list_parser = subparsers.add_parser("list-docs", help="List existing table documentation")
    list_parser.add_argument(
        "--schemas",
        type=str,
        default="",
        help="Comma-separated list of schemas to list",
    )
    
    args = parser.parse_args()
    
    if args.command == "ask":
        _handle_ask(args)
    elif args.command == "build-docs":
        _handle_build_docs(args)
    elif args.command == "list-docs":
        _handle_list_docs(args)
    else:
        # Default to ask for backward compatibility
        _handle_ask(args)


def _handle_ask(args) -> None:
    """Handle the ask command."""
    schemas: Optional[List[str]] = None
    if args.schemas.strip():
        schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]

    agent = SqlAgent()
    resp = agent.answer_question(question=args.question, schemas=schemas)

    print("SQL:\n" + resp.sql)
    print("\nColumns:", ", ".join(resp.result.columns))
    print("Rows:")
    for r in resp.result.rows:
        print(json.dumps(list(r)))
    print(f"\nRow count: {len(resp.result.rows)}")
    if resp.answer:
        print("\nAnswer:\n" + resp.answer)


def _handle_build_docs(args) -> None:
    """Handle the build-docs command."""
    schemas = None
    if args.schemas.strip():
        schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    
    print("Building table documentation and column registry...")
    build_table_docs(schemas)
    print("Build completed successfully!")


def _handle_list_docs(args) -> None:
    """Handle the list-docs command."""
    schemas = None
    if args.schemas.strip():
        schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    
    docs = list_table_docs(schemas)
    if not docs:
        print("No table documentation found.")
        return
    
    print(f"Found {len(docs)} table documentation entries:")
    for doc in docs:
        status = "✓" if doc['has_embedding'] else "✗"
        print(f"  {doc['schema']}.{doc['table']}: {status}")


if __name__ == "__main__":
    main() 