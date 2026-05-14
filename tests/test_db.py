"""
tests/test_db.py
----------------
Unit tests for the database layer.
"""

import os
import sqlite3
import pytest

from db.connection import get_connection
import config.settings as settings

def test_sqlite_backend_in_dev():
    """Asserts that get_connection() utilizes SQLite when DB_BACKEND='sqlite'."""
    original_backend = settings.DB_BACKEND
    settings.DB_BACKEND = "sqlite"
    
    conn = get_connection()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
    
    settings.DB_BACKEND = original_backend

def test_schema_tables_exist(tmp_path):
    """
    Connects to an SQLite DB, runs schema.sql, and asserts all 5 tables are created successfully.
    """
    original_url = settings.DATABASE_URL
    test_db_path = tmp_path / "test_sec_rag.db"
    settings.DATABASE_URL = f"sqlite:///{test_db_path}"
    
    try:
        conn = get_connection()
        
        # Read the schema
        schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_script = f.read()
            
        # Execute script
        conn.executescript(schema_script)
        conn.commit()
        
        # Verify the 5 tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        
        assert "filings" in tables
        assert "ingestion_logs" in tables
        assert "ragas_results" in tables
        assert "contradiction_events" in tables
        assert "benchmark_questions" in tables
        
        # Verify columns in filings for sanity check
        cursor = conn.execute("PRAGMA table_info(filings);")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "ticker" in columns
        assert "accession_number" in columns
        
        conn.close()
        
    finally:
        # Restore original URL
        settings.DATABASE_URL = original_url
