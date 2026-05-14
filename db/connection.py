"""
db/connection.py
----------------
Provides a single connection factory that switches between SQLite (dev)
and PostgreSQL (prod) based on DB_BACKEND.
"""

import os
import sqlite3
from config import settings

def get_connection():
    """
    Returns a database connection. SQLite in dev, PostgreSQL in prod.
    """
    if settings.DB_BACKEND == "postgresql":
        # In production, psycopg2 would be used
        # import psycopg2
        # return psycopg2.connect(settings.DATABASE_URL)
        raise NotImplementedError("PostgreSQL connection not implemented in this local build.")
    else:
        # Default to SQLite
        # Convert 'sqlite:///./sec_rag.db' to './sec_rag.db'
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        
        # Ensure directory exists if it's nested
        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        conn = sqlite3.connect(db_path)
        # Use row_factory to get dict-like rows
        conn.row_factory = sqlite3.Row
        
        # Enable foreign keys (though not strictly required for this schema, it's good practice)
        conn.execute("PRAGMA foreign_keys = ON;")
        
        return conn
