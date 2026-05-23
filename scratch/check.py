import os
import sys

# Ensure we can import our modules
sys.path.insert(0, os.path.abspath("."))

from db.connection import get_connection
import config.settings as settings

def check():
    conn = get_connection()
    c = conn.execute("SELECT COUNT(*) FROM filings")
    print("Filings:", c.fetchone()[0])
    
    from ingestion.vector_store import get_vector_store
    store = get_vector_store()
    print("Chunks in chroma:", store.collection.count())
    
check()
