import os
import sys

sys.path.insert(0, os.path.abspath("."))
import pytest

import config.settings as settings
from db.connection import get_connection

def setup_db():
    original_url = settings.DATABASE_URL
    test_db_path = os.path.join(os.path.abspath("."), "scratch", "test_db.sqlite")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    settings.DATABASE_URL = f"sqlite:///{test_db_path}"
    
    conn = get_connection()
    schema_path = os.path.join(os.path.abspath("."), "db", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    return original_url

def setup_vector_store():
    os.environ["VECTOR_STORE_BACKEND"] = "chromadb"
    from ingestion.vector_store import ChromaStore
    import chromadb
    
    class MockChromaStore(ChromaStore):
        def __init__(self):
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection("sec_chunks_debug")
    return MockChromaStore()

url = setup_db()
store = setup_vector_store()
try:
    from tests.test_integration import test_corpus_scale_20_tickers_8_quarters
    import tests.test_integration as ti
    
    # We will patch retrieve_hops to print if it fails
    original_retrieve_hops = ti.retrieve_hops
    def debug_retrieve_hops(query, hop_specs, top_k_per_hop=1):
        res = original_retrieve_hops(query, hop_specs, top_k_per_hop)
        print("QUERY:", query)
        print("HOP SPECS:", hop_specs)
        print("RESULTS:", res)
        # also print vector store counts
        print("STORE COUNT:", store.collection.count())
        # try manual search
        import ingestion.embedder as embedder
        qe = embedder.embed_query(query)
        hs = hop_specs[0]
        q_val = hs.quarter if hs.quarter is not None else ""
        filters = {
            "ticker": hs.ticker,
            "fiscal_year": hs.fiscal_year,
            "filing_type": hs.filing_type,
            "quarter": q_val
        }
        print("MANUAL SEARCH FILTERS:", filters)
        raw = store.search(query_embedding=qe, filters=filters, top_k=5)
        print("MANUAL SEARCH RAW:", raw)
        return res
        
    ti.retrieve_hops = debug_retrieve_hops

    test_corpus_scale_20_tickers_8_quarters(None, store)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    settings.DATABASE_URL = url
