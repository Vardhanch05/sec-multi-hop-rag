import pytest
import os
from datetime import date
from ingestion.section_chunker import Chunk
from ingestion.vector_store import ChromaStore, QdrantBackendStore

# A helper to create chunks
def create_chunk(accession_number, chunk_index, text, section_type, ticker="AAPL", quarter="Q1"):
    return Chunk(
        text=text,
        ticker=ticker,
        filing_type="10-Q",
        quarter=quarter,
        fiscal_year=2024,
        section_type=section_type,
        chunk_index=chunk_index,
        filing_date=date(2024, 1, 1),
        accession_number=accession_number,
        source_url="http://example.com"
    )

@pytest.fixture
def chroma_store(monkeypatch):
    import chromadb
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chromadb")
    
    class MockChromaStore(ChromaStore):
        def __init__(self):
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.create_collection("sec_chunks")
            
    return MockChromaStore()

@pytest.fixture
def qdrant_store(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "") # memory mode
    return QdrantBackendStore()

def run_store_tests(store):
    c1 = create_chunk("0001", 0, "Apple MD&A", "MD&A")
    c2 = create_chunk("0001", 1, "Apple Risk", "Risk Factors")
    c3 = create_chunk("0002", 0, "Microsoft Risk", "Risk Factors", ticker="MSFT", quarter=None)
    
    emb1 = [0.1] * 384
    emb2 = [0.2] * 384
    emb3 = [0.3] * 384
    
    # Test insertion
    store.insert_chunks([c1, c2, c3], [emb1, emb2, emb3])
    
    # Test search without filters
    res = store.search(emb1, filters={}, top_k=3)
    assert len(res) == 3
    
    # Test search with filters
    res = store.search(emb1, filters={"ticker": "AAPL"}, top_k=2)
    assert len(res) == 2
    assert all(r['ticker'] == "AAPL" for r in res)
    
    res = store.search(emb1, filters={"ticker": "MSFT"}, top_k=2)
    assert len(res) == 1
    assert res[0]['ticker'] == "MSFT"
    assert res[0]['quarter'] == "" # None becomes empty string
    
    res = store.search(emb1, filters={"section_type": "Risk Factors"}, top_k=5)
    assert len(res) == 2

def test_chroma_store(chroma_store):
    run_store_tests(chroma_store)

def test_qdrant_store(qdrant_store):
    run_store_tests(qdrant_store)
