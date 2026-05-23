"""
tests/test_integration.py
-------------------------
"""
import pytest
import os
import json
import re
from datetime import date
from unittest.mock import patch, MagicMock

import config.settings as settings
from db.connection import get_connection
from db.queries import insert_filing
from ingestion.edgar_client import FilingRef
from ingestion.section_chunker import Chunk
from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "BAC", "GS",
    "JNJ", "PFE", "UNH", "XOM", "CVX",
    "WMT", "HD", "V", "MA", "NFLX"
]
QUARTERS = ["Q1", "Q2", "Q3", None]
YEARS = [2023, 2024]

@pytest.fixture
def setup_db(tmp_path):
    original_url = settings.DATABASE_URL
    test_db_path = tmp_path / "test_sec_rag_integration.db"
    settings.DATABASE_URL = f"sqlite:///{test_db_path}"
    
    conn = get_connection()
    schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    
    yield
    
    settings.DATABASE_URL = original_url

@pytest.fixture
def setup_vector_store(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chromadb")
    from ingestion.vector_store import ChromaStore
    import chromadb
    
    class MockChromaStore(ChromaStore):
        def __init__(self):
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection("integration_sec_chunks")
            
    return MockChromaStore()

def test_corpus_scale_20_tickers_8_quarters(setup_db, setup_vector_store):
    """
    Property 1.10, 2.5: Full pipeline end-to-end correctness across 160 combinations
    """
    store = setup_vector_store
    
    chunks_to_insert = []
    embeddings = []
    idx = 0
    
    for ticker in TICKERS:
        for year in YEARS:
            for q in QUARTERS:
                idx += 1
                q_str = q if q else "K"
                acc_num = f"0000{idx}-{year}-{q_str}"
                filing_type = "10-Q" if q else "10-K"
                filing_date = date(year, 1, 1)
                
                fref = FilingRef(
                    ticker=ticker, filing_type=filing_type, accession_number=acc_num,
                    filing_date=filing_date, source_url="http://example.com",
                    quarter=q, fiscal_year=year
                )
                insert_filing(fref)
                
                chunk = Chunk(
                    text=f"Synthetic chunk for {ticker} {year} {q_str}",
                    ticker=ticker, filing_type=filing_type, quarter=q, fiscal_year=year,
                    section_type="Item 1A", chunk_index=0, filing_date=filing_date,
                    accession_number=acc_num, source_url="http://example.com"
                )
                chunks_to_insert.append(chunk)
                embeddings.append([0.1 + (idx / 1000.0)] * 384)
                
    store.insert_chunks(chunks_to_insert, embeddings)
    
    def mock_groq_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        system_content = messages[0]["content"] if messages else ""
        
        mock_completion = MagicMock()
        mock_message = MagicMock()
        
        if "classify user queries" in system_content:
            user_query = messages[1]["content"] if len(messages) > 1 else ""
            m = re.search(r"Querying (.+) (\d+) (Q[1-3]|K)", user_query)
            if m:
                t = m.group(1)
                y = int(m.group(2))
                q_str = m.group(3)
                q = q_str if q_str != "K" else None
                data = {
                    "hop_count": 1,
                    "query_type": "single_hop",
                    "tickers": [t],
                    "periods": [{"ticker": t, "quarter": q, "fiscal_year": y}],
                    "section_hint": "Item 1A",
                    "requires_contradiction_check": False
                }
            else:
                data = {}
            mock_message.content = json.dumps(data)
            
        elif "extract exactly one concise" in system_content:
            data = {"claims": ["extracted claim"]}
            mock_message.content = json.dumps(data)
            
        else:
            mock_message.content = "Synthesized Answer."
            
        mock_completion.choices = [MagicMock(message=mock_message)]
        return mock_completion

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_groq_create
    
    with patch("retrieval.query_classifier.Groq", return_value=mock_client), \
         patch("retrieval.claim_extractor.Groq", return_value=mock_client), \
         patch("synthesis.answer_synthesizer.Groq", return_value=mock_client), \
         patch("ingestion.embedder.embed_query", return_value=[0.1] * 384), \
         patch("contradiction.nli_scorer.get_model") as mock_nli, \
         patch("retrieval.retriever.get_vector_store", return_value=store):
             
        mock_nli_model = MagicMock()
        mock_nli_model.predict.return_value = [[0.1, 0.9, 0.0]]  # no contradiction
        mock_nli.return_value = mock_nli_model
        
        success_count = 0
        for ticker in TICKERS:
            for year in YEARS:
                for q in QUARTERS:
                    q_str = q if q else "K"
                    query = f"Querying {ticker} {year} {q_str}"
                    
                    ui_filters = UIFilters(tickers=[ticker])
                    hop_plan = classify_query(query, ui_filters)
                    
                    available_periods = get_available_periods(ticker)
                    hop_specs = plan_hops(hop_plan, available_periods)
                    
                    hop_results = retrieve_hops(query, hop_specs, top_k_per_hop=1)
                    
                    all_chunks = []
                    for chunks in hop_results.values():
                        all_chunks.extend(chunks)
                    
                    assert len(all_chunks) > 0, f"Retrieval failed for {ticker} {year} {q_str}"
                    
                    claims = extract_claims(query, all_chunks)
                    report = score_contradictions(claims)
                    payload = synthesize(query, claims, report)
                    
                    assert payload.answer == "Synthesized Answer."
                    success_count += 1
                    
        assert success_count == 160
