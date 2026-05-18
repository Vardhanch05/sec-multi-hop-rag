"""
tests/test_ingestion.py
-----------------------
Property tests for the ingestion pipeline orchestration.
"""

import os
import sqlite3
from datetime import date
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

import config.settings as settings_config
from db.connection import get_connection
from ingestion.edgar_client import FilingRef
from ingestion.pipeline import run_ingestion
from ingestion.section_chunker import Chunk

@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    # Set up in-memory or temp file DB for tests
    original_url = settings_config.DATABASE_URL
    original_db_backend = settings_config.DB_BACKEND
    
    test_db_path = tmp_path / "test_sec_rag_pipeline.db"
    settings_config.DATABASE_URL = f"sqlite:///{test_db_path}"
    settings_config.DB_BACKEND = "sqlite"
    
    schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_script = f.read()
        
    conn = get_connection()
    conn.executescript(schema_script)
    conn.commit()
    conn.close()
    
    yield
    
    settings_config.DATABASE_URL = original_url
    settings_config.DB_BACKEND = original_db_backend

# Strategy for FilingRef
filing_ref_st = st.builds(
    FilingRef,
    ticker=st.just("AAPL"),
    filing_type=st.sampled_from(["10-Q", "10-K"]),
    accession_number=st.uuids().map(str),
    filing_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2025, 1, 1)),
    source_url=st.just("http://example.com/pdf"),
    quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
    fiscal_year=st.integers(min_value=2000, max_value=2025)
)

@settings(max_examples=100)
@given(filing_ref=filing_ref_st)
def test_ingestion_deduplication(filing_ref):
    """
    Property 3: Ingestion deduplication (idempotence)
    Validates: Requirements 1.7
    Running ingestion twice for the same filing must not increase the row count in the filings table.
    """
    with patch("ingestion.pipeline.get_new_filings", return_value=[filing_ref]), \
         patch("ingestion.pipeline.download_filing_pdf", return_value=MagicMock()), \
         patch("ingestion.pipeline.is_extractable", return_value=True), \
         patch("ingestion.pipeline.extract_text", return_value="Dummy text"), \
         patch("ingestion.pipeline.chunk_filing", return_value=[
             Chunk(
                 text="Dummy text", ticker=filing_ref.ticker, filing_type=filing_ref.filing_type,
                 quarter=filing_ref.quarter, fiscal_year=filing_ref.fiscal_year,
                 section_type="Other", chunk_index=0, filing_date=filing_ref.filing_date,
                 accession_number=filing_ref.accession_number, source_url=filing_ref.source_url
             )
         ]), \
         patch("ingestion.pipeline.embed_chunks", return_value=[[0.1]*384]), \
         patch("ingestion.pipeline.get_vector_store") as mock_get_store:
         
        # Ensure we have a clean db for this iteration
        conn = get_connection()
        conn.execute("DELETE FROM filings")
        conn.commit()
        conn.close()

        # Run first time
        run_ingestion(tickers=["AAPL"])
        
        # Run second time (should trigger deduplication check)
        run_ingestion(tickers=["AAPL"])
        
        # Check database row count
        conn = get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM filings WHERE accession_number = ?", (filing_ref.accession_number,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1, "The row must only exist once despite two runs"

@settings(max_examples=100)
@given(filing_refs=st.lists(filing_ref_st, min_size=0, max_size=5))
def test_ingestion_log_completeness(filing_refs):
    """
    Property 5: Ingestion log completeness
    Validates: Requirements 1.9
    Every completed run must produce exactly one new ingestion_logs row with all four required fields populated.
    """
    conn = get_connection()
    conn.execute("DELETE FROM ingestion_logs")
    conn.execute("DELETE FROM filings")
    conn.commit()
    conn.close()

    with patch("ingestion.pipeline.get_new_filings", return_value=filing_refs), \
         patch("ingestion.pipeline.download_filing_pdf", return_value=MagicMock()), \
         patch("ingestion.pipeline.is_extractable", return_value=True), \
         patch("ingestion.pipeline.extract_text", return_value="Dummy text"), \
         patch("ingestion.pipeline.chunk_filing", side_effect=lambda text, f: [
             Chunk(
                 text=text, ticker=f.ticker, filing_type=f.filing_type,
                 quarter=f.quarter, fiscal_year=f.fiscal_year,
                 section_type="Other", chunk_index=0, filing_date=f.filing_date,
                 accession_number=f.accession_number, source_url=f.source_url
             )
         ]), \
         patch("ingestion.pipeline.embed_chunks", return_value=[[0.1]*384]), \
         patch("ingestion.pipeline.get_vector_store") as mock_get_store:
         
        tickers = ["AAPL"]
        run_ingestion(tickers=tickers)
        
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT run_timestamp, tickers_processed, filings_added, errors FROM ingestion_logs")
        rows = cursor.fetchall()
        conn.close()
        
        assert len(rows) == 1, "Exactly one log row must be produced"
        row = rows[0]
        
        # All four fields must be non-null and explicitly verified
        assert row["run_timestamp"] is not None, "run_timestamp must be populated"
        assert row["tickers_processed"] is not None, "tickers_processed must be populated"
        assert row["filings_added"] is not None, "filings_added must be populated"
        assert row["errors"] is not None, "errors must be populated"
        
        assert row["tickers_processed"] == len(tickers)
        assert row["filings_added"] == len(filing_refs)
