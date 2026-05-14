"""
tests/test_edgar_client.py
--------------------------
Tests for the SEC EDGAR client.
"""

import pytest
from unittest.mock import patch
from datetime import date, timedelta
from pathlib import Path
from hypothesis import given, settings
import hypothesis.strategies as st

from ingestion.edgar_client import get_new_filings, FilingRef, download_filing_pdf, DownloadError

# Feature: sec-rag-system, Property 4: Incremental ingestion only fetches new filings
@given(since_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2025, 1, 1)))
@settings(max_examples=100)
def test_incremental_fetch_date_filter(since_date):
    """
    Validates: Requirements 1.2
    Ensures that for any since_date, all returned FilingRef.filing_date values 
    are strictly after since_date.
    """
    # Create some dummy data crossing the boundary
    dummy_data = [
        FilingRef("AAPL", "10-Q", "001", since_date - timedelta(days=5), "url", "Q1", 2024),
        FilingRef("AAPL", "10-Q", "002", since_date, "url", "Q2", 2024),
        FilingRef("AAPL", "10-Q", "003", since_date + timedelta(days=5), "url", "Q3", 2024),
    ]
    
    with patch("ingestion.edgar_client._fetch_all_filings_from_sec", return_value=dummy_data):
        new_filings = get_new_filings("AAPL", since_date)
        
        # We expect only the third filing to be returned
        for f in new_filings:
            assert f.filing_date > since_date

def test_download_failure_logs_and_continues(tmp_path):
    """
    Validates: Requirements 1.6
    Ensures a DownloadError is raised after retries are exhausted.
    """
    filing = FilingRef("AAPL", "10-Q", "001", date(2024, 1, 1), "http://invalid-url.com/fail.pdf", "Q1", 2024)
    dest = tmp_path / "test.pdf"
    
    # We don't want the deduplication check to skip the download in the test
    with patch("db.queries.filing_exists", return_value=False):
        # We mock requests.get to always raise an exception
        import requests
        with patch("ingestion.edgar_client.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            # Use retries=1 so the test is fast
            with pytest.raises(DownloadError):
                # Also mock sleep to speed up the test
                with patch("time.sleep"):
                    download_filing_pdf(filing, dest, retries=1)
            
            # Ensure it tried (retries + 1) times
            assert mock_get.call_count == 2
