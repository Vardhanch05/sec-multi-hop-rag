"""
ingestion/edgar_client.py
-------------------------
Handles fetching filing metadata and PDFs from SEC EDGAR.
"""

import time
import requests
from dataclasses import dataclass
from datetime import date
from pathlib import Path

class DownloadError(Exception):
    """Raised when a PDF fails to download after retries."""
    pass

@dataclass
class FilingRef:
    ticker: str
    filing_type: str          # "10-Q" | "10-K"
    accession_number: str     # e.g. "0000320193-24-000123"
    filing_date: date
    source_url: str
    quarter: str | None       # "Q1"–"Q4" | None for 10-K
    fiscal_year: int

def get_new_filings(ticker: str, since_date: date) -> list[FilingRef]:
    """
    Returns list of new filing references since last ingestion run.
    Stub using SEC EDGAR Full-Text Search API and RSS feed.
    """
    all_filings = _fetch_all_filings_from_sec(ticker)
    
    # Filter by date
    return [f for f in all_filings if f.filing_date > since_date]

def _fetch_all_filings_from_sec(ticker: str) -> list[FilingRef]:
    """
    Stub for the actual SEC API call.
    In a full implementation, this hits the SEC EDGAR RSS or Full-Text API.
    """
    return []

def download_filing_pdf(filing_ref: FilingRef, dest_path: Path, retries: int = 3) -> Path | None:
    """
    Downloads filing PDF to dest_path. Raises DownloadError after retries exhausted.
    Skips downloading if the filing already exists in the database.
    """
    from db.queries import filing_exists

    # Deduplication check
    if filing_exists(filing_ref.accession_number):
        return None

    headers = {'User-Agent': 'Vardhan sec-rag-system/1.0 (vardhan@example.com)'}
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(filing_ref.source_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return dest_path
            
        except requests.RequestException as e:
            if attempt == retries:
                # Retries exhausted
                raise DownloadError(f"Failed to download {filing_ref.source_url} after {retries} retries: {e}")
            
            # Exponential backoff (1s, 2s, 4s...)
            time.sleep(2 ** attempt)

    return None
