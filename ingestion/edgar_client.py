"""
ingestion/edgar_client.py
-------------------------
Handles fetching filing metadata and PDFs from SEC EDGAR.
"""

from dataclasses import dataclass
from datetime import date

@dataclass
class FilingRef:
    ticker: str
    filing_type: str          # "10-Q" | "10-K"
    accession_number: str     # e.g. "0000320193-24-000123"
    filing_date: date
    source_url: str
    quarter: str | None       # "Q1"–"Q4" | None for 10-K
    fiscal_year: int
