"""
retrieval/hop_planner.py
------------------------
Resolves temporal references into explicit retrieval hops.
"""

from dataclasses import dataclass
from datetime import date

@dataclass
class FilingPeriod:
    ticker: str
    quarter: str | None
    fiscal_year: int
    filing_type: str
    filing_date: date

@dataclass
class HopSpec:
    ticker: str
    quarter: str | None
    fiscal_year: int
    filing_type: str
    section_type: str | None  # None = no section filter
