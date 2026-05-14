"""
contradiction/contradiction_report.py
-------------------------------------
Data contracts for contradiction detection events and reports.
"""

from dataclasses import dataclass

@dataclass
class ContradictionEvent:
    ticker: str
    filing_ref_a: str         # accession_number
    filing_ref_b: str
    claim_a: str
    claim_b: str
    confidence_score: float
    query_id: str | None      # None for batch eval runs

@dataclass
class ContradictionReport:
    contradictions: list['ContradictionEvent']
    timed_out: bool
