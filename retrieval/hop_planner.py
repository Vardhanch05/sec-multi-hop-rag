"""
retrieval/hop_planner.py
------------------------
Resolves temporal references into explicit retrieval hops.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional
from retrieval.query_classifier import HopPlan, PeriodSpec

class HopResolutionError(Exception):
    """Raised when the Hop Planner cannot resolve a temporal reference to a known period in the corpus."""
    pass

@dataclass
class FilingPeriod:
    ticker: str
    quarter: str | None  # None for 10-K
    fiscal_year: int
    filing_type: str     # "10-Q" | "10-K"
    filing_date: date

@dataclass(frozen=True)
class HopSpec:
    ticker: str
    quarter: str | None
    fiscal_year: int
    filing_type: str
    section_type: str | None  # None = no section filter

def get_available_periods(ticker: str) -> list[FilingPeriod]:
    """Queries db/queries.py to fetch all (quarter, fiscal_year, filing_type) tuples
    present in the filings table for the given ticker. Called by plan_hops internally."""
    from db.queries import get_filing_periods_for_ticker
    return get_filing_periods_for_ticker(ticker)

def format_available_periods(ticker: str, periods: list[FilingPeriod]) -> str:
    """Formats available filing periods into a sorted, comma-separated string."""
    if not periods:
        return "None"
    
    # Sort ascending for clean chronological list
    def sort_key(p: FilingPeriod):
        d = p.filing_date or date.min
        q_val = 0
        if p.quarter:
            try:
                q_val = int(p.quarter[1])
            except (ValueError, IndexError):
                q_val = 0
        return (p.fiscal_year, q_val, d)
        
    sorted_asc = sorted(periods, key=sort_key)
    formatted = []
    for p in sorted_asc:
        if p.quarter:
            formatted.append(f"{p.quarter} {p.fiscal_year}")
        else:
            formatted.append(f"10-K {p.fiscal_year}")
    return ", ".join(formatted)

def _sort_periods_descending(periods: list[FilingPeriod]) -> list[FilingPeriod]:
    """Sorts periods chronologically in descending order (latest first)."""
    def sort_key(p: FilingPeriod):
        d = p.filing_date or date.min
        q_val = 0
        if p.quarter:
            try:
                q_val = int(p.quarter[1])
            except (ValueError, IndexError):
                q_val = 0
        return (d, p.fiscal_year, q_val)
    return sorted(periods, key=sort_key, reverse=True)

def resolve_temporal_reference(
    ticker: str,
    ref: str,
    available_periods: list[FilingPeriod]
) -> list[FilingPeriod]:
    """
    Resolves a temporal reference string into one or more FilingPeriod objects.
    Recognized patterns:
      - "last quarter": most recent 10-Q filing period
      - "last 4 quarters": 4 most recent 10-Q filing periods
      - "last year": most recent 10-K filing period
      - "QX YYYY" (e.g. "Q3 2023"): exact 10-Q filing period
      - "YYYY" (e.g. "2023"): exact 10-K filing period
    """
    if not available_periods:
        raise HopResolutionError(
            f"No filings found for {ticker} {ref}. Available periods: None."
        )

    ref_clean = ref.strip().lower()

    if ref_clean in ("last quarter", "last_quarter"):
        quarters = [p for p in available_periods if p.quarter is not None or p.filing_type == "10-Q"]
        if not quarters:
            raise HopResolutionError(
                f"No filings found for {ticker} {ref}. "
                f"Available periods: {format_available_periods(ticker, available_periods)}."
            )
        sorted_q = _sort_periods_descending(quarters)
        return [sorted_q[0]]

    elif ref_clean in ("last 4 quarters", "last_4_quarters"):
        quarters = [p for p in available_periods if p.quarter is not None or p.filing_type == "10-Q"]
        if not quarters:
            raise HopResolutionError(
                f"No filings found for {ticker} {ref}. "
                f"Available periods: {format_available_periods(ticker, available_periods)}."
            )
        sorted_q = _sort_periods_descending(quarters)
        return sorted_q[:4]

    elif ref_clean in ("last year", "last_year"):
        years = [p for p in available_periods if p.quarter is None or p.filing_type == "10-K"]
        if not years:
            raise HopResolutionError(
                f"No filings found for {ticker} {ref}. "
                f"Available periods: {format_available_periods(ticker, available_periods)}."
            )
        sorted_y = _sort_periods_descending(years)
        return [sorted_y[0]]

    # Parse specific references like "Q3 2023" or "2023"
    ref_upper = ref.strip().upper()
    match_q = re.match(r'^(Q[1-4])\s+(\d{4})$', ref_upper)
    if match_q:
        q, y = match_q.groups()
        fiscal_year = int(y)
        matching = [p for p in available_periods if p.quarter == q and p.fiscal_year == fiscal_year]
        if not matching:
            raise HopResolutionError(
                f"No filings found for {ticker} {q} {fiscal_year}. "
                f"Available periods: {format_available_periods(ticker, available_periods)}."
            )
        return matching

    match_y = re.match(r'^(\d{4})$', ref_upper)
    if match_y:
        y = int(match_y.group(1))
        matching = [p for p in available_periods if p.fiscal_year == y and (p.quarter is None or p.filing_type == "10-K")]
        if not matching:
            # Fallback to any filing in that year
            matching = [p for p in available_periods if p.fiscal_year == y]
        if not matching:
            raise HopResolutionError(
                f"No filings found for {ticker} 10-K {y}. "
                f"Available periods: {format_available_periods(ticker, available_periods)}."
            )
        return matching

    # If nothing matched, raise error
    raise HopResolutionError(
        f"No filings found for {ticker} {ref}. "
        f"Available periods: {format_available_periods(ticker, available_periods)}."
    )

def plan_hops(hop_plan: HopPlan, available_periods: list[FilingPeriod]) -> list[HopSpec]:
    """
    Resolves temporal references to concrete hop specs.
    Raises HopResolutionError if a period is not in corpus.
    """
    hop_specs = []
    
    # Filter available periods for the ticker(s)
    ticker_periods = {}
    for p in available_periods:
        ticker_periods.setdefault(p.ticker.upper(), []).append(p)
        
    for period_spec in hop_plan.periods:
        ticker = period_spec.ticker.upper()
        ticker_avail = ticker_periods.get(ticker, [])
        
        q_str = period_spec.quarter
        fy = period_spec.fiscal_year
        
        if not ticker_avail:
            missing_p_str = f"{q_str} {fy}" if q_str else f"10-K {fy}"
            raise HopResolutionError(
                f"No filings found for {ticker} {missing_p_str}. "
                f"Available periods: None."
            )
        
        # Check if quarter is a relative temporal keyword
        if q_str and q_str.strip().lower() in ("last quarter", "last_quarter"):
            resolved = resolve_temporal_reference(ticker, "last quarter", ticker_avail)
        elif q_str and q_str.strip().lower() in ("last 4 quarters", "last_4_quarters"):
            resolved = resolve_temporal_reference(ticker, "last 4 quarters", ticker_avail)
        elif q_str and q_str.strip().lower() in ("last year", "last_year"):
            resolved = resolve_temporal_reference(ticker, "last year", ticker_avail)
        else:
            # It's a literal reference or QX YYYY or YYYY string
            if q_str and re.match(r'^(Q[1-4])\s+(\d{4})$', q_str.strip().upper()):
                resolved = resolve_temporal_reference(ticker, q_str, ticker_avail)
            else:
                # Literal match of quarter and fiscal_year
                norm_q = q_str.upper() if q_str else None
                matching = []
                for p in ticker_avail:
                    if p.fiscal_year == fy:
                        if norm_q in ("10-K", "FY", "ANNUAL", None):
                            if p.quarter is None or p.filing_type == "10-K":
                                matching.append(p)
                        elif p.quarter == norm_q:
                            matching.append(p)
                
                if not matching:
                    missing_p_str = f"{norm_q} {fy}" if norm_q else f"10-K {fy}"
                    raise HopResolutionError(
                        f"No filings found for {ticker} {missing_p_str}. "
                        f"Available periods: {format_available_periods(ticker, ticker_avail)}."
                    )
                resolved = matching
                
        # For each resolved FilingPeriod, build a HopSpec
        for rp in resolved:
            hop_specs.append(
                HopSpec(
                    ticker=rp.ticker,
                    quarter=rp.quarter,
                    fiscal_year=rp.fiscal_year,
                    filing_type=rp.filing_type,
                    section_type=hop_plan.section_hint
                )
            )
            
    return hop_specs
