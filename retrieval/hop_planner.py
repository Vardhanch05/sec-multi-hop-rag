"""
retrieval/hop_planner.py
------------------------
Resolves temporal references into explicit retrieval hops.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
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
    
    # Periods are already stored natively as fiscal
    fiscal_periods = []
    for p in periods:
        fiscal_periods.append((p.quarter, p.fiscal_year, p.filing_date))
    
    # Sort ascending for clean chronological list
    def sort_key(p):
        fq, fy, d = p
        q_val = 0
        if fq:
            try:
                q_val = int(fq[1])
            except (ValueError, IndexError):
                q_val = 0
        return (fy, q_val, d or date.min)
        
    sorted_asc = sorted(fiscal_periods, key=sort_key)
    
    # Deduplicate to avoid repeating 10-K and 10-Q for the same mapped quarter if they collide
    seen = set()
    formatted = []
    for fq, fy, _ in sorted_asc:
        # For display, if it's the 10-K, fq is None
        label = f"{fq} {fy}" if fq else f"10-K {fy}"
        if label not in seen:
            seen.add(label)
            formatted.append(label)
            
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
        
    # If the LLM extracted tickers but no periods, default to the most recent annual filing
    periods_to_plan = hop_plan.periods
    if not periods_to_plan and hop_plan.tickers:
        periods_to_plan = [PeriodSpec(ticker=t, quarter=None, fiscal_year=None) for t in hop_plan.tickers]
        
    for period_spec in periods_to_plan:
        ticker = period_spec.ticker.upper()
        ticker_avail = ticker_periods.get(ticker, [])
        
        q_str = period_spec.quarter
        fy = period_spec.fiscal_year
        
        if fy is None:
            if not q_str or "10-k" in str(q_str).lower() or "annual" in str(q_str).lower():
                q_str = "last year"
            else:
                q_str = "last quarter"
        
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
                match = re.match(r'^(Q[1-4])\s+(\d{4})$', q_str.strip().upper())
                raw_q = match.group(1)
                raw_y = int(match.group(2))
                
                if raw_q == "Q4":
                    logger.info(f"Routed Fiscal Q4 temporal reference to the annual 10-K filing for fiscal year {raw_y} by design.")
                    try:
                        resolved = resolve_temporal_reference(ticker, f"{raw_y}", ticker_avail)
                    except HopResolutionError:
                        raise HopResolutionError(
                            f"No filings found for {ticker} {raw_q} {raw_y}. "
                            f"Available periods: {format_available_periods(ticker, ticker_avail)}."
                        )
                else:
                    mapped_q_str = f"{raw_q} {raw_y}"
                    try:
                        resolved = resolve_temporal_reference(ticker, mapped_q_str, ticker_avail)
                    except HopResolutionError:
                        raise HopResolutionError(
                            f"No filings found for {ticker} {raw_q} {raw_y}. "
                            f"Available periods: {format_available_periods(ticker, ticker_avail)}."
                        )
            else:
                # Literal match of quarter and fiscal_year
                orig_q = q_str.strip().upper() if q_str else None
                orig_fy = fy
                norm_q = orig_q
                    
                matching = []
                for p in ticker_avail:
                    if p.fiscal_year == fy:
                        if norm_q in ("10-K", "FY", "ANNUAL", None):
                            if p.quarter is None or p.filing_type == "10-K":
                                matching.append(p)
                        elif p.quarter == norm_q:
                            matching.append(p)
                
                if orig_q == "Q4" and fy is not None:
                    matching_10k = [p for p in ticker_avail if p.fiscal_year == fy and (p.quarter is None or p.filing_type == "10-K")]
                    if matching_10k:
                        logger.info(f"Routed Fiscal Q4 temporal reference to the annual 10-K filing for fiscal year {fy} by design.")
                        matching = matching_10k
                    else:
                        matching = []
                
                if not matching:
                    missing_p_str = f"{orig_q} {orig_fy}" if orig_q else f"10-K {orig_fy}"
                    logger.warning(f"Filing genuinely missing for ticker {ticker} in period {missing_p_str}.")
                    raise HopResolutionError(
                        f"No filings found for {ticker} {missing_p_str}. "
                        f"Available periods: {format_available_periods(ticker, ticker_avail)}."
                    )
                resolved = matching
                
        # For each resolved FilingPeriod, build a HopSpec
        for rp in resolved:
            sec_hint = hop_plan.section_hint
            if sec_hint:
                sec_hint_lower = sec_hint.lower()
                if "md&a" in sec_hint_lower or ("management" in sec_hint_lower and "discussion" in sec_hint_lower):
                    sec_hint = "MD&A"
                elif "risk" in sec_hint_lower:
                    sec_hint = "Risk Factors"
                elif "forward" in sec_hint_lower or "guidance" in sec_hint_lower:
                    sec_hint = "Forward Guidance"
                elif "financial" in sec_hint_lower:
                    sec_hint = "Financial Statements"
                else:
                    sec_hint = None
                    
            hop_specs.append(
                HopSpec(
                    ticker=rp.ticker,
                    quarter=rp.quarter,
                    fiscal_year=rp.fiscal_year,
                    filing_type=rp.filing_type,
                    section_type=sec_hint
                )
            )
            
    return hop_specs
