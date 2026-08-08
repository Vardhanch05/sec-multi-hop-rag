"""
ingestion/edgar_client.py
-------------------------
Handles fetching filing metadata and text from SEC EDGAR using edgartools.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from edgar import Company, set_identity

logger = logging.getLogger(__name__)

# Set identity for edgartools (SEC compliance)
set_identity("Vardhan sec-rag-system@example.com")

class DownloadError(Exception):
    """Raised when text fails to fetch."""
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
    period_end_date: date | None = None

def get_quarter_from_period(period_of_report: date, ticker: str) -> str:
    """
    Use the period_of_report date from the filing itself.
    Do NOT infer from filing date — companies file weeks after period ends.
    Store as the calendar quarter of the period end date.
    Add a note in metadata that fiscal quarter mapping is not guaranteed.
    """
    month = period_of_report.month
    if month in (1, 2, 3):
        return "Q1"
    elif month in (4, 5, 6):
        return "Q2"
    elif month in (7, 8, 9):
        return "Q3"
    else:
        return "Q4"

def map_fiscal_to_calendar(ticker: str, fiscal_q: str | None, fiscal_y: int) -> tuple[str | None, int]:
    """Returns (calendar_quarter, calendar_year) based on ticker's fiscal schedule."""
    fy_end_month = {
        "AAPL": 9, "MSFT": 6, "NVDA": 1, "WMT": 1, "HD": 1, "V": 9, "CRM": 1, "DIS": 9
    }
    end_month = fy_end_month.get(ticker.upper(), 12)
    if end_month == 12:
        return fiscal_q, fiscal_y
        
    if fiscal_q is None:
        return None, fiscal_y
        
    q_num = int(fiscal_q[1])
    
    # AAPL's period ends drift into the next month, causing edgartools to assign shifted calendar quarters.
    if ticker.upper() == "AAPL":
        if q_num == 1: period_month = 12
        elif q_num == 2: period_month = 4
        elif q_num == 3: period_month = 7
        else: period_month = 9
    else:
        q_months = {
            4: end_month,
            3: (end_month - 3) if (end_month - 3) > 0 else (end_month + 9),
            2: (end_month - 6) if (end_month - 6) > 0 else (end_month + 6),
            1: (end_month - 9) if (end_month - 9) > 0 else (end_month + 3)
        }
        period_month = q_months[q_num]
    
    cal_y = fiscal_y
    if period_month > end_month:
        cal_y -= 1
        
    if period_month in (1, 2, 3):
        cal_q = "Q1"
    elif period_month in (4, 5, 6):
        cal_q = "Q2"
    elif period_month in (7, 8, 9):
        cal_q = "Q3"
    else:
        cal_q = "Q4"
        
    return cal_q, cal_y

def map_calendar_to_fiscal(ticker: str, cal_q: str | None, cal_y: int) -> tuple[str | None, int]:
    """Returns (fiscal_quarter, fiscal_year) by inverting map_fiscal_to_calendar."""
    fy_end_month = {
        "AAPL": 9, "MSFT": 6, "NVDA": 1, "WMT": 1, "HD": 1, "V": 9, "CRM": 1, "DIS": 9
    }
    if fy_end_month.get(ticker.upper(), 12) == 12:
        return cal_q, cal_y

    if cal_q is None:
        return None, cal_y
        
    if ticker.upper() == "AAPL":
        if cal_q == "Q4": return "Q1", cal_y + 1
        if cal_q == "Q2": return "Q2", cal_y
        if cal_q == "Q3": return "Q3", cal_y
        return cal_q, cal_y
        
    # Brute force search for other tickers since the space is small (4 quarters * 3 years)
    for fq in [1, 2, 3, 4]:
        for shift in [-1, 0, 1]:
            test_fy = cal_y + shift
            mapped_cq, mapped_cy = map_fiscal_to_calendar(ticker, f"Q{fq}", test_fy)
            if mapped_cq == cal_q and mapped_cy == cal_y:
                return f"Q{fq}", test_fy
                
    return cal_q, cal_y

def get_new_filings(ticker: str, since_date: date) -> list[FilingRef]:
    """
    Returns list of new filing references since last ingestion run using edgartools.
    """
    all_filings = _fetch_all_filings_from_sec(ticker)
    
    # Filter by date
    return [f for f in all_filings if f.filing_date > since_date]

def _fetch_all_filings_from_sec(ticker: str) -> list[FilingRef]:
    """
    Fetches all 10-K and 10-Q filings for the given ticker using edgartools.
    """
    filing_refs = []
    try:
        company = Company(ticker)
        filings = company.get_filings(form=["10-K", "10-Q"])
        
        if not filings:
            return []
            
        for f in filings:
            # Safely parse dates
            try:
                if isinstance(f.filing_date, str):
                    filing_dt = datetime.strptime(f.filing_date, "%Y-%m-%d").date()
                else:
                    filing_dt = f.filing_date
            except Exception:
                continue
                
            report_dt = None
            if hasattr(f, "report_date") and f.report_date:
                try:
                    if isinstance(f.report_date, str):
                        report_dt = datetime.strptime(f.report_date, "%Y-%m-%d").date()
                    else:
                        report_dt = f.report_date
                except Exception:
                    pass
            
            cal_year = report_dt.year if report_dt else filing_dt.year
            
            cal_quarter = None
            if f.form == "10-Q" and report_dt:
                cal_quarter = get_quarter_from_period(report_dt, ticker)
            
            # Map calendar to fiscal for storage
            fiscal_quarter, fiscal_year = map_calendar_to_fiscal(ticker, cal_quarter, cal_year)
            
            # A 10-Q can never be a fiscal Q4. If period drift causes it to be Q4, it's actually Q3.
            if f.form == "10-Q" and fiscal_quarter == "Q4":
                fiscal_quarter = "Q3"
                    
            filing_refs.append(
                FilingRef(
                    ticker=ticker,
                    filing_type=f.form.replace("/A", ""),
                    accession_number=f.accession_no,
                    filing_date=filing_dt,
                    source_url=f.homepage_url if hasattr(f, 'homepage_url') else getattr(f, 'url', ''),
                    quarter=fiscal_quarter,
                    fiscal_year=fiscal_year,
                    period_end_date=report_dt
                )
            )
    except Exception as e:
        logger.error(f"Error fetching filings for {ticker}: {e}")
        
    return filing_refs

def fetch_filing_text(filing_ref: FilingRef) -> str | None:
    """
    Fetches the clean text for a filing directly in memory using edgartools.
    """
    from db.queries import filing_exists

    if filing_exists(filing_ref.accession_number):
        return None

    try:
        c = Company(filing_ref.ticker)
        filings = c.get_filings(accession_number=filing_ref.accession_number)
        if filings and len(filings) > 0:
            return filings[0].text()
        else:
            raise DownloadError(f"Filing {filing_ref.accession_number} not found via edgartools.")
    except Exception as e:
        logger.error(f"Failed to extract text for {filing_ref.accession_number}: {e}")
        raise DownloadError(f"Failed to fetch text: {e}")
        
    return None
