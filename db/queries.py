"""
db/queries.py
-------------
Centralizes all SQL reads and writes. No other module executes raw SQL.
"""

import json
from datetime import datetime
from db.connection import get_connection

# Import dataclasses for type hints and return types
from ingestion.edgar_client import FilingRef
from retrieval.hop_planner import FilingPeriod
from contradiction.contradiction_report import ContradictionEvent
from evaluation.ragas_harness import RagasResult

def get_filing_periods_for_ticker(ticker: str) -> list[FilingPeriod]:
    """Queries the filings table and returns typed FilingPeriod objects."""
    query = """
        SELECT ticker, quarter, fiscal_year, filing_type, filing_date
        FROM filings
        WHERE ticker = ?
        ORDER BY fiscal_year DESC, quarter DESC
    """
    with get_connection() as conn:
        cursor = conn.execute(query, (ticker,))
        rows = cursor.fetchall()
        
    return [
        FilingPeriod(
            ticker=row["ticker"],
            quarter=row["quarter"],
            fiscal_year=row["fiscal_year"],
            filing_type=row["filing_type"],
            filing_date=datetime.strptime(row["filing_date"], "%Y-%m-%d").date() if isinstance(row["filing_date"], str) else row["filing_date"]
        ) for row in rows
    ]

def filing_exists(accession_number: str) -> bool:
    """Returns True if the accession_number is already in the filings table."""
    query = "SELECT 1 FROM filings WHERE accession_number = ?"
    with get_connection() as conn:
        cursor = conn.execute(query, (accession_number,))
        return cursor.fetchone() is not None

def insert_filing(filing_ref: FilingRef) -> None:
    """Inserts a new row into the filings table."""
    query = """
        INSERT INTO filings (ticker, filing_type, quarter, fiscal_year, filing_date, accession_number, source_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(query, (
            filing_ref.ticker,
            filing_ref.filing_type,
            filing_ref.quarter,
            filing_ref.fiscal_year,
            filing_ref.filing_date.isoformat() if hasattr(filing_ref.filing_date, 'isoformat') else filing_ref.filing_date,
            filing_ref.accession_number,
            filing_ref.source_url
        ))
        conn.commit()

def write_ingestion_log(run_timestamp, tickers_processed: int, filings_added: int, errors: list[str]) -> None:
    """Appends a row to ingestion_logs."""
    query = """
        INSERT INTO ingestion_logs (run_timestamp, tickers_processed, filings_added, errors)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(query, (
            run_timestamp.isoformat() if hasattr(run_timestamp, 'isoformat') else run_timestamp,
            tickers_processed,
            filings_added,
            json.dumps(errors)
        ))
        conn.commit()

def insert_contradiction_event(event: ContradictionEvent) -> None:
    """Inserts a row into contradiction_events."""
    query = """
        INSERT INTO contradiction_events (query_id, ticker, filing_ref_a, filing_ref_b, claim_a, claim_b, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(query, (
            event.query_id,
            event.ticker,
            event.filing_ref_a,
            event.filing_ref_b,
            event.claim_a,
            event.claim_b,
            event.confidence_score
        ))
        conn.commit()

def write_ragas_result(result: RagasResult) -> None:
    """Inserts a row into ragas_results."""
    query = """
        INSERT INTO ragas_results (run_timestamp, faithfulness, answer_relevance, context_precision, context_recall, subset_breakdowns)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(query, (
            result.run_timestamp.isoformat() if hasattr(result.run_timestamp, 'isoformat') else result.run_timestamp,
            result.faithfulness,
            result.answer_relevance,
            result.context_precision,
            result.context_recall,
            json.dumps(result.subset_breakdowns)
        ))
        conn.commit()

def get_corpus_stats() -> dict:
    """Returns total filings and unique tickers count for the UI."""
    query = "SELECT count(distinct ticker) as unique_tickers, count(*) as total_filings FROM filings"
    with get_connection() as conn:
        cursor = conn.execute(query)
        row = cursor.fetchone()
        if row:
            return {"unique_tickers": row["unique_tickers"], "total_filings": row["total_filings"]}
        return {"unique_tickers": 0, "total_filings": 0}

def get_all_tickers() -> list[str]:
    """Returns a sorted list of all unique tickers in the database."""
    query = "SELECT DISTINCT ticker FROM filings ORDER BY ticker"
    with get_connection() as conn:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        return [r["ticker"] for r in rows]
