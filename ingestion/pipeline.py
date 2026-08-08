import logging
import sys
import tempfile
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List

from db.queries import filing_exists, insert_filing, write_ingestion_log
from ingestion.edgar_client import get_new_filings, fetch_filing_text, DownloadError
from ingestion.filing_extractor import is_extractable
from ingestion.section_chunker import chunk_filing
from ingestion.embedder import embed_chunks
from ingestion.vector_store import get_vector_store

logger = logging.getLogger(__name__)

def run_ingestion(tickers: List[str], since_date: date = None, dry_run: bool = False):
    """
    Orchestrates the SEC EDGAR filing ingestion pipeline.
    """
    if since_date is None:
        # Default to yesterday for daily cron jobs
        since_date = date.today() - timedelta(days=1)
        
    run_timestamp = datetime.now()
    filings_added = 0
    errors = []
    
    for ticker in tickers:
        try:
            new_filings = get_new_filings(ticker, since_date)
            for filing_ref in new_filings:
                # 1. Deduplication check
                if filing_exists(filing_ref.accession_number):
                    logger.info(f"Filing {filing_ref.accession_number} already exists, skipping.")
                    continue
                    
                if dry_run:
                    logger.info(f"[DRY-RUN] Would process filing: {filing_ref.accession_number} for {ticker}")
                    continue
                    
                try:
                    # 3. Download / Fetch Text
                    text = fetch_filing_text(filing_ref)
                    
                    if text is None:
                         continue
                        
                    # 4. Check for unparseable or empty filings
                    if not is_extractable(text):
                        err_msg = f"Unparseable or empty filing for {filing_ref.accession_number}, skipping."
                        logger.warning(err_msg)
                        errors.append(err_msg)
                        continue
                        
                    # 5. Chunk
                    chunks = chunk_filing(text, filing_ref)
                    if not chunks:
                        err_msg = f"No chunks extracted from {filing_ref.accession_number}, skipping."
                        logger.warning(err_msg)
                        errors.append(err_msg)
                        continue
                            
                    # 6. Embed
                    embeddings = embed_chunks([c.text for c in chunks])
                    
                    # 7. Upsert to Vector Store
                    get_vector_store().insert_chunks(chunks, embeddings)
                    
                    # 8. Record in relational DB
                    insert_filing(filing_ref)
                    filings_added += 1
                    logger.info(f"Successfully processed and ingested {filing_ref.accession_number}")
                        
                except DownloadError as e:
                    err_msg = f"DownloadError for {filing_ref.accession_number}: {e}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    continue
                except Exception as e:
                    # Catch vector store failures or other unforeseen errors
                    err_msg = f"Error processing {filing_ref.accession_number}: {e}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    continue
                    
                import time
                time.sleep(0.2)
                        
        except Exception as e:
            err_msg = f"Failed to get new filings for {ticker}: {e}"
            logger.error(err_msg)
            errors.append(err_msg)
            
    # Final step: record run statistics
    if not dry_run:
        write_ingestion_log(
            run_timestamp=run_timestamp,
            tickers_processed=len(tickers),
            filings_added=filings_added,
            errors=errors
        )

if __name__ == "__main__":
    import json
    import argparse
    from config.settings import TICKERS_CONFIG
    
    parser = argparse.ArgumentParser(description="Run SEC ingestion pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Log what would be ingested without writing.")
    parser.add_argument("--since", type=str, help="Start date in YYYY-MM-DD format (e.g. 2023-01-01)")
    parser.add_argument("--tickers", type=str, nargs="+", help="Specific tickers to ingest")
    parser.add_argument("--force-reingest", action="store_true", help="Force re-ingestion by deleting existing data for the tickers")
    args = parser.parse_args()
    
    since_date = datetime.strptime(args.since, "%Y-%m-%d").date() if args.since else None
    
    # Load tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        with open(TICKERS_CONFIG, "r") as f:
            tickers = json.load(f)["tickers"]
            
    if args.force_reingest and not args.dry_run:
        from db.connection import get_connection
        store = get_vector_store()
        for t in tickers:
            print(f"Force re-ingest: Deleting existing data for {t}")
            # Delete from relational DB
            with get_connection() as conn:
                conn.execute("DELETE FROM filings WHERE ticker = ?", (t,))
                conn.commit()
            # Delete from vector store
            try:
                if hasattr(store, 'collection'): # Chroma
                    store.collection.delete(where={"ticker": t})
                elif hasattr(store, 'client'): # Qdrant
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    store.client.delete(
                        collection_name="sec_chunks",
                        points_selector=Filter(
                            must=[FieldCondition(key="ticker", match=MatchValue(value=t))]
                        )
                    )
            except Exception as e:
                print(f"Failed to delete {t} from vector store: {e}")
        
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.info(f"Starting ingestion run for {len(tickers)} tickers. Dry run: {args.dry_run}")
        
        run_ingestion(tickers, since_date=since_date, dry_run=args.dry_run)
        logger.info("Ingestion run complete.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
