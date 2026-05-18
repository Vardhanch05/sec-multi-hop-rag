import logging
import tempfile
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List

from db.queries import filing_exists, insert_filing, write_ingestion_log
from ingestion.edgar_client import get_new_filings, download_filing_pdf, DownloadError
from ingestion.pdf_extractor import is_extractable, extract_text
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
                    
                # 2. Setup temporary directory for safe cleanup
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = Path(temp_dir) / f"{filing_ref.accession_number}.pdf"
                    
                    try:
                        # 3. Download
                        download_result = download_filing_pdf(filing_ref, pdf_path)
                        
                        # Handle case where download_filing_pdf skipped downloading because it exists
                        if download_result is None:
                             continue
                            
                        # 4. Extract and check for unparseable image-only PDFs
                        if not is_extractable(pdf_path):
                            err_msg = f"Unparseable image PDF for {filing_ref.accession_number}, skipping."
                            logger.warning(err_msg)
                            errors.append(err_msg)
                            continue
                            
                        text = extract_text(pdf_path)
                        
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
