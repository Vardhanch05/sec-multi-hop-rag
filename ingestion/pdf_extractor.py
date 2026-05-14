"""
ingestion/pdf_extractor.py
--------------------------
Extracts text from downloaded SEC PDF filings using pdfplumber.
Handles edge cases like image-only scanned PDFs.
"""

import pdfplumber
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def is_extractable(pdf_path: Path) -> bool:
    """
    Checks if a PDF is extractable (contains real text) vs an image-only scan.
    Returns False if no text is found in the first 3 pages.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(3, len(pdf.pages))
            
            for i in range(pages_to_check):
                page = pdf.pages[i]
                text = page.extract_text()
                # If we find any meaningful text (more than just a stray character)
                if text and len(text.strip()) > 10:
                    return True
                    
            return False
    except Exception as e:
        logger.warning(f"Error checking extractability of {pdf_path}: {e}")
        return False

def extract_text(pdf_path: Path) -> str:
    """
    Extracts text page-by-page from the PDF.
    """
    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
                    
        return "\n\n".join(extracted_text)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise
