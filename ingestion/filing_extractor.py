"""
ingestion/filing_extractor.py
-----------------------------
Extracts and validates text from SEC filings using edgartools.
"""
import logging

logger = logging.getLogger(__name__)

def is_extractable(text: str) -> bool:
    """
    Checks if the filing text is actually extractable (contains real text).
    Returns False if the text is empty or too short, which indicates 
    a parsing failure or a non-standard filing.
    """
    if not text:
        return False
        
    # An actual 10-K/10-Q will be massive (tens of thousands of characters).
    # If it's less than 500 characters, it's likely broken or just a stub.
    if len(text.strip()) < 500:
        logger.warning(f"Filing text is too short ({len(text)} chars) to be a valid 10-K/10-Q.")
        return False
        
    return True
