"""
tests/test_pdf_extractor.py
---------------------------
Tests the PDF extraction logic.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
from ingestion.pdf_extractor import is_extractable, extract_text

def test_image_only_pdf_rejection():
    """
    Validates that is_extractable returns False for scanned/image-only PDFs.
    Requirement 5.2 & 5.3
    """
    mock_pdf = MagicMock()
    # Mock 3 pages, all returning empty text or None
    page1, page2, page3 = MagicMock(), MagicMock(), MagicMock()
    page1.extract_text.return_value = "   "
    page2.extract_text.return_value = None
    page3.extract_text.return_value = ""
    mock_pdf.pages = [page1, page2, page3]
    
    with patch("ingestion.pdf_extractor.pdfplumber.open") as mock_open:
        # pdfplumber.open returns a context manager
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        result = is_extractable(Path("dummy.pdf"))
        assert result is False

def test_pdf_text_extraction():
    """
    Validates that extract_text perfectly extracts text from standard PDFs.
    Requirement 5.1 & 5.4
    """
    mock_pdf = MagicMock()
    page1, page2 = MagicMock(), MagicMock()
    page1.extract_text.return_value = "Hello World"
    page2.extract_text.return_value = "Page Two Content"
    mock_pdf.pages = [page1, page2]
    
    with patch("ingestion.pdf_extractor.pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        result = extract_text(Path("dummy.pdf"))
        
        assert "Hello World" in result
        assert "Page Two Content" in result
        assert result == "Hello World\n\nPage Two Content"

def test_is_extractable_true():
    """Validates is_extractable returns True when text is present."""
    mock_pdf = MagicMock()
    page1 = MagicMock()
    page1.extract_text.return_value = "This is a valid SEC filing text."
    mock_pdf.pages = [page1]
    
    with patch("ingestion.pdf_extractor.pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        assert is_extractable(Path("dummy.pdf")) is True
