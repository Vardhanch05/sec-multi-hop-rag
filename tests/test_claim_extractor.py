import json
import pytest
from datetime import date
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

from retrieval.claim_extractor import extract_claims, Claim, _get_first_sentence
from retrieval.retriever import ChunkResult

# Feature: sec-rag-system, Property 9: Claim extractor cardinality
@given(
    chunks=st.lists(
        st.builds(
            ChunkResult,
            text=st.text(min_size=10, max_size=500),
            ticker=st.text(min_size=1, max_size=5).map(lambda s: s.upper()),
            filing_type=st.sampled_from(["10-Q", "10-K"]),
            quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
            fiscal_year=st.integers(min_value=2000, max_value=2030),
            section_type=st.sampled_from(["MD&A", "Risk Factors"]),
            chunk_index=st.integers(min_value=0, max_value=100),
            filing_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 1, 1)),
            accession_number=st.text(min_size=5, max_size=20),
            source_url=st.text(min_size=10, max_size=50),
            score=st.floats(min_value=0.0, max_value=1.0)
        ),
        min_size=0,
        max_size=10
    )
)
@settings(max_examples=100)
@patch('retrieval.claim_extractor.Groq')
def test_claim_extractor_cardinality(mock_groq_class, chunks):
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "claims": [f"Factual Claim {i}" for i in range(len(chunks))]
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    
    claims = extract_claims("test query", chunks)
    
    # Assert exact N cardinality
    assert len(claims) == len(chunks)
    
    # Assert exact metadata match
    for claim, chunk in zip(claims, chunks):
        assert claim.ticker == chunk.ticker
        assert claim.quarter == chunk.quarter
        assert claim.fiscal_year == chunk.fiscal_year
        assert claim.section_type == chunk.section_type
        assert claim.chunk_index == chunk.chunk_index
        assert claim.filing_date == chunk.filing_date
        assert claim.accession_number == chunk.accession_number
        assert claim.source_url == chunk.source_url

@patch('retrieval.claim_extractor.Groq')
def test_claim_extractor_fallback(mock_groq_class):
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # 1. Invalid JSON fallback
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Invalid JSON response text"))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    
    chunks = [
        ChunkResult(
            text="AAPL revenue grew. This was driven by iPhone sales.",
            ticker="AAPL",
            filing_type="10-Q",
            quarter="Q1",
            fiscal_year=2024,
            section_type="MD&A",
            chunk_index=0,
            filing_date=date(2024, 1, 1),
            accession_number="123",
            source_url="http",
            score=0.9
        )
    ]
    
    claims = extract_claims("What was AAPL growth?", chunks)
    assert len(claims) == 1
    assert claims[0].claim_text == "AAPL revenue grew."

    # 2. Too few elements fallback
    mock_response_too_few = MagicMock()
    mock_response_too_few.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "claims": ["Factual claim 1"]
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_response_too_few
    
    two_chunks = chunks + [
        ChunkResult(
            text="MSFT cloud segment excelled! Azure growth is strong.",
            ticker="MSFT",
            filing_type="10-Q",
            quarter="Q1",
            fiscal_year=2024,
            section_type="MD&A",
            chunk_index=1,
            filing_date=date(2024, 1, 1),
            accession_number="456",
            source_url="http2",
            score=0.8
        )
    ]
    
    claims_2 = extract_claims("What were developments?", two_chunks)
    assert len(claims_2) == 2
    assert claims_2[0].claim_text == "Factual claim 1"
    assert claims_2[1].claim_text == "MSFT cloud segment excelled!"

def test_get_first_sentence():
    assert _get_first_sentence("First sentence. Second sentence!") == "First sentence."
    assert _get_first_sentence("Single sentence.") == "Single sentence."
    assert _get_first_sentence("No punctuation here") == "No punctuation here"
    assert _get_first_sentence("  Leading whitespace. Next  ") == "Leading whitespace."
