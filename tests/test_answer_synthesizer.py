"""
tests/test_answer_synthesizer.py
--------------------------------
"""
import pytest
from datetime import date
from hypothesis import given, settings, strategies as st
from unittest.mock import patch, MagicMock

from groq import RateLimitError
from config.settings import FALLBACK_LLM, PRIMARY_LLM
from retrieval.claim_extractor import Claim
from contradiction.contradiction_report import ContradictionEvent, ContradictionReport
from synthesis.answer_synthesizer import synthesize

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

claim_st = st.builds(
    Claim,
    claim_text=st.text(min_size=10, max_size=100),
    ticker=st.just("AAPL"),
    quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
    fiscal_year=st.integers(min_value=2020, max_value=2025),
    section_type=st.just("Item 1A"),
    chunk_index=st.integers(min_value=0, max_value=10),
    filing_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31)),
    accession_number=st.text(min_size=10, max_size=20),
    source_url=st.just("http://example.com")
)

contradiction_st = st.builds(
    ContradictionEvent,
    ticker=st.just("AAPL"),
    filing_ref_a=st.text(min_size=10, max_size=20),
    filing_ref_b=st.text(min_size=10, max_size=20),
    claim_a=st.text(min_size=5, max_size=50),
    claim_b=st.text(min_size=5, max_size=50),
    confidence_score=st.floats(min_value=0.75, max_value=1.0),
    query_id=st.just("dummy-query-id")
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(claims=st.lists(claim_st, min_size=1, max_size=5))
def test_llm_retry_and_fallback(claims):
    """
    Property 13: LLM retry and fallback
    Validates: Requirements 5.3
    """
    report = ContradictionReport(contradictions=[], timed_out=False)
    
    with patch("synthesis.answer_synthesizer.Groq") as mock_groq_class, \
         patch("time.sleep"):  # avoid actual sleeping during backoff
        
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        # We need the first 3 calls to raise RateLimitError, and the 4th to succeed
        mock_message = MagicMock()
        mock_message.content = "Fallback answer"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        # Create a mock response for the RateLimitError
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        
        rate_limit_error = RateLimitError(
            message="Rate limited",
            response=mock_response,
            body={"error": {"message": "Rate limited"}}
        )
        
        # side_effect list: first 3 raise, 4th returns completion
        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            rate_limit_error,
            mock_completion
        ]
        
        payload = synthesize("What is the revenue?", claims, report)
        
        assert mock_client.chat.completions.create.call_count == 4
        # The 4th call should use FALLBACK_LLM
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == FALLBACK_LLM
        assert payload.model_used == FALLBACK_LLM
        assert payload.answer == "Fallback answer"

@settings(max_examples=100)
@given(claims=st.lists(claim_st, min_size=1, max_size=10))
def test_citation_field_completeness(claims):
    """
    Property 14: Citation field completeness
    Validates: Requirements 5.1
    """
    report = ContradictionReport(contradictions=[], timed_out=False)
    
    with patch("synthesis.answer_synthesizer.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = "Dummy answer"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        payload = synthesize("test query", claims, report)
        
        # Every citation must have all fields non-null
        for cit in payload.citations:
            assert cit.filing_type is not None
            assert cit.section is not None
            assert cit.ticker is not None
            assert cit.fiscal_year is not None
            assert cit.accession_number is not None

@settings(max_examples=100)
@given(contradictions=st.lists(contradiction_st, min_size=0, max_size=5))
def test_contradiction_payload_propagation(contradictions):
    """
    Property 15: Contradiction payload propagation
    Validates: Requirements 5.4
    """
    report = ContradictionReport(contradictions=contradictions, timed_out=False)
    claims = [] # Empty claims list is fine
    
    with patch("synthesis.answer_synthesizer.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = "Dummy answer"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        payload = synthesize("test query", claims, report)
        
        assert len(payload.contradictions) == len(contradictions)
        for original, copied in zip(contradictions, payload.contradictions):
            assert original == copied
