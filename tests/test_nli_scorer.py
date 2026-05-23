"""
tests/test_nli_scorer.py
------------------------
"""
import pytest
import time
from datetime import date
from hypothesis import given, settings, strategies as st
from unittest.mock import patch, MagicMock

from config.settings import CONTRADICTION_THRESHOLD, MAX_NLI_PAIRS
from retrieval.claim_extractor import Claim
from contradiction.nli_scorer import score_contradictions, _process_claims, is_cross_period

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

@settings(max_examples=100)
@given(claims=st.lists(claim_st, min_size=2, max_size=10))
def test_nli_cross_period_only(claims):
    """
    Property 11: NLI scoring restricted to cross-period pairs
    Validates: Requirements 4.1
    """
    with patch("contradiction.nli_scorer.get_model") as mock_get_model, \
         patch("contradiction.nli_scorer.insert_contradiction_event") as mock_db:
        
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        def fake_predict(text_pairs, apply_softmax=False):
            return [[0.1, 0.8, 0.1] for _ in text_pairs]
        
        mock_model.predict.side_effect = fake_predict
        
        _process_claims(claims)
        
        if mock_model.predict.called:
            args, kwargs = mock_model.predict.call_args
            text_pairs = args[0]
            
            cross_period_pairs = []
            for i in range(len(claims)):
                for j in range(len(claims)):
                    if i != j and is_cross_period(claims[i], claims[j]):
                        cross_period_pairs.append((claims[i], claims[j]))
            
            expected_len = min(len(cross_period_pairs), MAX_NLI_PAIRS)
            assert len(text_pairs) == expected_len

@settings(max_examples=100)
@given(claims=st.lists(claim_st, min_size=2, max_size=10))
def test_contradiction_event_completeness(claims):
    """
    Property 12: Contradiction event completeness
    Validates: Requirements 4.2
    """
    with patch("contradiction.nli_scorer.get_model") as mock_get_model, \
         patch("contradiction.nli_scorer.insert_contradiction_event") as mock_db:
         
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        def fake_predict(text_pairs, apply_softmax=False):
            # 0 is the contradiction class, giving 0.99
            return [[0.99, 0.005, 0.005] for _ in text_pairs]
            
        mock_model.predict.side_effect = fake_predict
        
        events = _process_claims(claims)
        
        for event in events:
            assert event.ticker is not None
            assert event.filing_ref_a is not None
            assert event.filing_ref_b is not None
            assert event.claim_a is not None
            assert event.claim_b is not None
            assert event.confidence_score >= CONTRADICTION_THRESHOLD
            assert event.query_id is not None

def test_contradiction_threshold_default():
    assert isinstance(CONTRADICTION_THRESHOLD, float)

def test_nli_timeout_sets_flag():
    with patch("contradiction.nli_scorer._process_claims") as mock_process:
        def slow_process(claims):
            time.sleep(0.5)
            return []
        mock_process.side_effect = slow_process
        
        claims = [Claim(claim_text="dummy", ticker="AAPL", quarter="Q1", fiscal_year=2024, section_type="Item 1", chunk_index=0, filing_date=date.today(), accession_number="123", source_url="")]
        report = score_contradictions(claims, timeout_seconds=0.1)
        
        assert report.timed_out is True
        assert len(report.contradictions) == 0
