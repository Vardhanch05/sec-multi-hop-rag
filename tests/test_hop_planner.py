import pytest
from datetime import date
from hypothesis import given, settings, strategies as st
from retrieval.hop_planner import (
    FilingPeriod,
    HopSpec,
    HopResolutionError,
    resolve_temporal_reference,
    plan_hops
)
from retrieval.query_classifier import HopPlan, PeriodSpec

# Feature: sec-rag-system, Property 8: Temporal reference resolution
@given(
    ticker=st.text(min_size=1, max_size=5).map(lambda s: s.upper()),
    ref=st.sampled_from(["last quarter", "last 4 quarters", "last year", "Q3 2023", "2023"]),
    available_periods=st.lists(
        st.builds(
            FilingPeriod,
            ticker=st.text(min_size=1, max_size=5).map(lambda s: s.upper()),
            quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
            fiscal_year=st.integers(min_value=2000, max_value=2030),
            filing_type=st.sampled_from(["10-Q", "10-K"]),
            filing_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 1, 1))
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100)
def test_temporal_reference_resolution(ticker, ref, available_periods):
    # Normalize ticker and quarters/filing_types to match
    for p in available_periods:
        p.ticker = ticker
        if p.filing_type == "10-K":
            p.quarter = None
        else:
            if p.quarter is None:
                p.quarter = "Q1"

    try:
        resolved = resolve_temporal_reference(ticker, ref, available_periods)
        for rp in resolved:
            assert rp in available_periods
            assert rp.ticker == ticker
    except HopResolutionError:
        pass

def test_hop_resolution_error_message():
    ticker = "AAPL"
    available = [
        FilingPeriod("AAPL", "Q1", 2023, "10-Q", date(2023, 2, 1)),
        FilingPeriod("AAPL", "Q2", 2023, "10-Q", date(2023, 5, 1)),
        FilingPeriod("AAPL", "Q3", 2023, "10-Q", date(2023, 8, 1)),
        FilingPeriod("AAPL", "Q4", 2023, "10-Q", date(2023, 11, 1)),
    ]
    
    with pytest.raises(HopResolutionError) as exc_info:
        resolve_temporal_reference(ticker, "Q3 2022", available)
        
    err_msg = str(exc_info.value)
    assert ticker in err_msg
    assert "Q3 2022" in err_msg
    assert "Q1 2023" in err_msg
    assert "Q2 2023" in err_msg
    assert "Q3 2023" in err_msg
    assert "Q4 2023" in err_msg
    assert err_msg == "No filings found for AAPL Q3 2022. Available periods: Q1 2023, Q2 2023, Q3 2023, Q4 2023."

def test_resolve_temporal_reference_rules():
    available = [
        FilingPeriod("MSFT", "Q1", 2023, "10-Q", date(2023, 2, 1)),
        FilingPeriod("MSFT", "Q2", 2023, "10-Q", date(2023, 5, 1)),
        FilingPeriod("MSFT", None, 2023, "10-K", date(2023, 12, 1)),
        FilingPeriod("MSFT", "Q3", 2023, "10-Q", date(2023, 8, 1)),
    ]
    
    # 1. Test "last quarter"
    resolved_lq = resolve_temporal_reference("MSFT", "last quarter", available)
    assert len(resolved_lq) == 1
    assert resolved_lq[0].quarter == "Q3"
    assert resolved_lq[0].fiscal_year == 2023
    
    # 2. Test "last year"
    resolved_ly = resolve_temporal_reference("MSFT", "last year", available)
    assert len(resolved_ly) == 1
    assert resolved_ly[0].quarter is None
    assert resolved_ly[0].filing_type == "10-K"
    assert resolved_ly[0].fiscal_year == 2023

    # 3. Test "last 4 quarters"
    available_5 = available + [
        FilingPeriod("MSFT", "Q4", 2022, "10-Q", date(2022, 11, 1)),
        FilingPeriod("MSFT", "Q3", 2022, "10-Q", date(2022, 8, 1)),
    ]
    resolved_4q = resolve_temporal_reference("MSFT", "last 4 quarters", available_5)
    assert len(resolved_4q) == 4
    quarters_returned = [p.quarter for p in resolved_4q]
    assert "Q3" in quarters_returned
    assert "Q2" in quarters_returned
    assert "Q1" in quarters_returned
    assert "Q4" in quarters_returned
    
    # 4. Test "Q2 2023"
    resolved_lit = resolve_temporal_reference("MSFT", "Q2 2023", available)
    assert len(resolved_lit) == 1
    assert resolved_lit[0].quarter == "Q2"
    assert resolved_lit[0].fiscal_year == 2023

def test_plan_hops_basic():
    available = [
        FilingPeriod("AAPL", "Q1", 2023, "10-Q", date(2023, 2, 1)),
        FilingPeriod("AAPL", "Q2", 2023, "10-Q", date(2023, 5, 1)),
        FilingPeriod("MSFT", "Q1", 2023, "10-Q", date(2023, 2, 1)),
    ]
    
    plan = HopPlan(
        hop_count=2,
        query_type="temporal_comparison",
        tickers=["AAPL"],
        periods=[
            PeriodSpec("AAPL", "last quarter", 0),
            PeriodSpec("AAPL", "Q1", 2023)
        ],
        section_hint="MD&A",
        requires_contradiction_check=True
    )
    
    specs = plan_hops(plan, available)
    assert len(specs) == 2
    
    assert specs[0].ticker == "AAPL"
    assert specs[0].quarter == "Q2"
    assert specs[0].fiscal_year == 2023
    assert specs[0].filing_type == "10-Q"
    assert specs[0].section_type == "MD&A"
    
    assert specs[1].ticker == "AAPL"
    assert specs[1].quarter == "Q1"
    assert specs[1].fiscal_year == 2023
    assert specs[1].filing_type == "10-Q"
    assert specs[1].section_type == "MD&A"
