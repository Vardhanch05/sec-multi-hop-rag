import pytest
from datetime import date
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

from retrieval.retriever import retrieve_hops, ChunkResult, doc_to_chunk_result
from retrieval.hop_planner import HopSpec, FilingPeriod, plan_hops, HopResolutionError
from retrieval.query_classifier import HopPlan, PeriodSpec

# Feature: sec-rag-system, Property 7: Retriever hop count and filter correctness
@given(
    hop_specs=st.lists(
        st.builds(
            HopSpec,
            ticker=st.text(min_size=1, max_size=5).map(lambda s: s.upper()),
            quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
            fiscal_year=st.integers(min_value=2000, max_value=2030),
            filing_type=st.sampled_from(["10-Q", "10-K"]),
            section_type=st.sampled_from(["MD&A", "Risk Factors", "Forward Guidance", "Financial Statements", None])
        ),
        min_size=0,
        max_size=10
    )
)
@settings(max_examples=100)
@patch('retrieval.retriever.get_vector_store')
@patch('retrieval.retriever.embed_query')
def test_retriever_hop_count_and_filters(mock_embed, mock_get_store, hop_specs):
    # Setup mocks
    mock_embed.return_value = [0.1] * 384
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store
    mock_store.search.return_value = []

    # Run retrieve_hops
    results = retrieve_hops("test query", hop_specs)

    # Assert exactly unique N queries were returned
    assert len(results) == len(set(hop_specs))
    assert mock_store.search.call_count == len(hop_specs)

    # Verify that each HopSpec had a corresponding vector store search call with the correct filters
    calls = mock_store.search.call_args_list
    assert len(calls) == len(hop_specs)
    
    for spec in hop_specs:
        expected_quarter = spec.quarter if spec.quarter is not None else ""
        found = False
        for call in calls:
            _, kwargs = call
            filters = kwargs.get("filters", {})
            if (filters.get("ticker") == spec.ticker and
                filters.get("fiscal_year") == spec.fiscal_year and
                filters.get("filing_type") == spec.filing_type and
                filters.get("quarter") == expected_quarter):
                
                if spec.section_type:
                    if filters.get("section_type") == spec.section_type:
                        found = True
                        break
                else:
                    if "section_type" not in filters:
                        found = True
                        break
        assert found, f"No matching search call found for HopSpec: {spec}"

# Feature: sec-rag-system, Property 10: Section type filter enforcement
@given(
    hop_spec=st.builds(
        HopSpec,
        ticker=st.text(min_size=1, max_size=5).map(lambda s: s.upper()),
        quarter=st.sampled_from(["Q1", "Q2", "Q3", "Q4", None]),
        fiscal_year=st.integers(min_value=2000, max_value=2030),
        filing_type=st.sampled_from(["10-Q", "10-K"]),
        section_type=st.sampled_from(["MD&A", "Risk Factors", "Forward Guidance", "Financial Statements"])
    )
)
@settings(max_examples=100)
@patch('retrieval.retriever.get_vector_store')
@patch('retrieval.retriever.embed_query')
def test_section_type_filter_enforcement(mock_embed, mock_get_store, hop_spec):
    mock_embed.return_value = [0.1] * 384
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store
    
    # Return mock results matching the search filter
    mock_store.search.return_value = [
        {
            "text": "test",
            "ticker": hop_spec.ticker,
            "filing_type": hop_spec.filing_type,
            "quarter": hop_spec.quarter if hop_spec.quarter is not None else "",
            "fiscal_year": hop_spec.fiscal_year,
            "section_type": hop_spec.section_type,
            "chunk_index": 0,
            "filing_date": "2024-01-01",
            "accession_number": "123",
            "source_url": "http",
            "score": 0.9
        }
    ]
    
    results = retrieve_hops("test query", [hop_spec])
    for spec, chunks in results.items():
        if spec.section_type:
            for chunk in chunks:
                assert chunk.section_type == spec.section_type

# Feature: sec-rag-system, Property 19: UI filter propagation to hop specs
@given(
    ui_filing_type=st.sampled_from(["10-Q", "10-K", None]),
    ui_tickers=st.lists(st.text(min_size=1, max_size=5).map(lambda s: s.upper()), min_size=0, max_size=3)
)
@settings(max_examples=100)
def test_ui_filter_propagation(ui_filing_type, ui_tickers):
    available = [
        FilingPeriod("AAPL", "Q1", 2023, "10-Q", date(2023, 2, 1)),
        FilingPeriod("AAPL", None, 2023, "10-K", date(2023, 12, 1)),
        FilingPeriod("MSFT", "Q1", 2023, "10-Q", date(2023, 2, 1)),
    ]
    
    filtered_avail = available
    if ui_tickers:
        filtered_avail = [p for p in filtered_avail if p.ticker in ui_tickers]
    if ui_filing_type:
        filtered_avail = [p for p in filtered_avail if p.filing_type == ui_filing_type]
        
    periods = []
    for p in filtered_avail:
        periods.append(PeriodSpec(p.ticker, p.quarter, p.fiscal_year))
        
    plan = HopPlan(
        hop_count=len(periods),
        query_type="temporal_comparison",
        tickers=ui_tickers if ui_tickers else ["AAPL"],
        periods=periods,
        section_hint="MD&A",
        requires_contradiction_check=False
    )
    
    if not filtered_avail and periods:
        with pytest.raises(HopResolutionError):
            plan_hops(plan, available)
    else:
        specs = plan_hops(plan, available)
        for spec in specs:
            if ui_filing_type:
                assert spec.filing_type == ui_filing_type
            if ui_tickers:
                assert spec.ticker in ui_tickers

# Feature: sec-rag-system, collection targeting validation
def test_results_from_sec_chunks_collection(monkeypatch):
    from ingestion.vector_store import get_vector_store
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chromadb")
    
    store = get_vector_store()
    assert store.collection.name == "sec_chunks"
