import json
from unittest.mock import patch, MagicMock
from retrieval.query_classifier import classify_query, UIFilters

@patch('retrieval.query_classifier.Groq')
def test_query_classifier_single_hop(mock_groq_class):
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "hop_count": 1,
            "query_type": "single_hop",
            "tickers": ["AAPL"],
            "periods": [{"ticker": "AAPL", "quarter": "Q1", "fiscal_year": 2024}],
            "section_hint": "MD&A",
            "requires_contradiction_check": False
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    
    plan = classify_query("What was AAPL revenue in Q1 2024?", UIFilters())
    
    assert plan.hop_count == 1
    assert plan.query_type == "single_hop"
    assert plan.tickers == ["AAPL"]
    assert len(plan.periods) == 1
    assert plan.periods[0].fiscal_year == 2024
    assert plan.section_hint == "MD&A"
    assert not plan.requires_contradiction_check

@patch('retrieval.query_classifier.Groq')
def test_query_classifier_cross_company(mock_groq_class):
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "hop_count": 2,
            "query_type": "cross_company",
            "tickers": ["MSFT", "GOOGL"],
            "periods": [
                {"ticker": "MSFT", "quarter": "Q1", "fiscal_year": 2024},
                {"ticker": "GOOGL", "quarter": "Q1", "fiscal_year": 2024}
            ],
            "section_hint": None,
            "requires_contradiction_check": True
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    
    plan = classify_query("Compare MSFT and GOOGL performance in Q1 2024", UIFilters())
    
    assert plan.query_type == "cross_company"
    assert set(plan.tickers) == {"MSFT", "GOOGL"}
    assert len(plan.periods) == 2
    assert plan.hop_count == 2
    assert plan.requires_contradiction_check
