import json
import logging
from dataclasses import dataclass
from typing import List, Optional
from datetime import date
from groq import Groq
from config.settings import GROQ_API_KEY, PRIMARY_LLM

logger = logging.getLogger(__name__)

@dataclass
class PeriodSpec:
    ticker: str
    quarter: Optional[str]
    fiscal_year: int

@dataclass
class HopPlan:
    hop_count: int
    query_type: str
    tickers: List[str]
    periods: List[PeriodSpec]
    section_hint: Optional[str]
    requires_contradiction_check: bool

@dataclass
class UIFilters:
    tickers: Optional[List[str]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    filing_type: Optional[str] = None

SYSTEM_PROMPT = """
You are an expert financial research AI assistant. Your task is to classify user queries related to SEC filings and produce a structured JSON execution plan (HopPlan).
You must respond with ONLY valid JSON and no markdown formatting or extra text.
The JSON must have the following schema:
{
  "hop_count": <integer>,
  "query_type": <"single_hop"|"temporal_comparison"|"cross_company"|"trend_analysis">,
  "tickers": [<string>, ...],
  "periods": [
    {"ticker": <string>, "quarter": <string or null>, "fiscal_year": <integer>},
    ...
  ],
  "section_hint": <string or null>,
  "requires_contradiction_check": <boolean>
}
If the user specifies UI filters, use them to restrict your ticker and date selections.
"""

def classify_query(query: str, ui_filters: UIFilters) -> HopPlan:
    """
    Classifies a query into a HopPlan using Groq LLM.
    """
    # Using a dummy client if GROQ_API_KEY is missing for CI tests.
    api_key = GROQ_API_KEY or "dummy-key"
    client = Groq(api_key=api_key)
    
    user_prompt = f"User Query: {query}\n"
    if ui_filters:
        user_prompt += f"UI Filters Context: Tickers={ui_filters.tickers}, Start={ui_filters.start_date}, End={ui_filters.end_date}, FilingType={ui_filters.filing_type}\n"
        
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model=PRIMARY_LLM,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)
        
        periods = [PeriodSpec(**p) for p in data.get("periods", [])]
        tickers = data.get("tickers", [])
        
        hop_count = len(periods)
        
        # Apply cross-company detection rule (N companies × M periods = N×M hops)
        if data.get("query_type") == "cross_company":
            unique_tickers = len(set(p.ticker for p in periods))
            unique_timeframes = len(set((p.fiscal_year, p.quarter) for p in periods))
            if len(periods) < len(tickers) and unique_timeframes > 0:
                hop_count = len(tickers) * unique_timeframes
        
        return HopPlan(
            hop_count=hop_count,
            query_type=data.get("query_type", "single_hop"),
            tickers=tickers,
            periods=periods,
            section_hint=data.get("section_hint"),
            requires_contradiction_check=data.get("requires_contradiction_check", False)
        )
        
    except Exception as e:
        logger.error(f"Failed to classify query: {e}")
        # Return a safe fallback
        return HopPlan(
            hop_count=1,
            query_type="single_hop",
            tickers=ui_filters.tickers if ui_filters and ui_filters.tickers else [],
            periods=[],
            section_hint=None,
            requires_contradiction_check=False
        )
