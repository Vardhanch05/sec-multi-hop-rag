import json
import logging
from dataclasses import dataclass
from typing import List, Optional
from datetime import date
import time
from groq import Groq, RateLimitError
from config.settings import GROQ_API_KEY, PRIMARY_LLM, FALLBACK_LLM

logger = logging.getLogger(__name__)

class QueryParseError(Exception):
    pass

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
If the user query does not explicitly name a company or ticker symbol, and no UI filters are provided, you MUST output an empty list [] for tickers. Do not hallucinate or guess a ticker (e.g., do not output placeholders like PROJECTED_TICKER).

When determining section_hint, use these rules:
- Revenue figures, financial results, segment performance -> "Financial Statements"
- Management commentary on results, business trends -> "MD&A"  
- General risk disclosures (market risk, credit risk, operational risk) -> "Risk Factors"
- Legal proceedings, pending litigation, lawsuits, legal claims, investigations, SEC inquiries, regulatory enforcement, government actions, penalties, fines, consent orders -> null (do not filter by section, search across all sections)
- "Item 1. Business" is a 10-K only section about company description —
  NEVER use it for financial figures or quarterly data
"""

def classify_query(query: str, ui_filters: UIFilters) -> HopPlan:
    """
    Classifies a query into a HopPlan using Groq LLM.
    """
    # Using a dummy client if GROQ_API_KEY is missing for CI tests.
    api_key = GROQ_API_KEY or "dummy-key"
    client = Groq(api_key=api_key)
    
    current_date = date.today().isoformat()
    user_prompt = f"Current Date: {current_date}\nUser Query: {query}\n"
    user_prompt += "Important: Resolve relative references like 'last year' to concrete fiscal_year integers before outputting JSON.\n"
    user_prompt += "If the query does not specify a temporal reference, output null for fiscal_year.\n"
    user_prompt += "If the query contains a garbled or completely unrecognizable temporal reference, you MUST output -1 for fiscal_year.\n"
    if ui_filters:
        user_prompt += f"UI Filters Context: Tickers={ui_filters.tickers}, Start={ui_filters.start_date}, End={ui_filters.end_date}, FilingType={ui_filters.filing_type}\n"
        
    # Try with PRIMARY_LLM, retry on RateLimitError, and fallback to FALLBACK_LLM
    backoffs = [0, 0]
    max_attempts = 3
    force_fallback = False
    completion = None
    response_text = None
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        current_model = FALLBACK_LLM if force_fallback else (PRIMARY_LLM if attempt == 1 else FALLBACK_LLM)
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=current_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=256
            )
            response_text = completion.choices[0].message.content
            break
        except RateLimitError as e:
            last_error = e
            if not force_fallback:
                logger.warning(f"RateLimitError in query classification (attempt {attempt}). Switching to fallback immediately.")
                force_fallback = True
            else:
                logger.warning(f"RateLimitError in query classification (attempt {attempt}). Fallback also rate-limited.")
        except Exception as e:
            last_error = e
            logger.error(f"Error during query classification attempt {attempt} with model {current_model}: {e}")
            if attempt < max_attempts:
                sleep_time = 1.0 * attempt
                logger.warning(f"Transient error in query classification. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                # If we hit an exception with the primary model, also fallback to the smaller model for subsequent attempts
                if not force_fallback:
                    force_fallback = True
            else:
                raise RuntimeError(f"Query classification failed: {e}")
                
    if response_text is None:
        raise RuntimeError(f"Query classification failed: {last_error or 'No response generated due to rate limits'}")
        
    try:
        data = json.loads(response_text)
        
        tickers = data.get("tickers", [])
        
        from db.queries import get_all_tickers
        valid_tickers = get_all_tickers()
        
        # Filter out any hallucinated tickers that are not in the database
        invalid_tickers = [t for t in tickers if t.upper() not in valid_tickers]
        tickers = [t for t in tickers if t.upper() in valid_tickers]
        
        # Merge UI filters
        if ui_filters and ui_filters.tickers:
            for t in ui_filters.tickers:
                if t not in tickers:
                    tickers.append(t)
                    
        if not tickers:
            error_msg = "Please specify a valid company in your query or select a ticker in the UI filters."
            if invalid_tickers:
                error_msg = f"The requested company/ticker ({', '.join(invalid_tickers)}) is not available in the database. " + error_msg
            raise QueryParseError(error_msg)
            
        periods = []
        for p in data.get("periods", []):
            spec = PeriodSpec(**p)
            if spec.fiscal_year == -1:
                raise QueryParseError("Could not resolve relative temporal reference to a concrete year")
                
            if not spec.ticker:
                spec.ticker = tickers[0]
                
            periods.append(spec)
        
        hop_count = len(periods)
        
        # Apply cross-company detection rule (N companies × M periods = N×M hops)
        if data.get("query_type") == "cross_company":
            unique_tickers = len(set(p.ticker for p in periods))
            unique_timeframes = len(set((p.fiscal_year, p.quarter) for p in periods))
            if len(periods) < len(tickers) and unique_timeframes > 0:
                hop_count = len(tickers) * unique_timeframes
        
        # Override: single-ticker queries never need cross-claim NLI contradiction checking.
        # NLI only produces value when comparing two *different companies* making conflicting claims
        # about the same metric. Single-ticker temporal comparisons are handled by the synthesizer directly.
        requires_contradiction_check = data.get("requires_contradiction_check", False)
        if len(tickers) <= 1:
            requires_contradiction_check = False
        
        return HopPlan(
            hop_count=hop_count,
            query_type=data.get("query_type", "single_hop"),
            tickers=tickers,
            periods=periods,
            section_hint=data.get("section_hint"),
            requires_contradiction_check=requires_contradiction_check
        )
        
    except QueryParseError as e:
        raise
    except Exception as e:
        logger.error(f"Failed to classify query: {e}")
        raise RuntimeError(f"Query classification failed: {e}")
