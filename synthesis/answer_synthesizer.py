"""
synthesis/answer_synthesizer.py
-------------------------------
Synthesizes the final answer using retrieved claims and contradiction reports.
"""

import time
import logging
from dataclasses import dataclass
from typing import List

from groq import Groq, RateLimitError
from config.settings import GROQ_API_KEY, PRIMARY_LLM, FALLBACK_LLM
from retrieval.claim_extractor import Claim
from contradiction.contradiction_report import ContradictionEvent, ContradictionReport

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Citation:
    filing_type: str
    section: str
    ticker: str
    fiscal_year: int
    accession_number: str

@dataclass(frozen=True)
class ResponsePayload:
    answer: str
    citations: List[Citation]
    contradictions: List[ContradictionEvent]
    latency_ms: int
    model_used: str
    contradiction_detection_skipped: bool

SYSTEM_PROMPT = """
You are an expert financial analyst. Synthesize an answer to the user's query based ONLY on the provided financial claims.
If there are contradictions provided, mention them explicitly in your answer.
Do not include any citations or markdown links in your text; the system will attach structured citations separately.
Be concise, factual, and direct.
"""

def synthesize(query: str, claims: List[Claim], contradiction_report: ContradictionReport) -> ResponsePayload:
    start_time = time.time()
    
    # 1. Build Citation list
    citations_map = {}
    for c in claims:
        filing_type = "10-Q" if c.quarter else "10-K"
        citation = Citation(
            filing_type=filing_type,
            section=c.section_type,
            ticker=c.ticker,
            fiscal_year=c.fiscal_year,
            accession_number=c.accession_number
        )
        # Deduplicate citations by accession_number and section
        key = (citation.accession_number, citation.section)
        if key not in citations_map:
            citations_map[key] = citation
    citations = list(citations_map.values())
    
    # 2. Build Prompt
    user_prompt = f"User Query: {query}\n\n"
    
    user_prompt += "--- Retrieved Claims ---\n"
    for i, c in enumerate(claims, 1):
        user_prompt += f"[{i}] Ticker: {c.ticker}, {c.fiscal_year} {c.quarter or 'Annual'}: {c.claim_text}\n"
        
    if contradiction_report.contradictions:
        user_prompt += "\n--- Detected Contradictions ---\n"
        for i, ce in enumerate(contradiction_report.contradictions, 1):
            user_prompt += f"Contradiction {i}: The claim '{ce.claim_a}' conflicts with '{ce.claim_b}'.\n"

    api_key = GROQ_API_KEY or "dummy-key"
    client = Groq(api_key=api_key)
    
    answer_text = "Failed to generate answer."
    model_used = PRIMARY_LLM
    
    # Retry logic (19.2)
    backoffs = [0, 1, 2]
    max_attempts = 4
    
    for attempt in range(1, max_attempts + 1):
        current_model = PRIMARY_LLM if attempt <= 3 else FALLBACK_LLM
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=current_model,
                temperature=0.2
            )
            answer_text = completion.choices[0].message.content
            model_used = current_model
            break
        except RateLimitError as e:
            if attempt <= 3:
                sleep_time = backoffs[attempt - 1]
                logger.warning(f"RateLimitError on attempt {attempt}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.warning(f"RateLimitError on attempt {attempt}. Exhausted fallback attempts.")
                answer_text = "System is currently overloaded (Rate Limit). Please try again later."
                model_used = current_model
                # No exception raised, gracefully returns
        except Exception as e:
            logger.error(f"Error during LLM synthesis: {e}")
            answer_text = f"An error occurred during synthesis: {e}"
            model_used = current_model
            break
    
    # 3. Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)
    
    return ResponsePayload(
        answer=answer_text,
        citations=citations,
        contradictions=contradiction_report.contradictions,
        latency_ms=latency_ms,
        model_used=model_used,
        contradiction_detection_skipped=contradiction_report.timed_out
    )
