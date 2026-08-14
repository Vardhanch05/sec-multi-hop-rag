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

A list of potential contradictions detected by an automated system may be provided:
- Critically evaluate each potential contradiction.
- Distinguish between true factual conflicts (e.g., conflicting numbers for the exact same metric, period, and filing) versus compatible differences (e.g., three months vs. six months ended, Q2 2023 vs. Q2 2022, or different vehicle models).
- If they are compatible, resolve them logically in your explanation (e.g., by explaining that one refers to the quarter and the other to the first half of the year). ONLY report a contradiction in your answer if it is a true, irreconcilable conflict.

Do not include any citations or markdown links in your text; the system will attach structured citations separately.
Be concise, factual, and direct.
"""

def synthesize(query: str, claims: List[Claim], contradiction_report: ContradictionReport) -> ResponsePayload:
    start_time = time.time()
    
    if not claims:
        return ResponsePayload(
            answer="No relevant information was found in the retrieved documents to answer this query.",
            citations=[],
            contradictions=[],
            latency_ms=int((time.time() - start_time) * 1000),
            model_used="none",
            contradiction_detection_skipped=False
        )
        
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
    force_fallback = False
    
    for attempt in range(1, 4):
        current_model = FALLBACK_LLM if force_fallback else PRIMARY_LLM
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=current_model,
                temperature=0.2,
                max_tokens=512
            )
            answer_text = completion.choices[0].message.content
            model_used = current_model
            break
        except RateLimitError as e:
            if not force_fallback:
                logger.warning(f"RateLimitError on attempt {attempt}. Switching to fallback immediately.")
                force_fallback = True
                model_used = FALLBACK_LLM
                continue
            else:
                logger.warning(f"RateLimitError on attempt {attempt}. Both models rate-limited.")
                answer_text = "System is currently overloaded (Rate Limit). Please try again shortly."
                model_used = current_model
        except Exception as e:
            err_str = str(e)
            if ("413" in err_str or "Request too large" in err_str) and not force_fallback:
                logger.warning(f"Context too large for {current_model} (attempt {attempt}). Switching to fallback.")
                force_fallback = True
                continue
            
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


def synthesize_stream(query: str, claims: List[Claim], contradiction_report: ContradictionReport):
    """
    Yields answer tokens sequentially during generation.
    Yields a ResponsePayload as the final item for metadata collection.
    """
    start_time = time.time()
    
    if not claims:
        msg = "No relevant information was found in the retrieved documents to answer this query."
        yield msg
        yield ResponsePayload(
            answer=msg,
            citations=[],
            contradictions=[],
            latency_ms=int((time.time() - start_time) * 1000),
            model_used="none",
            contradiction_detection_skipped=False
        )
        return
        
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
    
    model_used = PRIMARY_LLM
    force_fallback = False
    full_text = []
    
    for attempt in range(1, 4):
        current_model = FALLBACK_LLM if force_fallback else PRIMARY_LLM
        try:
            completion_stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=current_model,
                temperature=0.2,
                stream=True,
                max_tokens=512
            )
            for chunk in completion_stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text.append(content)
                    yield content
            model_used = current_model
            break
        except RateLimitError as e:
            if not force_fallback:
                logger.warning(f"RateLimitError on streaming attempt {attempt}. Switching to fallback immediately.")
                force_fallback = True
                model_used = FALLBACK_LLM
                continue
            else:
                logger.warning(f"RateLimitError on streaming attempt {attempt}. Both models rate-limited.")
                err_msg = "System is currently overloaded (Rate Limit). Please try again shortly."
                full_text.append(err_msg)
                yield err_msg
                model_used = current_model
                break
        except Exception as e:
            err_str = str(e)
            if ("413" in err_str or "Request too large" in err_str) and not force_fallback:
                logger.warning(f"Context too large for stream {current_model} (attempt {attempt}). Switching to fallback.")
                force_fallback = True
                continue
            
            logger.error(f"Error during LLM streaming synthesis: {e}")
            err_msg = f"An error occurred during synthesis: {e}"
            full_text.append(err_msg)
            yield err_msg
            model_used = current_model
            break
            
    latency_ms = int((time.time() - start_time) * 1000)
    answer_text = "".join(full_text)
    
    yield ResponsePayload(
        answer=answer_text,
        citations=citations,
        contradictions=contradiction_report.contradictions,
        latency_ms=latency_ms,
        model_used=model_used,
        contradiction_detection_skipped=contradiction_report.timed_out
    )

