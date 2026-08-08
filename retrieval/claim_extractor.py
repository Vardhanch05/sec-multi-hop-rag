"""
retrieval/claim_extractor.py
----------------------------
Extracts factual claims from SEC filing text chunks using batched Groq LLM queries.
"""

import json
import re
import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from groq import Groq
from config.settings import GROQ_API_KEY, PRIMARY_LLM
from retrieval.retriever import ChunkResult

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Claim:
    claim_text: str
    ticker: str
    quarter: Optional[str]
    fiscal_year: int
    section_type: str
    chunk_index: int
    filing_date: date
    accession_number: str
    source_url: str

SYSTEM_PROMPT = """
You are an expert financial research AI assistant. Your task is to extract exactly one concise, factual claim or statement from each of the provided numbered financial document chunks (1 to N) that is relevant to the user's query.

If a chunk does not contain any information relevant to the user's query, you MUST return null (or None) for that chunk's claim in the list.

You must respond with ONLY a valid JSON object containing a single key "claims", which maps to a list:
{
  "claims": [
    "concise factual statement relevant to the query from chunk 1",
    null
  ]
}
Do not include any markdown styling, formatting, or extra text.
"""

def _get_first_sentence(text: str) -> str:
    """Extracts the first sentence of a text chunk using regex matching."""
    text = text.strip()
    if not text:
        return ""
    # Split by typical sentence boundaries: . ! ? followed by space or newline
    match = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
    if match:
        return match[0]
    return text

def extract_claims(query: str, chunk_results: List[ChunkResult]) -> List[Claim]:
    """
    Extracts structured factual claims from a list of retrieved SEC chunks relative
    to the query. Executes a single batched call to the Groq LLM, parsing the output
    and falling back to the first sentence of a chunk if claims are missing or unparsable.
    """
    if not chunk_results:
        return []

    # Format the numbered chunk texts
    user_prompt = f"User Query: {query}\n\n"
    for i, chunk in enumerate(chunk_results, start=1):
        user_prompt += f"--- Chunk {i} ---\nTicker: {chunk.ticker}, Period: {chunk.quarter or '10-K'} {chunk.fiscal_year}\nContent:\n{chunk.text}\n\n"

    parsed_claims = []
    api_key = GROQ_API_KEY or "dummy-key"
    client = Groq(api_key=api_key)
    
    # Try with PRIMARY_LLM, fallback to FALLBACK_LLM on failure (e.g. 413 or 429 errors)
    from config.settings import FALLBACK_LLM
    
    api_success = False
    for model_to_try in [PRIMARY_LLM, FALLBACK_LLM]:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_to_try,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            data = json.loads(response_text)
            if isinstance(data, dict) and "claims" in data:
                parsed_claims = data["claims"]
                api_success = True
                break # Success!
                
        except Exception as e:
            logger.warning(f"Failed to extract claims via Groq model {model_to_try}: {e}")

    claims = []
    
    # Common patterns indicating "no information"
    no_info_patterns = [
        r"(?i)^no\s+relevant\s+information",
        r"(?i)^no\s+information",
        r"(?i)^not\s+mentioned",
        r"(?i)^n/a",
        r"(?i)^not\s+found",
        r"(?i)^chunk\s+\d+\s+does\s+not",
        r"(?i)^this\s+chunk\s+does\s+not",
        r"(?i)^no\s+claim",
        r"(?i)^null$"
    ]

    for i, chunk in enumerate(chunk_results):
        claim_text = None
        
        if api_success:
            if i < len(parsed_claims):
                val = parsed_claims[i]
                if isinstance(val, str) and val.strip():
                    claim_text = val.strip()
        else:
            # Fallback to the first sentence if the entire API call failed
            claim_text = _get_first_sentence(chunk.text)
            if not claim_text:
                claim_text = chunk.text[:200]
                
        # If we have a claim text, verify it is not a "no information" statement
        if claim_text:
            is_no_info = any(re.search(pat, claim_text) for pat in no_info_patterns)
            if is_no_info:
                claim_text = None
                
        if claim_text:
            claims.append(
                Claim(
                    claim_text=claim_text,
                    ticker=chunk.ticker,
                    quarter=chunk.quarter,
                    fiscal_year=chunk.fiscal_year,
                    section_type=chunk.section_type,
                    chunk_index=chunk.chunk_index,
                    filing_date=chunk.filing_date,
                    accession_number=chunk.accession_number,
                    source_url=chunk.source_url
                )
            )
            
    return claims

