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
You are a financial fact extractor. Given N numbered document chunks and a user query, extract one concise factual statement per chunk that is relevant to the query.

Rules:
- Preserve EXACT numbers (dollar amounts, percentages, counts). Never generalize specific figures.
- Be precise about time frames: "six months ended" ≠ "Q2"; "through Q2" ≠ "in Q2".
- If a chunk is not relevant to the query, output null for that chunk.
- Combine all relevant details from a chunk into ONE statement.

Respond with ONLY valid JSON: {"claims": ["statement or null", ...]}
The list MUST have exactly N elements matching the N input chunks. No markdown, no extra keys.
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
    to the query. Executes parallel batched calls to the Groq LLM to respect TPM limits,
    parsing the output and falling back to the first sentence of a chunk if claims are missing/unparsable.
    """
    if not chunk_results:
        return []

    # Sort chunks to ensure contiguous chunks are adjacent
    sorted_chunks = sorted(
        chunk_results,
        key=lambda c: (
            c.ticker,
            c.fiscal_year,
            c.filing_type,
            c.quarter or "",
            c.section_type or "",
            c.chunk_index
        )
    )

    merged_chunks = []
    if sorted_chunks:
        current = sorted_chunks[0]
        last_index = current.chunk_index
        MAX_MERGE_CHAR_LIMIT = 8000
        for next_chunk in sorted_chunks[1:]:
            if (next_chunk.ticker == current.ticker and
                next_chunk.fiscal_year == current.fiscal_year and
                next_chunk.filing_type == current.filing_type and
                next_chunk.quarter == current.quarter and
                next_chunk.section_type == current.section_type and
                next_chunk.accession_number == current.accession_number and
                next_chunk.chunk_index == last_index + 1 and
                len(current.text) + len(next_chunk.text) < MAX_MERGE_CHAR_LIMIT):
                
                # Merge contiguous chunk content
                current = ChunkResult(
                    text=current.text + "\n" + next_chunk.text,
                    ticker=current.ticker,
                    filing_type=current.filing_type,
                    quarter=current.quarter,
                    fiscal_year=current.fiscal_year,
                    section_type=current.section_type,
                    chunk_index=current.chunk_index,
                    filing_date=current.filing_date,
                    accession_number=current.accession_number,
                    source_url=current.source_url,
                    score=max(current.score, next_chunk.score)
                )
                last_index = next_chunk.chunk_index
            else:
                merged_chunks.append(current)
                current = next_chunk
                last_index = current.chunk_index
        merged_chunks.append(current)

    logger.info(f"Merged {len(chunk_results)} retrieved chunks into {len(merged_chunks)} contiguous blocks.")

    # Batch size of 3 ensures we stay well below TPM limit per call
    BATCH_SIZE = 3
    batches = [merged_chunks[i:i + BATCH_SIZE] for i in range(0, len(merged_chunks), BATCH_SIZE)]
    
    api_key = GROQ_API_KEY or "dummy-key"
    import concurrent.futures
    
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

    state = {"primary_tpd_reached": False}

    def process_batch(batch_chunks: List[ChunkResult], batch_idx: int) -> List[Claim]:
        if not batch_chunks:
            return []
            
        # Format the numbered chunk texts
        user_prompt = f"User Query: {query}\n\n"
        for i, chunk in enumerate(batch_chunks, start=1):
            user_prompt += f"--- Chunk {i} ---\nTicker: {chunk.ticker}, Period: {chunk.quarter or '10-K'} {chunk.fiscal_year}\nContent:\n{chunk.text}\n\n"

        client = Groq(api_key=api_key)
        from config.settings import FALLBACK_LLM
        
        parsed_claims = []
        api_success = False
        import time
        max_retries = 3
        backoff_factor = 2.0
        
        models_to_try = [PRIMARY_LLM, FALLBACK_LLM]
        if state["primary_tpd_reached"]:
            models_to_try = [FALLBACK_LLM]
            
        for model_to_try in models_to_try:
            retries = 0
            should_bisect = False
            while retries <= max_retries:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model_to_try,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        max_tokens=1024
                    )
                    
                    response_text = completion.choices[0].message.content
                    data = json.loads(response_text)
                    if isinstance(data, dict) and "claims" in data:
                        parsed_claims = data["claims"]
                        api_success = True
                        break # Success!
                except Exception as e:
                    err_msg = str(e)
                    is_rate_limit = False
                    if "429" in err_msg or "rate_limit" in err_msg or "too many requests" in err_msg.lower() or "413" in err_msg:
                        is_rate_limit = True
                    
                    if is_rate_limit:
                        # Check if daily token limit (TPD) was reached
                        if "tpd" in err_msg.lower() or "tokens per day" in err_msg.lower():
                            logger.warning(f"Daily token limit (TPD) reached for model {model_to_try} in batch {batch_idx}. Falling back immediately.")
                            if model_to_try == PRIMARY_LLM:
                                state["primary_tpd_reached"] = True
                            break
                        
                        # Check if request is statically larger than the model's TPM limit (cannot be solved by waiting)
                        limit_match = re.search(r"Limit\s+(\d+),\s+Requested\s+(\d+)", err_msg)
                        if not limit_match:
                            limit_match = re.search(r"Requested\s+(\d+),\s+Limit\s+(\d+)", err_msg)
                        
                        if limit_match:
                            val1, val2 = int(limit_match.group(1)), int(limit_match.group(2))
                            if "limit" in limit_match.group(0).lower():
                                limit = val1 if "limit" in err_msg.lower().split("requested")[0] else val2
                                requested = val2 if "limit" in err_msg.lower().split("requested")[0] else val1
                            else:
                                limit = val2
                                requested = val1
                                
                            if requested > limit:
                                logger.warning(f"Request size {requested} exceeds model {model_to_try} TPM limit {limit}. Bisecting batch {batch_idx} of size {len(batch_chunks)}.")
                                should_bisect = True
                                break
                        
                        if retries < max_retries:
                            # Only sleep if Groq explicitly gives a retry-after header/message
                            match = re.search(r"try again in ([\d\.]+)s", err_msg)
                            if match:
                                sleep_time = min(float(match.group(1)) + 0.2, 3.0)  # cap at 3s
                                logger.warning(f"Rate limited for model {model_to_try} in batch {batch_idx}. Retrying in {sleep_time:.2f}s... Error: {e}")
                                time.sleep(sleep_time)
                                retries += 1
                                continue
                            else:
                                # No retry-after hint — switch model immediately, no sleep
                                logger.warning(f"Rate limited for model {model_to_try} in batch {batch_idx}. Switching model immediately.")
                                break
                        else:
                            logger.warning(f"Failed to extract claims via Groq model {model_to_try} for batch {batch_idx} after retries: {e}")
                            break
                    else:
                        logger.warning(f"Failed to extract claims via Groq model {model_to_try} for batch {batch_idx}: {e}")
                        break
            
            if should_bisect:
                if len(batch_chunks) > 1:
                    mid = len(batch_chunks) // 2
                    left_batch = batch_chunks[:mid]
                    right_batch = batch_chunks[mid:]
                    logger.info(f"Splitting batch {batch_idx} into left (size {len(left_batch)}) and right (size {len(right_batch)}).")
                    left_claims = process_batch(left_batch, batch_idx * 10 + 1)
                    right_claims = process_batch(right_batch, batch_idx * 10 + 2)
                    return left_claims + right_claims
                else:
                    # Single chunk is too large! Truncate the text to 1/2 size and retry
                    chunk = batch_chunks[0]
                    logger.warning(f"Single chunk index {chunk.chunk_index} is too large ({len(chunk.text)} chars). Truncating text by half.")
                    truncated_chunk = ChunkResult(
                        text=chunk.text[:len(chunk.text) // 2],
                        ticker=chunk.ticker,
                        filing_type=chunk.filing_type,
                        quarter=chunk.quarter,
                        fiscal_year=chunk.fiscal_year,
                        section_type=chunk.section_type,
                        chunk_index=chunk.chunk_index,
                        filing_date=chunk.filing_date,
                        accession_number=chunk.accession_number,
                        source_url=chunk.source_url,
                        score=chunk.score
                    )
                    return process_batch([truncated_chunk], batch_idx)
            
            if api_success:
                break

        batch_claims = []
        for i, chunk in enumerate(batch_chunks):
            claim_text = None
            if api_success and i < len(parsed_claims):
                val = parsed_claims[i]
                if isinstance(val, str) and val.strip():
                    claim_text = val.strip()
                    
            if not claim_text and (not api_success or i >= len(parsed_claims)):
                # Fallback to the first sentence if the entire API call failed or response was incomplete
                claim_text = _get_first_sentence(chunk.text)
                if not claim_text:
                    claim_text = chunk.text[:200]
                    
            # If we have a claim text, verify it is not a "no information" statement
            if claim_text:
                is_no_info = any(re.search(pat, claim_text) for pat in no_info_patterns)
                if is_no_info:
                    claim_text = None
                    
            if claim_text:
                batch_claims.append(
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
        return batch_claims

    claims = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all batches
        futures = [executor.submit(process_batch, batch, idx) for idx, batch in enumerate(batches)]
        # Gather results in order
        for future in futures:
            claims.extend(future.result())
            
    return claims

