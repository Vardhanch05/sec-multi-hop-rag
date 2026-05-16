import re
import json
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

from ingestion.edgar_client import FilingRef

@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str
    quarter: str | None
    fiscal_year: int
    section_type: str         # "MD&A"|"Risk Factors"|"Forward Guidance"|"Financial Statements"|"Other"
    chunk_index: int          # 0-based within filing
    filing_date: date
    accession_number: str
    source_url: str

    def to_json(self) -> str:
        d = self.__dict__.copy()
        d['filing_date'] = d['filing_date'].isoformat()
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        d = json.loads(json_str)
        d['filing_date'] = date.fromisoformat(d['filing_date'])
        return cls(**d)

PATTERNS = {
    "MD&A": re.compile(r"(?i)(item\s+2\.?\s*management.{0,30}discussion)"),
    "Risk Factors": re.compile(r"(?i)(item\s+1a\.?\s*risk\s+factors)"),
    "Forward Guidance": re.compile(r"(?i)(forward[- ]looking\s+statements?|outlook|guidance)"),
    "Financial Statements": re.compile(r"(?i)(item\s+[18]\.?\s*(financial\s+statements|quantitative))")
}

def extract_sections(text: str) -> List[Tuple[str, str]]:
    matches = []
    for section_type, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append((m.start(), section_type))
    
    matches.sort(key=lambda x: x[0])
    
    sections = []
    last_pos = 0
    last_type = "Other"
    
    for pos, sec_type in matches:
        if pos > last_pos:
            sections.append((last_type, text[last_pos:pos]))
        last_pos = pos
        last_type = sec_type
        
    if last_pos < len(text):
        sections.append((last_type, text[last_pos:]))
        
    return sections

def chunk_text_by_tokens(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    # Word-based splitting as a proxy for tokens for speed and simplicity without external deps
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += (chunk_size - overlap)
        if chunk_size <= overlap:
            i += 1 # prevent infinite loop
    return chunks

def chunk_filing(text: str, filing_ref: FilingRef) -> List[Chunk]:
    sections = extract_sections(text)
    chunks = []
    chunk_index = 0
    
    for sec_type, sec_text in sections:
        sec_text_clean = sec_text.strip()
        if not sec_text_clean:
            continue
            
        text_chunks = chunk_text_by_tokens(sec_text_clean, chunk_size=1000, overlap=200)
        for tc in text_chunks:
            chunks.append(Chunk(
                text=tc,
                ticker=filing_ref.ticker,
                filing_type=filing_ref.filing_type,
                quarter=filing_ref.quarter,
                fiscal_year=filing_ref.fiscal_year,
                section_type=sec_type,
                chunk_index=chunk_index,
                filing_date=filing_ref.filing_date,
                accession_number=filing_ref.accession_number,
                source_url=filing_ref.source_url
            ))
            chunk_index += 1
            
    return chunks
