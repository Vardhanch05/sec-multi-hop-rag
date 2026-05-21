"""
retrieval/retriever.py
----------------------
Executes parallel vector similarity searches using concurrent thread workers.
"""

import concurrent.futures
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any, Optional

from retrieval.hop_planner import HopSpec
from ingestion.embedder import embed_query
from ingestion.vector_store import get_vector_store

@dataclass(frozen=True)
class ChunkResult:
    text: str
    ticker: str
    filing_type: str
    quarter: Optional[str]
    fiscal_year: int
    section_type: str
    chunk_index: int
    filing_date: date
    accession_number: str
    source_url: str
    score: float

def doc_to_chunk_result(doc: Dict[str, Any]) -> ChunkResult:
    """Converts a raw vector database document search result dictionary back into a ChunkResult."""
    q = doc.get("quarter")
    if q == "":
        q = None
        
    fd = doc.get("filing_date")
    if isinstance(fd, str):
        fd = date.fromisoformat(fd)
        
    return ChunkResult(
        text=doc.get("text", ""),
        ticker=doc.get("ticker", ""),
        filing_type=doc.get("filing_type", ""),
        quarter=q,
        fiscal_year=int(doc.get("fiscal_year", 0)),
        section_type=doc.get("section_type", ""),
        chunk_index=int(doc.get("chunk_index", 0)),
        filing_date=fd,
        accession_number=doc.get("accession_number", ""),
        source_url=doc.get("source_url", ""),
        score=float(doc.get("score", 0.0))
    )

def retrieve_hops(
    query: str,
    hop_specs: List[HopSpec],
    top_k_per_hop: int = 5
) -> Dict[HopSpec, List[ChunkResult]]:
    """
    Embeds the user query once, then executes parallel metadata-filtered vector
    searches across the specified HopSpecs, returning similarity search results
    keyed by their HopSpec.
    """
    if not hop_specs:
        return {}

    # Call embedder EXACTLY once
    query_embedding = embed_query(query)

    store = get_vector_store()
    results = {}

    def retrieve_single_hop(hop_spec: HopSpec) -> List[ChunkResult]:
        # Normalise None quarter to empty string to match ingestion
        q_val = hop_spec.quarter
        if q_val is None:
            q_val = ""
            
        filters = {
            "ticker": hop_spec.ticker,
            "fiscal_year": hop_spec.fiscal_year,
            "filing_type": hop_spec.filing_type,
            "quarter": q_val
        }
        if hop_spec.section_type:
            filters["section_type"] = hop_spec.section_type

        raw_results = store.search(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k_per_hop
        )
        return [doc_to_chunk_result(r) for r in raw_results]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_hop = {executor.submit(retrieve_single_hop, spec): spec for spec in hop_specs}
        for future in concurrent.futures.as_completed(future_to_hop):
            spec = future_to_hop[future]
            try:
                results[spec] = future.result()
            except Exception as exc:
                raise exc

    return results
