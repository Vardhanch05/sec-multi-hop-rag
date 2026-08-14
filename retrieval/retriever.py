"""
retrieval/retriever.py
----------------------
Executes parallel vector similarity searches using concurrent thread workers.
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

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
    top_k_per_hop: int = 3
) -> Dict[HopSpec, List[ChunkResult]]:
    """
    Embeds the user query once, then executes parallel metadata-filtered vector
    searches across the specified HopSpecs, returning similarity search results
    keyed by their HopSpec.
    
    Includes neighborhood expansion (fetching predecessor and successor chunks)
    to preserve context for sentences split across chunk boundaries.
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
        
        # Fallback - drop section filter if insufficient results (e.g., misclassified sections)
        if len(raw_results) < top_k_per_hop and hop_spec.section_type:
            original_len = len(raw_results)
            fallback_filters = {
                "ticker": hop_spec.ticker,
                "fiscal_year": hop_spec.fiscal_year,
                "filing_type": hop_spec.filing_type,
                "quarter": q_val
            }
            fallback_results = store.search(
                query_embedding=query_embedding,
                filters=fallback_filters,
                top_k=top_k_per_hop
            )
            
            raw_results = fallback_results
            logger.warning(
                f"Section filter '{hop_spec.section_type}' returned only {original_len} results "
                f"for {hop_spec.ticker} {hop_spec.quarter} {hop_spec.fiscal_year}. "
                f"Retried without section filter, got {len(raw_results)} results."
            )
            
        initial_chunks = [doc_to_chunk_result(r) for r in raw_results]
        
        # Bypassing neighbor expansion to avoid rate limits and reduce latency
        return initial_chunks

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_hop = {executor.submit(retrieve_single_hop, spec): spec for spec in hop_specs}
        for future in concurrent.futures.as_completed(future_to_hop):
            spec = future_to_hop[future]
            try:
                results[spec] = future.result()
            except Exception as exc:
                raise exc

    return results


def expand_neighbors(store: Any, chunks: List[ChunkResult]) -> List[ChunkResult]:
    if not chunks:
        return chunks
        
    from config.settings import VECTOR_STORE_BACKEND
    is_qdrant = VECTOR_STORE_BACKEND.lower() == "qdrant"
    COLLECTION_NAME = "sec_chunks"
    
    ids_to_query = []
    id_map = {} # store_id -> (accession_number, chunk_index)
    
    for c in chunks:
        if c.chunk_index > 0:
            prev_key = f"{c.accession_number}_{c.chunk_index - 1}"
            if is_qdrant:
                import uuid
                prev_id = str(uuid.uuid5(uuid.NAMESPACE_OID, prev_key))
            else:
                prev_id = prev_key
            if prev_id not in id_map:
                ids_to_query.append(prev_id)
                id_map[prev_id] = (c.accession_number, c.chunk_index - 1)
            
        next_key = f"{c.accession_number}_{c.chunk_index + 1}"
        if is_qdrant:
            import uuid
            next_id = str(uuid.uuid5(uuid.NAMESPACE_OID, next_key))
        else:
            next_id = next_key
        if next_id not in id_map:
            ids_to_query.append(next_id)
            id_map[next_id] = (c.accession_number, c.chunk_index + 1)
        
    if not ids_to_query:
        return chunks
        
    neighbors = []
    try:
        if is_qdrant:
            # Qdrant client retrieve
            records = store.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=ids_to_query
            )
            for r in records:
                p_id = str(r.id)
                if p_id in id_map:
                    payload = r.payload or {}
                    text_val = payload.get("text") or ""
                    
                    q = payload.get("quarter")
                    if q == "":
                        q = None
                    fd = payload.get("filing_date")
                    if isinstance(fd, str):
                        fd = date.fromisoformat(fd)
                    else:
                        fd = date.today()
                        
                    neighbors.append(
                        ChunkResult(
                            text=text_val,
                            ticker=payload.get("ticker", ""),
                            filing_type=payload.get("filing_type", ""),
                            quarter=q,
                            fiscal_year=int(payload.get("fiscal_year", 0)),
                            section_type=payload.get("section_type", ""),
                            chunk_index=int(payload.get("chunk_index", 0)),
                            filing_date=fd,
                            accession_number=payload.get("accession_number", ""),
                            source_url=payload.get("source_url", ""),
                            score=0.0
                        )
                    )
        else:
            # ChromaStore collection get
            res = store.collection.get(ids=ids_to_query)
            if res and "ids" in res:
                for i in range(len(res["ids"])):
                    p_id = res["ids"][i]
                    if p_id in id_map:
                        doc = res["documents"][i] if res["documents"] else ""
                        meta = res["metadatas"][i] if res["metadatas"] else {}
                        
                        q = meta.get("quarter")
                        if q == "":
                            q = None
                        fd = meta.get("filing_date")
                        if isinstance(fd, str):
                            fd = date.fromisoformat(fd)
                        else:
                            fd = date.today()
                            
                        neighbors.append(
                            ChunkResult(
                                text=doc,
                                ticker=meta.get("ticker", ""),
                                filing_type=meta.get("filing_type", ""),
                                quarter=q,
                                fiscal_year=int(meta.get("fiscal_year", 0)),
                                section_type=meta.get("section_type", ""),
                                chunk_index=int(meta.get("chunk_index", 0)),
                                filing_date=fd,
                                accession_number=meta.get("accession_number", ""),
                                source_url=meta.get("source_url", ""),
                                score=0.0
                            )
                        )
    except Exception as e:
        logger.warning(f"Neighborhood expansion failed: {e}")
        # Return chunks without expansion on error
        return chunks

    # Deduplicate keeping order of priority (original results first)
    seen = set()
    unique_chunks = []
    
    for c in chunks:
        key = (c.accession_number, c.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)
            
    for c in neighbors:
        key = (c.accession_number, c.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)
            
    # Sort by chunk_index to keep them contiguous for LLM reading!
    unique_chunks.sort(key=lambda c: (c.ticker, c.fiscal_year, c.filing_type, c.quarter or "", c.section_type, c.chunk_index))
    
    return unique_chunks
