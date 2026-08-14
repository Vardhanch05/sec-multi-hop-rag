"""
ui/api.py
---------
FastAPI backend for SEC Multi-Hop RAG system.
"""

import sys
import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load env
load_dotenv()

from db.queries import get_corpus_stats, get_all_tickers, get_ragas_results
from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize_stream
from ingestion.embedder import embed_query
from retrieval.semantic_cache import get_semantic_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_rag_api")

app = FastAPI(title="SEC RAG System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageParam(BaseModel):
    role: str
    content: str
    imageData: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[MessageParam]
    tickers: Optional[List[str]] = None

@app.get("/api/stats")
def read_stats():
    try:
        stats = get_corpus_stats()
        return {
            "totalFilings": stats.get("total_filings", 0),
            "uniqueTickers": stats.get("unique_tickers", 0)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickers")
def read_tickers():
    try:
        tickers = get_all_tickers()
        return tickers
    except Exception as e:
        logger.error(f"Error getting tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ragas")
def read_ragas():
    try:
        results = get_ragas_results()
        formatted_results = []
        for r in results:
            ts = r.get("run_timestamp", "")
            date_str = ts
            if ts and len(ts) >= 10:
                parts = ts.split(" ")[0].split("-")
                if len(parts) == 3:
                    date_str = f"{parts[1]}/{parts[2]}"
            formatted_results.append({
                "date": date_str,
                "faithfulness": r.get("faithfulness", 0.0),
                "relevance": r.get("answer_relevance", 0.0),
                "precision": r.get("context_precision", 0.0),
                "recall": r.get("context_recall", 0.0),
                "is_mock": r.get("is_mock", False),
                "timestamp": ts
            })
        return formatted_results
    except Exception as e:
        logger.error(f"Error getting ragas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    last_msg = request.messages[-1]
    query = last_msg.content
    selected_tickers = request.tickers
    
    async def event_generator():
        try:
            # 1. Embed query
            q_emb = embed_query(query)
            cache = get_semantic_cache()
            cached_payload = cache.get(query, q_emb)
            
            if cached_payload:
                yield json.dumps({"type": "text", "content": cached_payload.answer}) + "\n"
                
                citations_serializable = [
                    {
                        "ticker": cit.ticker,
                        "fiscal_year": cit.fiscal_year,
                        "filing_type": cit.filing_type,
                        "section": cit.section,
                        "accession_number": cit.accession_number
                    }
                    for cit in cached_payload.citations
                ]
                contradictions_serializable = [
                    {
                        "confidence_score": c.confidence_score,
                        "filing_ref_a": c.filing_ref_a,
                        "claim_a": c.claim_a,
                        "filing_ref_b": c.filing_ref_b,
                        "claim_b": c.claim_b
                    }
                    for c in cached_payload.contradictions
                ]
                
                payload = {
                    "citations": citations_serializable,
                    "contradictions": contradictions_serializable,
                    "latency_ms": None,
                    "model_used": cached_payload.model_used,
                    "contradiction_detection_skipped": cached_payload.contradiction_detection_skipped
                }
                yield json.dumps({"type": "metadata", "payload": payload}) + "\n"
                return
            
            # Cache miss: Run RAG pipeline
            ui_filters = UIFilters(tickers=selected_tickers if selected_tickers else None)
            hop_plan = classify_query(query, ui_filters)
            
            # Get available periods
            all_tickers = get_all_tickers()
            tickers_to_fetch = hop_plan.tickers if hop_plan.tickers else selected_tickers
            if not tickers_to_fetch and all_tickers:
                tickers_to_fetch = all_tickers
            
            available_periods = []
            for t in tickers_to_fetch or []:
                available_periods.extend(get_available_periods(t))
                
            hop_specs = plan_hops(hop_plan, available_periods)
            # Use wider retrieval for global (no-section) queries like litigation/regulatory
            top_k = 5 if hop_plan.section_hint is None else 3
            hop_results = retrieve_hops(query, hop_specs, top_k_per_hop=top_k)
            
            all_chunks = []
            for chunks in hop_results.values():
                all_chunks.extend(chunks)
                
            if not all_chunks:
                yield json.dumps({"type": "text", "content": "No relevant documents found in Vector DB."}) + "\n"
                return
            
            claims = extract_claims(query, all_chunks)
            if not claims:
                yield json.dumps({"type": "text", "content": "No claims extracted from documents."}) + "\n"
                return
                
            if hop_plan.requires_contradiction_check:
                report = score_contradictions(claims)
            else:
                from contradiction.contradiction_report import ContradictionReport
                report = ContradictionReport(contradictions=[], timed_out=False)
            
            # Stream the synthesis
            metadata_container = {}
            for item in synthesize_stream(query, claims, report):
                if isinstance(item, str):
                    yield json.dumps({"type": "text", "content": item}) + "\n"
                else:
                    metadata_container["payload"] = item
            
            # Yield metadata at the end
            payload = metadata_container.get("payload")
            if payload:
                # Save to cache
                cache.set(query, q_emb, payload)
                
                citations_serializable = [
                    {
                        "ticker": cit.ticker,
                        "fiscal_year": cit.fiscal_year,
                        "filing_type": cit.filing_type,
                        "section": cit.section,
                        "accession_number": cit.accession_number
                    }
                    for cit in payload.citations
                ]
                contradictions_serializable = [
                    {
                        "confidence_score": c.confidence_score,
                        "filing_ref_a": c.filing_ref_a,
                        "claim_a": c.claim_a,
                        "filing_ref_b": c.filing_ref_b,
                        "claim_b": c.claim_b
                    }
                    for c in payload.contradictions
                ]
                
                payload_data = {
                    "citations": citations_serializable,
                    "contradictions": contradictions_serializable,
                    "latency_ms": payload.latency_ms,
                    "model_used": payload.model_used,
                    "contradiction_detection_skipped": payload.contradiction_detection_skipped
                }
                yield json.dumps({"type": "metadata", "payload": payload_data}) + "\n"
        except Exception as e:
            logger.error(f"Error in stream: {e}", exc_info=True)
            yield json.dumps({"type": "text", "content": f"\n[Error during query execution: {str(e)}]"}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("ui.api:app", host="127.0.0.1", port=port, reload=True)

