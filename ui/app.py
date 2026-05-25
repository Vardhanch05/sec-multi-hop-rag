"""
ui/app.py
---------
Streamlit UI for SEC Multi-Hop RAG system.
"""

import streamlit as st
import time

from db.queries import get_corpus_stats, get_all_tickers
from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize

st.set_page_config(page_title="SEC RAG System", layout="wide")

def main():
    st.title("SEC Multi-Hop RAG Analyst")
    
    # --- Sidebar ---
    st.sidebar.header("Corpus Status")
    stats = get_corpus_stats()
    st.sidebar.metric("Total Filings", stats["total_filings"])
    st.sidebar.metric("Ticker Coverage", stats["unique_tickers"])
    
    st.sidebar.header("Filters")
    all_tickers = get_all_tickers()
    selected_tickers = st.sidebar.multiselect("Select Tickers", options=all_tickers)
    
    # --- Main Panel ---
    query = st.chat_input("Ask a question about SEC filings...")
    
    if query:
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            with st.status("Processing query...", expanded=True) as status:
                try:
                    # 1. Classification
                    st.write("Classifying query...")
                    ui_filters = UIFilters(tickers=selected_tickers if selected_tickers else None)
                    hop_plan = classify_query(query, ui_filters)
                    
                    # 2. Planning
                    st.write("Planning retrieval hops...")
                    # Get available periods across all relevant tickers
                    available_periods = []
                    tickers_to_fetch = hop_plan.tickers if hop_plan.tickers else selected_tickers
                    if not tickers_to_fetch and all_tickers:
                        tickers_to_fetch = all_tickers
                        
                    for t in tickers_to_fetch:
                        available_periods.extend(get_available_periods(t))
                        
                    hop_specs = plan_hops(hop_plan, available_periods)
                    
                    # 3. Retrieval
                    st.write("Retrieving relevant chunks...")
                    hop_results = retrieve_hops(query, hop_specs)
                    
                    # Flatten chunks
                    all_chunks = []
                    for chunks in hop_results.values():
                        all_chunks.extend(chunks)
                    
                    if not all_chunks:
                        status.update(label="No relevant documents found.", state="error")
                        st.stop()
                    
                    # 4. Extraction
                    st.write("Extracting claims...")
                    claims = extract_claims(query, all_chunks)
                    
                    if not claims:
                        status.update(label="No claims extracted.", state="error")
                        st.stop()
                    
                    # 5. Contradiction Scoring
                    st.write("Checking for contradictions...")
                    report = score_contradictions(claims)
                    
                    # 6. Synthesis
                    st.write("Synthesizing final answer...")
                    payload = synthesize(query, claims, report)
                    
                    status.update(label="Query processed successfully!", state="complete")
                    
                except Exception as e:
                    status.update(label=f"Error: {e}", state="error")
                    st.stop()
            
            # --- Render Results ---
            st.write(payload.answer)
            
            if payload.contradictions:
                st.warning("⚠️ Contradictions Detected in Filings")
                for i, c in enumerate(payload.contradictions, 1):
                    st.error(f"**Contradiction {i}**\n- {c.claim_a}\n- {c.claim_b}\n*(Confidence: {c.confidence_score:.2f})*")
            
            if payload.citations:
                st.write("### Citations")
                cols = st.columns(4)
                for i, cit in enumerate(payload.citations):
                    with cols[i % 4]:
                        st.caption(f"🛡️ {cit.ticker} {cit.fiscal_year} {cit.filing_type} ({cit.section})")
                        
            st.caption(f"⏱️ Latency: {payload.latency_ms}ms | 🤖 Model: {payload.model_used}")

if __name__ == "__main__":
    main()
