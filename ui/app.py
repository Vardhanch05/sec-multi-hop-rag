"""
ui/app.py
---------
Streamlit UI for SEC Multi-Hop RAG system.
"""

import streamlit as st
import time
import threading

from db.queries import get_corpus_stats, get_all_tickers
from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize

st.set_page_config(page_title="SEC RAG System", layout="wide")

def contradiction_card_color(score: float) -> str | None:
    if score >= 0.90:
        return "#FF4444"
    elif score >= 0.75:
        return "#FFA500"
    return None

def process_query(query, selected_tickers, all_tickers, result_container):
    try:
        # 1. Classification
        ui_filters = UIFilters(tickers=selected_tickers if selected_tickers else None)
        hop_plan = classify_query(query, ui_filters)
        
        # 2. Planning
        available_periods = []
        tickers_to_fetch = hop_plan.tickers if hop_plan.tickers else selected_tickers
        if not tickers_to_fetch and all_tickers:
            tickers_to_fetch = all_tickers
            
        for t in tickers_to_fetch:
            available_periods.extend(get_available_periods(t))
            
        hop_specs = plan_hops(hop_plan, available_periods)
        
        # 3. Retrieval
        hop_results = retrieve_hops(query, hop_specs)
        
        all_chunks = []
        for chunks in hop_results.values():
            all_chunks.extend(chunks)
        
        if not all_chunks:
            result_container["error"] = "No relevant documents found."
            return
        
        # 4. Extraction
        claims = extract_claims(query, all_chunks)
        if not claims:
            result_container["error"] = "No claims extracted."
            return
        
        # 5. Contradiction Scoring
        report = score_contradictions(claims)
        
        # 6. Synthesis
        payload = synthesize(query, claims, report)
        result_container["payload"] = payload
    except Exception as e:
        result_container["error"] = str(e)


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
            result_container = {}
            t = threading.Thread(target=process_query, args=(query, selected_tickers, all_tickers, result_container))
            t.start()
            
            status_container = st.empty()
            
            # Cold start polling
            start_time = time.time()
            show_loading = False
            
            while t.is_alive():
                elapsed = time.time() - start_time
                if elapsed > 120:
                    status_container.error("Request timed out (120s limit).")
                    return
                elif elapsed > 5:
                    if not show_loading:
                        show_loading = True
                    status_container.info(f"Processing query... (Elapsed: {int(elapsed)}s)")
                
                time.sleep(2)
                
            t.join()
            status_container.empty()
            
            if "error" in result_container:
                st.error(f"Error: {result_container['error']}")
                return
                
            payload = result_container.get("payload")
            if not payload:
                st.error("Unknown error occurred.")
                return
            
            # --- Render Results ---
            st.write(payload.answer)
            
            if payload.contradiction_detection_skipped:
                st.info("Notice: Contradiction detection was skipped due to NLI timeout.")
                
            if payload.contradictions:
                st.write("### Contradictions Detected in Filings")
                for i, c in enumerate(payload.contradictions, 1):
                    color = contradiction_card_color(c.confidence_score)
                    if color:
                        st.markdown(
                            f"""
                            <div style="background-color: {color}; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: white;">
                                <h4>Contradiction {i} (Confidence: {c.confidence_score:.2f})</h4>
                                <div style="display: flex; gap: 20px;">
                                    <div style="flex: 1;">
                                        <strong>{c.filing_ref_a}</strong><br/>
                                        {c.claim_a}
                                    </div>
                                    <div style="flex: 1;">
                                        <strong>{c.filing_ref_b}</strong><br/>
                                        {c.claim_b}
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            if payload.citations:
                st.write("### Citations")
                cols = st.columns(4)
                for i, cit in enumerate(payload.citations):
                    with cols[i % 4]:
                        st.caption(f"🛡️ {cit.ticker} {cit.fiscal_year} {cit.filing_type} ({cit.section})")
                        
            st.caption(f"⏱️ Latency: {payload.latency_ms}ms | 🤖 Model: {payload.model_used}")

if __name__ == "__main__":
    main()
