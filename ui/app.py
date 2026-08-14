"""
ui/app.py
---------
Streamlit UI for SEC Multi-Hop RAG system.
"""

import streamlit as st
import time
import pandas as pd
import sys
import os

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.queries import get_corpus_stats, get_all_tickers, get_ragas_results
from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize_stream
from ingestion.embedder import embed_query
from retrieval.semantic_cache import get_semantic_cache

st.set_page_config(page_title="SEC RAG System", layout="wide")

# Cached database query calls to avoid hitting SQLite on every Streamlit rerun
@st.cache_data(ttl=600)
def cached_get_corpus_stats():
    return get_corpus_stats()

@st.cache_data(ttl=600)
def cached_get_all_tickers():
    return get_all_tickers()

def contradiction_card_color(score: float) -> str | None:
    if score >= 0.90:
        return "#FF4444"
    elif score >= 0.75:
        return "#FFA500"
    return None

def check_for_legacy_calendar_data() -> bool:
    """Checks if there are any 10-Q filings labeled 'Q4'. In fiscal reporting, Q4 is the 10-K. 
    A 'Q4' 10-Q means the data was ingested using legacy calendar mapping."""
    from db.connection import get_connection
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM filings WHERE filing_type='10-Q' AND quarter='Q4'")
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False

def render_assistant_details(
    contradictions,
    citations,
    latency_ms,
    model_used,
    contradiction_detection_skipped
):
    if contradiction_detection_skipped:
        st.info("Notice: Contradiction detection was skipped due to NLI timeout.")
        
    if contradictions:
        st.write("### Contradictions Detected in Filings")
        for i, c in enumerate(contradictions, 1):
            is_dict = isinstance(c, dict)
            confidence_score = c["confidence_score"] if is_dict else c.confidence_score
            filing_ref_a = c["filing_ref_a"] if is_dict else c.filing_ref_a
            claim_a = c["claim_a"] if is_dict else c.claim_a
            filing_ref_b = c["filing_ref_b"] if is_dict else c.filing_ref_b
            claim_b = c["claim_b"] if is_dict else c.claim_b
            
            color = contradiction_card_color(confidence_score)
            if color:
                st.markdown(
                    f"""
                    <div style="background-color: {color}; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: white;">
                        <h4>Contradiction {i} (Confidence: {confidence_score:.2f})</h4>
                        <div style="display: flex; gap: 20px;">
                            <div style="flex: 1;">
                                <strong>{filing_ref_a}</strong><br/>
                                {claim_a}
                            </div>
                            <div style="flex: 1;">
                                <strong>{filing_ref_b}</strong><br/>
                                {claim_b}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    if citations:
        st.write("### Citations")
        cols = st.columns(4)
        for i, cit in enumerate(citations):
            is_dict = isinstance(cit, dict)
            ticker = cit["ticker"] if is_dict else cit.ticker
            fiscal_year = cit["fiscal_year"] if is_dict else cit.fiscal_year
            filing_type = cit["filing_type"] if is_dict else cit.filing_type
            section = cit["section"] if is_dict else cit.section
            
            with cols[i % 4]:
                st.caption(f"🛡️ {ticker} {fiscal_year} {filing_type} ({section})")
                
    latency_str = f"{latency_ms}ms" if latency_ms is not None else "0ms (Cached)"
    st.caption(f"Latency: {latency_str} | Model: {model_used}")

def main():
    st.title("SEC Multi-Hop RAG Analyst")
    
    if check_for_legacy_calendar_data():
        st.warning(
            "⚠️ **Legacy Data Detected!** Your database contains filings indexed by Calendar Period. "
            "The system has been updated to standardize storage on Fiscal Period natively. "
            "Please purge your `sec_rag.db` and Chroma collections and re-ingest the data to avoid retrieval errors."
        )
    
    # --- Sidebar ---
    st.sidebar.header("Corpus Status")
    stats = cached_get_corpus_stats()
    st.sidebar.metric("Total Filings", stats["total_filings"])
    st.sidebar.metric("Ticker Coverage", stats["unique_tickers"])
    
    st.sidebar.header("Filters")
    all_tickers = cached_get_all_tickers()
    selected_tickers = st.sidebar.multiselect("Select Tickers", options=all_tickers)
    
    tab1, tab2 = st.tabs(["Query Interface", "RAGAS Dashboard"])
    
    with tab1:
        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        chat_container = st.container()
        
        # --- Main Panel ---
        query = st.chat_input("Ask a question about SEC filings...")
        
        with chat_container:
            # Render previous chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant":
                        render_assistant_details(
                            contradictions=message.get("contradictions"),
                            citations=message.get("citations"),
                            latency_ms=message.get("latency_ms"),
                            model_used=message.get("model_used"),
                            contradiction_detection_skipped=message.get("contradiction_detection_skipped", False)
                        )
                        
            if query:
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.chat_message("user").write(query)
                
                with st.chat_message("assistant"):
                    status_placeholder = st.empty()
                    status_placeholder.info("Embedding query and checking semantic cache...")
                
                # Check semantic cache first
                q_emb = embed_query(query)
                cache = get_semantic_cache()
                cached_payload = cache.get(query, q_emb)
                
                if cached_payload:
                    status_placeholder.success("Semantic Cache Hit!")
                    time.sleep(0.5) # subtle visual feedback
                    status_placeholder.empty()
                    
                    st.write(cached_payload.answer)
                    
                    # Convert to serializable format for storage
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
                    
                    render_assistant_details(
                        contradictions=contradictions_serializable,
                        citations=citations_serializable,
                        latency_ms=None,
                        model_used=cached_payload.model_used,
                        contradiction_detection_skipped=cached_payload.contradiction_detection_skipped
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": cached_payload.answer,
                        "contradictions": contradictions_serializable,
                        "citations": citations_serializable,
                        "latency_ms": None,
                        "model_used": cached_payload.model_used,
                        "contradiction_detection_skipped": cached_payload.contradiction_detection_skipped
                    })
                    
                else:
                    # Cache miss: Run RAG Pipeline steps
                    try:
                        # 1. Classification
                        status_placeholder.info("🔄 Classifying user query...")
                        ui_filters = UIFilters(tickers=selected_tickers if selected_tickers else None)
                        hop_plan = classify_query(query, ui_filters)
                        
                        # 2. Planning
                        status_placeholder.info("🔄 Planning retrieval hops...")
                        available_periods = []
                        tickers_to_fetch = hop_plan.tickers if hop_plan.tickers else selected_tickers
                        if not tickers_to_fetch and all_tickers:
                            tickers_to_fetch = all_tickers
                        for t in tickers_to_fetch:
                            available_periods.extend(get_available_periods(t))
                        hop_specs = plan_hops(hop_plan, available_periods)
                        
                        # 3. Retrieval
                        status_placeholder.info("🔄 Retrieving documents from Vector Database...")
                        hop_results = retrieve_hops(query, hop_specs)
                        all_chunks = []
                        for chunks in hop_results.values():
                            all_chunks.extend(chunks)
                            
                        if not all_chunks:
                            status_placeholder.error("No relevant documents found in Vector DB.")
                        else:
                            # 4. Extraction
                            status_placeholder.info("🔄 Extracting key claims via LLM...")
                            claims = extract_claims(query, all_chunks)
                            
                            if not claims:
                                status_placeholder.error("No claims extracted from documents.")
                            else:
                                # 5. Contradiction Scoring
                                if hop_plan.requires_contradiction_check:
                                    status_placeholder.info("🔄 Running local NLI model to check contradictions...")
                                    report = score_contradictions(claims)
                                else:
                                    from contradiction.contradiction_report import ContradictionReport
                                    report = ContradictionReport(contradictions=[], timed_out=False)
                                
                                # 6. Synthesis (Streamed response)
                                status_placeholder.info("🔄 Generating synthesis report...")
                                
                                metadata_container = {}
                                def stream_wrapper():
                                    for item in synthesize_stream(query, claims, report):
                                        if isinstance(item, str):
                                            yield item
                                        else:
                                            metadata_container["payload"] = item
                                
                                status_placeholder.empty()
                                full_answer = st.write_stream(stream_wrapper())
                                
                                # Process metadata once generation finishes
                                payload = metadata_container.get("payload")
                                if payload:
                                    # Save to semantic cache
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
                                    
                                    render_assistant_details(
                                        contradictions=contradictions_serializable,
                                        citations=citations_serializable,
                                        latency_ms=payload.latency_ms,
                                        model_used=payload.model_used,
                                        contradiction_detection_skipped=payload.contradiction_detection_skipped
                                    )
                                    
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": full_answer,
                                        "contradictions": contradictions_serializable,
                                        "citations": citations_serializable,
                                        "latency_ms": payload.latency_ms,
                                        "model_used": payload.model_used,
                                        "contradiction_detection_skipped": payload.contradiction_detection_skipped
                                    })
                                    
                    except Exception as e:
                        status_placeholder.error(f"Error during query execution: {e}")

    with tab2:
        st.header("Evaluation Metrics")
        results = get_ragas_results()
        
        if not results:
            st.info("No RAGAS evaluation results found in the database. Run the evaluation harness to see metrics.")
        else:
            latest = results[-1]
            mock_label = " 🧪 (MOCK)" if latest.get('is_mock') else " 🟢 (REAL)"
            st.write(f"**Latest Evaluation:** {latest['run_timestamp']}{mock_label}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness", f"{latest['faithfulness']:.2f}")
            c2.metric("Answer Relevance", f"{latest['answer_relevance']:.2f}")
            c3.metric("Context Precision", f"{latest['context_precision']:.2f}")
            c4.metric("Context Recall", f"{latest['context_recall']:.2f}")
            
            st.write("### Trend Over Time")
            df = pd.DataFrame(results)
            # Parse datetime string to ensure proper plotting
            df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
            df.set_index('run_timestamp', inplace=True)
            
            # Select only the metrics for the chart
            chart_data = df[['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']]
            st.line_chart(chart_data)
            
            st.write("### Evaluation History")
            
            def highlight_mock(row):
                color = 'background-color: rgba(255, 0, 0, 0.1)' if row.get('is_mock', False) else ''
                return [color] * len(row)
                
            st.dataframe(df.style.apply(highlight_mock, axis=1))

if __name__ == "__main__":
    main()
