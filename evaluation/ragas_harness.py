"""
evaluation/ragas_harness.py
---------------------------
Runner for the RAGAS evaluation benchmark.
"""

from dataclasses import dataclass
import datetime
import logging
import json
import os
import sys

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


from retrieval.query_classifier import classify_query, UIFilters
from retrieval.hop_planner import plan_hops, get_available_periods
from retrieval.retriever import retrieve_hops
from retrieval.claim_extractor import extract_claims
from contradiction.nli_scorer import score_contradictions
from synthesis.answer_synthesizer import synthesize

logger = logging.getLogger(__name__)

@dataclass
class RagasResult:
    run_timestamp: datetime.datetime
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    subset_breakdowns: dict  # Serialized to JSON when saved
    is_mock: bool = False

def parse_txt_file(filepath: str) -> dict:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("##"):
                continue
            parts = line.split(". ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                data[int(parts[0])] = parts[1]
    return data

def run_evaluation_suite():
    logger.info("Parsing questions and ground truths...")
    questions = parse_txt_file("questions_sec.txt")
    ground_truths = parse_txt_file("answer_sec.txt")
    
    # Configure RAGAS LLM and Embeddings using the faster model to avoid strict rate limits
    # The Llama-3.3-70b rate limit is too low for the concurrent `evaluate` calls.
    judge_model = "llama-3.1-8b-instant"
    eval_llm = ChatGroq(model=judge_model, temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    ui_filters = UIFilters()
    
    for q_id, query in questions.items():
        if q_id not in ground_truths:
            continue
            
        logger.info(f"Generating system response for Q{q_id}...")
        try:
            hop_plan = classify_query(query, ui_filters)
            available_periods = []
            tickers = hop_plan.tickers if hop_plan.tickers else []
            for t in tickers:
                available_periods.extend(get_available_periods(t))
                
            hop_specs = plan_hops(hop_plan, available_periods)
            hop_results = retrieve_hops(query, hop_specs)
            
            all_chunks = []
            for chunks in hop_results.values():
                all_chunks.extend(chunks)
                
            if not all_chunks:
                logger.warning(f"No contexts found for Q{q_id}")
                answer_text = "No relevant documents found."
                contexts_text = [""]
            else:
                claims = extract_claims(query, all_chunks)
                report = score_contradictions(claims)
                payload = synthesize(query, claims, report)
                answer_text = payload.answer
                contexts_text = [c.text for c in all_chunks]
                
        except Exception as e:
            logger.error(f"Error on Q{q_id}: {e}")
            answer_text = str(e)
            contexts_text = [""]
            
        data_dict["question"].append(query)
        data_dict["answer"].append(answer_text)
        data_dict["contexts"].append(contexts_text)
        data_dict["ground_truth"].append(ground_truths[q_id])
        
    dataset = Dataset.from_dict(data_dict)
    
    logger.info(f"Starting RAGAS evaluation with {judge_model}...")
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=RunConfig(max_workers=2)
        )
    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}")
        return

    scores = result if isinstance(result, dict) else result.scores if hasattr(result, 'scores') else {}
    
    faithfulness_score = result.get("faithfulness", 0.0) if hasattr(result, 'get') else 0.0
    answer_relevance_score = result.get("answer_relevance", 0.0) if hasattr(result, 'get') else 0.0
    context_precision_score = result.get("context_precision", 0.0) if hasattr(result, 'get') else 0.0
    context_recall_score = result.get("context_recall", 0.0) if hasattr(result, 'get') else 0.0

    rr = RagasResult(
        run_timestamp=datetime.datetime.now(),
        faithfulness=faithfulness_score,
        answer_relevance=answer_relevance_score,
        context_precision=context_precision_score,
        context_recall=context_recall_score,
        subset_breakdowns={
            "judge_llm": judge_model,
            "note": "Evaluated using Groq. Scores may differ from GPT-4 benchmarks.",
            "total_questions": len(data_dict["question"])
        }
    )
    
    from db.queries import write_ragas_result
    write_ragas_result(rr)
    logger.info("Evaluation complete and saved to database.")
    print("\n--- Evaluation Complete ---")
    print(f"Faithfulness: {faithfulness_score}")
    print(f"Answer Relevance: {answer_relevance_score}")
    print(f"Context Precision: {context_precision_score}")
    print(f"Context Recall: {context_recall_score}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_evaluation_suite()
