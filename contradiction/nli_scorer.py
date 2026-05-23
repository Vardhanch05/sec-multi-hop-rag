import logging
import random
from typing import List
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from sentence_transformers import CrossEncoder

from config.settings import CONTRADICTION_THRESHOLD, NLI_TIMEOUT_SECONDS, MAX_NLI_PAIRS
from retrieval.claim_extractor import Claim
from contradiction.contradiction_report import ContradictionEvent, ContradictionReport
from db.queries import insert_contradiction_event

logger = logging.getLogger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        # Load the NLI model. Note: this takes time on first load.
        _model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    return _model

def is_cross_period(c1: Claim, c2: Claim) -> bool:
    """Returns True if claims are from different periods."""
    return (c1.fiscal_year, c1.quarter) != (c2.fiscal_year, c2.quarter)

def _process_claims(claims: List[Claim]) -> List[ContradictionEvent]:
    if not claims:
        return []

    # 1. Filter cross-period pairs only
    pairs = []
    for i in range(len(claims)):
        for j in range(len(claims)):
            if i != j and is_cross_period(claims[i], claims[j]):
                pairs.append((claims[i], claims[j]))

    if not pairs:
        return []

    # 2. Cap at MAX_NLI_PAIRS via random sampling
    if len(pairs) > MAX_NLI_PAIRS:
        pairs = random.sample(pairs, MAX_NLI_PAIRS)

    # 3. Construct directed premise/hypothesis pairs
    text_pairs = [[p[0].claim_text, p[1].claim_text] for p in pairs]

    # 4. Score with cross-encoder/nli-deberta-v3-base
    model = get_model()
    # The DeBERTa-v3-base NLI model predicts classes: 0: contradiction, 1: entailment, 2: neutral
    scores = model.predict(text_pairs, apply_softmax=True)
    
    contradictions = []
    query_id = str(uuid.uuid4())
    
    for i, pair in enumerate(pairs):
        c1, c2 = pair
        # 5. Apply CONTRADICTION_THRESHOLD
        contradiction_prob = float(scores[i][0])
        
        if contradiction_prob >= CONTRADICTION_THRESHOLD:
            event = ContradictionEvent(
                ticker=c1.ticker,
                filing_ref_a=c1.accession_number,
                filing_ref_b=c2.accession_number,
                claim_a=c1.claim_text,
                claim_b=c2.claim_text,
                confidence_score=contradiction_prob,
                query_id=query_id
            )
            contradictions.append(event)
            # 18.2 Persist to database
            try:
                insert_contradiction_event(event)
            except Exception as e:
                logger.error(f"Failed to persist contradiction event: {e}")

    return contradictions

def score_contradictions(claims: List[Claim], timeout_seconds: float = NLI_TIMEOUT_SECONDS) -> ContradictionReport:
    """
    Scorer pipeline that runs in a ThreadPoolExecutor to enforce a strict timeout.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_claims, claims)
        try:
            contradictions = future.result(timeout=timeout_seconds)
            return ContradictionReport(contradictions=contradictions, timed_out=False)
        except TimeoutError:
            logger.warning(f"NLI scoring timed out after {timeout_seconds} seconds")
            return ContradictionReport(contradictions=[], timed_out=True)
