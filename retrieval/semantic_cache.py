import logging
import numpy as np
from typing import List, Dict, Any, Optional
from synthesis.answer_synthesizer import ResponsePayload

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.96, max_size: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cache: List[Dict[str, Any]] = []

    def get(self, query: str, query_embedding: List[float]) -> Optional[ResponsePayload]:
        if not self.cache:
            return None
        
        q_emb = np.array(query_embedding)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return None
            
        best_sim = -1.0
        best_entry = None
        
        for entry in self.cache:
            entry_emb = np.array(entry['embedding'])
            entry_norm = np.linalg.norm(entry_emb)
            if entry_norm == 0:
                continue
            sim = np.dot(q_emb, entry_emb) / (q_norm * entry_norm)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
                
        if best_sim >= self.similarity_threshold and best_entry is not None:
            logger.info(f"Semantic cache hit! Similarity: {best_sim:.4f} for query: '{best_entry['query']}' (requested: '{query}')")
            # Return a copy of payload with updated latency/cache hit info if desired
            return best_entry['response']
            
        return None

    def set(self, query: str, query_embedding: List[float], response: ResponsePayload):
        # Prevent caching errors/overloaded answers
        if "error" in response.answer.lower() or "overloaded" in response.answer.lower():
            return
            
        if len(self.cache) >= self.max_size:
            self.cache.pop(0) # FIFO eviction
            
        self.cache.append({
            'query': query,
            'embedding': query_embedding,
            'response': response
        })
        logger.info(f"Cached response for query: '{query}'")

_global_cache = SemanticCache()

def get_semantic_cache() -> SemanticCache:
    global _global_cache
    return _global_cache
