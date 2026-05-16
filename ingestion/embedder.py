from sentence_transformers import SentenceTransformer
from typing import List

_model = None

def _get_model() -> SentenceTransformer:
    """Lazily load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_query(text: str) -> List[float]:
    """
    Embeds a single query string for retrieval.
    Used by retriever before vector search.
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def embed_chunks(texts: List[str]) -> List[List[float]]:
    """
    Embeds a batch of strings during ingestion.
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
