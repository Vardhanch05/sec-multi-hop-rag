import pytest
from hypothesis import given, strategies as st
from ingestion.embedder import embed_query, embed_chunks

def test_embed_query_dimension():
    text = "This is a sample financial sentence."
    embedding = embed_query(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

@given(texts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
def test_embed_chunks_dimension_and_batch_size(texts):
    embeddings = embed_chunks(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 384
        assert all(isinstance(x, float) for x in emb)

def test_embed_chunks_empty_list():
    assert embed_chunks([]) == []

def test_embed_query_determinism():
    text = "Company ABC reported a Q1 revenue of $10M."
    emb1 = embed_query(text)
    emb2 = embed_query(text)
    assert emb1 == emb2
