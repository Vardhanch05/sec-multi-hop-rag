import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import asdict

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from config.settings import VECTOR_STORE_BACKEND, QDRANT_URL, QDRANT_API_KEY, CHROMA_PERSIST_DIR
from ingestion.section_chunker import Chunk

COLLECTION_NAME = "sec_chunks"

class VectorStore:
    def insert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        raise NotImplementedError
        
    def search(self, query_embedding: List[float], filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

class ChromaStore(VectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        meta = asdict(chunk)
        # remove text from metadata
        del meta['text']
        # handle None and date
        if meta.get('quarter') is None:
            meta['quarter'] = ""
        meta['filing_date'] = meta['filing_date'].isoformat()
        return meta

    def insert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks:
            return
            
        ids = [f"{c.accession_number}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._prepare_metadata(c) for c in chunks]
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    def search(self, query_embedding: List[float], filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        where = {}
        for k, v in filters.items():
            if v is not None:
                where[k] = v
                
        if len(where) > 1:
            where_clause = {"$and": [{k: v} for k, v in where.items()]}
        elif len(where) == 1:
            where_clause = where
        else:
            where_clause = None

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            query_args["where"] = where_clause

        results = self.collection.query(**query_args)
        
        output = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "score": results['distances'][0][i]
                }
                doc.update(results['metadatas'][0][i])
                output.append(doc)
        return output

class QdrantBackendStore(VectorStore):
    def __init__(self):
        # We handle empty QDRANT_URL for testing/mocking environments
        if not QDRANT_URL:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
        collections = self.client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        meta = asdict(chunk)
        if meta.get('quarter') is None:
            meta['quarter'] = ""
        meta['filing_date'] = meta['filing_date'].isoformat()
        return meta

    def insert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks:
            return
            
        points = []
        for c, emb in zip(chunks, embeddings):
            meta = self._prepare_metadata(c)
            # UUID based on accession and chunk index
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{c.accession_number}_{c.chunk_index}"))
            points.append(PointStruct(
                id=point_id,
                vector=emb,
                payload=meta
            ))
            
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
    def search(self, query_embedding: List[float], filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        must_conditions = []
        for k, v in filters.items():
            if v is not None:
                must_conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
                
        filter_obj = Filter(must=must_conditions) if must_conditions else None
        
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=filter_obj,
            limit=top_k
        )
        
        output = []
        for res in results:
            doc = {
                "id": str(res.id),
                "text": res.payload.get("text", ""),
                "score": res.score
            }
            for k, v in res.payload.items():
                if k != "text":
                    doc[k] = v
            output.append(doc)
            
        return output

_store = None

def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        if VECTOR_STORE_BACKEND.lower() == "qdrant":
            _store = QdrantBackendStore()
        else:
            _store = ChromaStore()
    return _store
