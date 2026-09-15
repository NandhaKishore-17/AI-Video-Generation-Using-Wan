"""
autonomous/evidence_store.py - Local Qdrant Vector Store & Provenance-Preserving Retrieval.

Integrates with the existing Qdrant instance at backend/data/qdrant_knowledge using an
additive collection ('autonomous_research_evidence') and the local 'all-MiniLM-L6-v2'
model running strictly on CPU to protect GPU VRAM. Enforces deterministic UUID5 point IDs.
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from autonomous.research_processor import EvidenceChunk

logger = logging.getLogger("autonomous.evidence_store")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except ImportError:
    QdrantClient = None
    PointStruct = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


RESEARCH_COLLECTION = "autonomous_research_evidence"
EMBEDDING_DIM = 384


_SHARED_QDRANT_CLIENTS: Dict[str, Any] = {}
_SHARED_EMBEDDING_MODELS: Dict[str, Any] = {}


class EvidenceStore:
    """Manages indexing and semantic retrieval of research evidence chunks in Qdrant."""

    def __init__(
        self,
        qdrant_path: Optional[str] = None,
        collection_name: str = RESEARCH_COLLECTION,
        device: str = "cpu"
    ):
        # Default to existing backend/data/qdrant_knowledge path
        if qdrant_path is None:
            backend_dir = Path("backend")
            if backend_dir.exists():
                qdrant_path = str((backend_dir / "data" / "qdrant_knowledge").resolve())
            else:
                qdrant_path = str(Path("data/qdrant_knowledge").resolve())

        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.device = device
        self._qdrant_client = None
        self._embedding_model = None

    @property
    def qdrant(self) -> Optional[Any]:
        """Lazy-initialize Qdrant client and ensure collection exists."""
        if self._qdrant_client is None and QdrantClient is not None:
            norm_path = str(Path(self.qdrant_path).resolve())
            if norm_path in _SHARED_QDRANT_CLIENTS:
                self._qdrant_client = _SHARED_QDRANT_CLIENTS[norm_path]
            else:
                try:
                    os.makedirs(self.qdrant_path, exist_ok=True)
                    self._qdrant_client = QdrantClient(path=self.qdrant_path)
                    _SHARED_QDRANT_CLIENTS[norm_path] = self._qdrant_client
                    # Ensure additive collection exists
                    collections = [c.name for c in self._qdrant_client.get_collections().collections]
                    if self.collection_name not in collections:
                        self._qdrant_client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                        )
                        logger.info(f"Created additive Qdrant collection: '{self.collection_name}' in {self.qdrant_path}")
                except Exception as e:
                    logger.error(f"Failed to initialize Qdrant client at {self.qdrant_path}: {e}")
                    self._qdrant_client = None
        return self._qdrant_client

    @property
    def embedding_model(self) -> Optional[Any]:
        """Lazy-initialize SentenceTransformer model strictly on CPU."""
        if self._embedding_model is None and SentenceTransformer is not None:
            cache_key = f"all-MiniLM-L6-v2_{self.device}"
            if cache_key in _SHARED_EMBEDDING_MODELS:
                self._embedding_model = _SHARED_EMBEDDING_MODELS[cache_key]
            else:
                try:
                    logger.info(f"Loading embedding model 'all-MiniLM-L6-v2' on device: {self.device}")
                    self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                    _SHARED_EMBEDDING_MODELS[cache_key] = self._embedding_model
                except Exception as e:
                    logger.error(f"Failed to load sentence-transformers model: {e}")
                    self._embedding_model = None
        return self._embedding_model

    def index_chunks(
        self,
        chunks: List[EvidenceChunk],
        topic: str,
        episode_id: str
    ) -> Dict[str, Any]:
        """
        Embed and upsert evidence chunks into Qdrant.
        Enforces deterministic UUID5 IDs for strict idempotency.
        """
        if not chunks:
            return {"indexed_count": 0, "embedding_time_s": 0.0, "qdrant_time_s": 0.0}

        t0 = time.time()
        texts = [c.text for c in chunks]

        # 1. Generate Embeddings (on CPU)
        if self.embedding_model:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                device=self.device,
                normalize_embeddings=True
            )
        else:
            # Fallback mock embedding if library unavailable
            embeddings = [[0.0] * EMBEDDING_DIM for _ in texts]

        t_embed = time.time() - t0

        # 2. Build PointStruct with complete provenance payload
        t1 = time.time()
        points = []
        for chunk, emb in zip(chunks, embeddings):
            # Deterministic UUID5 point ID prevents vector duplication (Correction 10)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{episode_id}_{chunk.content_hash}_{chunk.chunk_index}"))

            vec_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

            payload = {
                "source_id": chunk.source_id,
                "title": chunk.title,
                "url": chunk.url,
                "publisher": chunk.publisher,
                "domain": getattr(chunk, "domain", None),
                "author": getattr(chunk, "author", None),
                "source_origin": getattr(chunk, "source_origin", "external"),
                "independence_group": getattr(chunk, "independence_group", "") or chunk.publisher or "",
                "source_role": getattr(chunk, "source_role", "primary_evidence"),
                "evidence_weight_class": getattr(chunk, "evidence_weight_class", "PRIMARY"),
                "source_type": "research_evidence",
                "credibility_tier": chunk.credibility_tier,
                "retrieved_at": chunk.retrieval_timestamp,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "topic": topic,
                "episode_id": episode_id,
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "word_count": chunk.word_count
            }

            if PointStruct:
                points.append(PointStruct(id=point_id, vector=vec_list, payload=payload))

        # 3. Upsert into Qdrant
        indexed_count = 0
        if points and self.qdrant:
            try:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                indexed_count = len(points)
                logger.info(f"Upserted {indexed_count} evidence points into Qdrant collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Failed to upsert points into Qdrant: {e}")

        t_qdrant = time.time() - t1

        return {
            "indexed_count": indexed_count,
            "embedding_time_s": round(t_embed, 3),
            "qdrant_time_s": round(t_qdrant, 3),
            "total_chunks": len(chunks)
        }

    def retrieve_evidence(
        self,
        query: str,
        episode_id: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Semantically retrieve the most relevant evidence chunks for a question.
        Returns scored passages with source provenance.
        """
        if not self.embedding_model or not self.qdrant:
            logger.warning("EvidenceStore offline or Qdrant unavailable. Retrieval returning empty.")
            return []

        try:
            query_vector = self.embedding_model.encode(
                query,
                device=self.device,
                normalize_embeddings=True
            ).tolist()

            # Optional filter by episode_id
            search_filter = None
            if episode_id and Filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="episode_id",
                            match=MatchValue(value=episode_id)
                        )
                    ]
                )

            results = []
            if hasattr(self.qdrant, "query_points"):
                resp = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=top_k
                )
                results = resp.points
            elif hasattr(self.qdrant, "search"):
                results = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=top_k
                )

            evidence_items = []
            for hit in results:
                p = hit.payload or {}
                evidence_items.append({
                    "chunk_id": p.get("chunk_id", ""),
                    "score": round(float(hit.score), 4),
                    "text": p.get("text", ""),
                    "source_id": p.get("source_id", ""),
                    "title": p.get("title", ""),
                    "publisher": p.get("publisher", ""),
                    "url": p.get("url"),
                    "domain": p.get("domain"),
                    "author": p.get("author"),
                    "source_origin": p.get("source_origin", "external"),
                    "independence_group": p.get("independence_group", ""),
                    "source_role": p.get("source_role", "primary_evidence"),
                    "evidence_weight_class": p.get("evidence_weight_class", "PRIMARY"),
                    "credibility_tier": p.get("credibility_tier", 3),
                    "chunk_index": p.get("chunk_index", 0),
                    "content_hash": p.get("content_hash", "")
                })

            return evidence_items

        except Exception as e:
            logger.error(f"Error during evidence retrieval for query '{query}': {e}")
            return []
