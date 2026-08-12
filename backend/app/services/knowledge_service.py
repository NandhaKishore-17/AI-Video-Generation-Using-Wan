import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# Optional imports handled gracefully to avoid crashing if not installed yet
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

from app.core.database import SessionLocal
from app.models.domain import KnowledgeDocument, KnowledgeChunk
from app.core.config import settings

logger = logging.getLogger("knowledge_service")

KNOWLEDGE_COLLECTION = "reference_knowledge"
EMBEDDING_DIM = 384  # size for all-MiniLM-L6-v2

class KnowledgeService:
    def __init__(self):
        self.upload_dir = os.path.join(settings.BASE_DIR, "data", "knowledge")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self._embedding_model = None
        self._qdrant_client = None

    @property
    def qdrant(self):
        if self._qdrant_client is None and QdrantClient is not None:
            qdrant_path = os.path.join(settings.BASE_DIR, "data", "qdrant_knowledge")
            os.makedirs(qdrant_path, exist_ok=True)
            self._qdrant_client = QdrantClient(path=qdrant_path)
            
            # Ensure collection exists
            try:
                collections = self._qdrant_client.get_collections().collections
                if not any(c.name == KNOWLEDGE_COLLECTION for c in collections):
                    self._qdrant_client.create_collection(
                        collection_name=KNOWLEDGE_COLLECTION,
                        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                    )
            except Exception as e:
                logger.error(f"Error initializing Qdrant collection: {e}")
        return self._qdrant_client

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            if SentenceTransformer is None:
                raise ValueError("Embedding model (sentence-transformers) is not installed.")
            logger.info("Loading lightweight embedding model (all-MiniLM-L6-v2)...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    async def process_upload(self, file_name: str, file_content: bytes, file_type: str, db: Session = None) -> KnowledgeDocument:
        """Saves file and kicks off processing."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
            
        try:
            print("[RAG] Upload started")
            print(f"[RAG] Filename: {file_name}")
            # Save file
            file_path = os.path.join(self.upload_dir, f"{uuid.uuid4()}_{file_name}")
            with open(file_path, "wb") as f:
                f.write(file_content)
            print("[RAG] File saved")
            
            file_extension = "PDF" if file_path.lower().endswith(".pdf") else "TEXT"
            if file_type == "application/pdf":
                file_extension = "PDF"
            print(f"[RAG] File type detected: {file_extension}")

            # Create document record
            doc = KnowledgeDocument(
                name=file_name,
                file_path=file_path,
                file_type=file_type,
                status="PROCESSING"
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            # Process synchronously for simplicity (could be moved to Celery background task)
            self._extract_and_index(db, doc)
            
            return doc
        except Exception as e:
            print("[RAG][ERROR] Stage: Upload")
            print(f"[RAG][ERROR] Reason: {e}")
            logger.error(f"Upload failed: {e}")
            raise e
        finally:
            if close_db:
                db.close()

    def _extract_and_index(self, db, doc: KnowledgeDocument):
        try:
            # 1. Extract Text
            print("[RAG] Text extraction started")
            text = self._extract_text(doc.file_path, doc.file_type)
            if not text.strip():
                raise ValueError("No text extracted from document")
            print(f"[RAG] Extracted characters: {len(text)}")

            # 2. Chunking
            print("[RAG] Chunking started")
            chunks = self._chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks created")
            print(f"[RAG] Chunks created: {len(chunks)}")

            # 3. Store chunks in DB and create embeddings
            vectors = []
            db_chunks = []
            
            if not self.qdrant:
                raise ValueError("Qdrant client is not available.")
                
            print("[RAG] Embedding generation started")

            # Generate embeddings in batch
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            print(f"[RAG] Embeddings generated: {len(embeddings)}")
            
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                
                # Basic category extraction (could use LLM here, but keeping it fast)
                category = "THEME" if "theme" in chunk_text.lower() else "NARRATIVE_PATTERN"
                
                db_chunk = KnowledgeChunk(
                    id=chunk_id,
                    document_id=doc.id,
                    chunk_index=i,
                    text=chunk_text,
                    category=category,
                    metadata_json={"document_name": doc.name}
                )
                db_chunks.append(db_chunk)
                
                vectors.append(PointStruct(
                    id=chunk_id,
                    vector=embedding.tolist(),
                    payload={
                        "document_id": doc.id,
                        "document_name": doc.name,
                        "chunk_index": i,
                        "text": chunk_text,
                        "category": category,
                        "source": doc.name,
                        "source_type": "reference"
                    }
                ))
            
            # Save to SQLite
            db.add_all(db_chunks)
            doc.chunk_count = len(db_chunks)
            
            if doc.chunk_count == 0:
                raise ValueError("No readable knowledge chunks were extracted from the document.")
                
            doc.status = "COMPLETED"
            db.commit()
            
            # Save to Qdrant
            if vectors and self.qdrant:
                print("[RAG] Vector storage started")
                self.qdrant.upsert(
                    collection_name=KNOWLEDGE_COLLECTION,
                    points=vectors
                )
                print("[RAG] Vector storage completed")
                print("[RAG] Document processing completed")
                logger.info(f"Indexed {len(vectors)} chunks into Qdrant for {doc.name}")

        except Exception as e:
            # Try to identify the failing stage
            stage = "Unknown"
            if 'text' not in locals():
                stage = "Text Extraction"
            elif 'chunks' not in locals():
                stage = "Chunking"
            elif 'embeddings' not in locals():
                stage = "Embedding generation"
            else:
                stage = "Vector storage"
                
            print(f"[RAG][ERROR] Stage: {stage}")
            print(f"[RAG][ERROR] Reason: {e}")
            logger.error(f"Processing failed for document {doc.id} at stage {stage}: {e}")
            doc.status = "FAILED"
            db.commit()

    def _extract_text(self, file_path: str, file_type: str) -> str:
        text = ""
        import os
        if file_type == "application/pdf" and fitz:
            page_count = 0
            with fitz.open(file_path) as pdf:
                page_count = len(pdf)
                for page in pdf:
                    text += page.get_text() + "\n"
            print(f"[RAG] pages: {page_count}")
        elif "wordprocessingml" in file_type and docx:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_type == "text/plain":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            # Fallback text reading for PDFs if type is misidentified
            if file_path.lower().endswith('.pdf') and fitz:
                page_count = 0
                with fitz.open(file_path) as pdf:
                    page_count = len(pdf)
                    for page in pdf:
                        text += page.get_text() + "\n"
                print(f"[RAG] pages: {page_count}")
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        return text

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        # Simple character-based chunking
        words = text.split()
        chunks = []
        i = 0
        words_per_chunk = chunk_size // 5 # approx 200 words
        overlap_words = overlap // 5 # approx 40 words
        
        while i < len(words):
            chunk = " ".join(words[i:i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
            i += (words_per_chunk - overlap_words)
        return chunks

    def retrieve_relevant_themes(self, query: str, document_ids: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves semantically similar chunks from specified reference documents."""
        if not self.embedding_model or not self.qdrant:
            logger.warning("Embedding model or Qdrant not initialized. Cannot retrieve themes.")
            return []
            
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Build filter if document_ids provided
            search_filter = None
            if document_ids:
                from qdrant_client.models import Filter, FieldCondition, MatchAny
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchAny(any=document_ids)
                        )
                    ]
                )
                
            results = self.qdrant.search(
                collection_name=KNOWLEDGE_COLLECTION,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k
            )
            
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {
                        "document_id": hit.payload.get("document_id"),
                        "document_name": hit.payload.get("document_name"),
                        "category": hit.payload.get("category")
                    }
                }
                for hit in results
            ]
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

knowledge_service = KnowledgeService()
