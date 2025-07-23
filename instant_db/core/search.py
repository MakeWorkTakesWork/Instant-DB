"""
Vector search engine for Instant-DB
Supports multiple vector databases with unified search interface
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in store"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, db_path: Path, collection_name: str = "instant_db"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to ChromaDB"""
        if not documents or len(embeddings) == 0:
            return
        
        # Prepare data for ChromaDB
        ids = [doc["id"] for doc in documents]
        metadatas = [
            {k: v for k, v in doc.items() if k != "id" and k != "content"}
            for doc in documents
        ]
        documents_content = [doc.get("content", "") for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents_content
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search ChromaDB for similar documents"""
        # Convert filters to ChromaDB format
        where_clause = None
        if filters:
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        # Convert to standard format
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i] or {}
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": metadata
                }
                # Also add key metadata fields to top level for convenience
                if "document_id" in metadata:
                    result["document_id"] = metadata["document_id"]
                search_results.append(result)
        
        return search_results
    
    def get_document_count(self) -> int:
        """Get total document count"""
        return self.collection.count()


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, db_path: Path, dimension: int):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
        
        self.db_path = db_path
        self.dimension = dimension
        self.index_path = db_path / "faiss.index"
        self.metadata_path = db_path / "faiss_metadata.json"
        
        # Initialize FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Load metadata
        self.documents = []
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.documents = json.load(f)
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to FAISS index"""
        if not documents or len(embeddings) == 0:
            return
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # Store metadata
        self.documents.extend(documents)
        
        # Save index and metadata
        self._save()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search FAISS index for similar documents"""
        if self.get_document_count() == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embedding, min(top_k, self.get_document_count()))
        
        # Convert to standard format and apply filters
        search_results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    doc_value = doc.get(key)
                    if isinstance(value, list):
                        if doc_value not in value:
                            skip = True
                            break
                    else:
                        if doc_value != value:
                            skip = True
                            break
                
                if skip:
                    continue
            
            result = {
                "id": doc["id"],
                "content": doc.get("content", ""),
                "similarity": float(similarity),
                "metadata": {k: v for k, v in doc.items() if k not in ["id", "content"]}
            }
            search_results.append(result)
        
        return search_results
    
    def get_document_count(self) -> int:
        """Get total document count"""
        return len(self.documents)
    
    def _save(self):
        """Save FAISS index and metadata"""
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.documents, f)


class SQLiteVectorStore(BaseVectorStore):
    """SQLite vector store implementation (simple, for small datasets)"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_file = db_path / "sqlite_vectors.db"
        
        # Initialize SQLite database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_file)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to SQLite database"""
        if not documents or len(embeddings) == 0:
            return
        
        conn = sqlite3.connect(self.db_file)
        
        for doc, embedding in zip(documents, embeddings):
            metadata = {k: v for k, v in doc.items() if k not in ["id", "content"]}
            
            conn.execute("""
                INSERT OR REPLACE INTO vectors (id, content, embedding, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                doc["id"],
                doc.get("content", ""),
                pickle.dumps(embedding),
                json.dumps(metadata)
            ))
        
        conn.commit()
        conn.close()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search SQLite database for similar documents"""
        conn = sqlite3.connect(self.db_file)
        
        # Build filter query
        where_clause = ""
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    placeholders = ",".join("?" * len(value))
                    conditions.append(f"json_extract(metadata, '$.{key}') IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
            
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)
        
        # Get all documents (SQLite doesn't have native vector search)
        cursor = conn.execute(f"SELECT id, content, embedding, metadata FROM vectors{where_clause}", params)
        rows = cursor.fetchall()
        
        # Calculate similarities
        similarities = []
        for row in rows:
            doc_id, content, embedding_blob, metadata_json = row
            doc_embedding = pickle.loads(embedding_blob)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            similarities.append({
                "id": doc_id,
                "content": content,
                "similarity": float(similarity),
                "metadata": json.loads(metadata_json)
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        conn.close()
        
        return similarities[:top_k]
    
    def get_document_count(self) -> int:
        """Get total document count"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        conn.close()
        return count


class SearchEngine:
    """
    Main search engine that orchestrates vector search operations
    """
    
    def __init__(self, db_path: Path, vector_db: str, embedding_provider):
        """
        Initialize search engine
        
        Args:
            db_path: Path to database storage
            vector_db: Vector database type ('chroma', 'faiss', 'sqlite')
            embedding_provider: EmbeddingProvider instance
        """
        self.db_path = db_path
        self.vector_db_name = vector_db
        self.embedding_provider = embedding_provider
        
        # Initialize vector store
        if vector_db == "chroma":
            self.vector_store = ChromaVectorStore(db_path)
        elif vector_db == "faiss":
            dimension = embedding_provider.get_dimension()
            self.vector_store = FAISSVectorStore(db_path, dimension)
        elif vector_db == "sqlite":
            self.vector_store = SQLiteVectorStore(db_path)
        else:
            raise ValueError(f"Unsupported vector database: {vector_db}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add documents to the search index
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and metadata
            
        Returns:
            Results dictionary with processing stats
        """
        if not documents:
            return {"status": "success", "documents_added": 0}
        
        # Extract content for embedding
        texts = [doc.get("content", "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_provider.encode(texts)
        
        # Add to vector store
        self.vector_store.add_documents(documents, embeddings)
        
        return {
            "status": "success",
            "documents_added": len(documents),
            "embeddings_generated": len(embeddings)
        }
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with similarity scores
        """
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.encode([query])[0]
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k, filters)
        
        # Enhance results with additional metadata
        for result in results:
            result.update({
                "query": query,
                "vector_db": self.vector_db_name,
                "embedding_provider": self.embedding_provider.provider_name
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "vector_db": self.vector_db_name,
            "embedding_provider": self.embedding_provider.provider_name,
            "embedding_model": self.embedding_provider.model_name,
            "document_count": self.vector_store.get_document_count(),
            "embedding_dimension": self.embedding_provider.get_dimension(),
            "cache_stats": self.embedding_provider.get_cache_stats()
        } 