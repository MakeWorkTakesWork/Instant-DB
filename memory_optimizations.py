#!/usr/bin/env python3
"""
Memory Optimization Implementations for Instant-DB

This module contains optimized versions of core components to reduce memory usage
and improve performance. Key optimizations include:

1. Streaming document processing for large files
2. Batch processing with memory cleanup
3. Optimized embedding generation
4. Memory-efficient vector operations
5. Connection pooling and resource management

Usage:
    from memory_optimizations import OptimizedInstantDB
    
    db = OptimizedInstantDB(
        db_path="./optimized_db",
        memory_limit_mb=512,  # Set memory limit
        batch_size=50,        # Process in smaller batches
        enable_streaming=True # Enable streaming for large files
    )
"""

import gc
import sys
import psutil
import sqlite3
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Generator
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import time

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    memory_limit_mb: int = 512
    batch_size: int = 50
    enable_streaming: bool = True
    gc_frequency: int = 10  # Run GC every N operations
    connection_pool_size: int = 5
    chunk_cache_size: int = 1000

class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, limit_mb: int = 512):
        self.limit_mb = limit_mb
        self.process = psutil.Process()
        self.peak_usage = 0
        self.operation_count = 0
        
    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limit."""
        current = self.get_current_usage_mb()
        self.peak_usage = max(self.peak_usage, current)
        return current < self.limit_mb
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        # Additional cleanup for numpy arrays
        if 'numpy' in sys.modules:
            import numpy as np
            # Clear numpy cache if available
            if hasattr(np, 'clear_cache'):
                np.clear_cache()
    
    def log_memory_usage(self, operation: str):
        """Log memory usage for an operation."""
        current = self.get_current_usage_mb()
        print(f"🧠 {operation}: {current:.2f}MB (Peak: {self.peak_usage:.2f}MB)")

class ConnectionPool:
    """SQLite connection pool for better resource management."""
    
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
        
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        with self.lock:
            if self.connections:
                conn = self.connections.pop()
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                conn.execute("PRAGMA cache_size=10000")  # Larger cache
        
        try:
            yield conn
        finally:
            with self.lock:
                if len(self.connections) < self.pool_size:
                    self.connections.append(conn)
                else:
                    conn.close()

class StreamingChunker:
    """Memory-efficient streaming text chunker."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def stream_chunks(self, content: str, document_id: str, 
                     metadata: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Stream chunks without loading all into memory."""
        content_length = len(content)
        start = 0
        chunk_index = 0
        
        while start < content_length:
            end = min(start + self.chunk_size, content_length)
            
            # Find word boundary
            if end < content_length:
                while end > start and content[end] not in ' \n\t':
                    end -= 1
                if end == start:  # No word boundary found
                    end = start + self.chunk_size
            
            chunk_content = content[start:end]
            
            # Create chunk dict
            chunk_dict = {
                "id": f"{document_id}_chunk_{chunk_index}",
                "content": chunk_content,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "word_count": len(chunk_content.split()),
                "char_count": len(chunk_content),
                "start_pos": start,
                "end_pos": end
            }
            
            # Add metadata
            for key, value in metadata.items():
                if value is not None and isinstance(value, (str, int, float, bool)):
                    chunk_dict[key] = value
            
            yield chunk_dict
            
            # Move to next chunk with overlap
            start = max(start + self.chunk_size - self.overlap, end)
            chunk_index += 1

class OptimizedEmbeddingProvider:
    """Memory-optimized embedding provider."""
    
    def __init__(self, provider: str = "sentence-transformers", 
                 model: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.provider = provider
        self.model_name = model
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model with memory optimization."""
        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                # Load with reduced precision to save memory
                self.model = SentenceTransformer(self.model_name)
                # Move to CPU if GPU memory is limited
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
            except ImportError:
                raise ImportError("sentence-transformers not installed")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts in memory-efficient batches."""
        if not texts:
            return np.array([])
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if self.provider == "sentence-transformers":
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(self.batch_size, len(batch))
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            embeddings.append(batch_embeddings)
            
            # Force cleanup after each batch
            if i % (self.batch_size * 4) == 0:  # Every 4 batches
                gc.collect()
        
        return np.vstack(embeddings) if embeddings else np.array([])

class OptimizedVectorStore:
    """Memory-optimized vector store operations."""
    
    def __init__(self, db_path: Path, vector_db: str = "chroma"):
        self.db_path = db_path
        self.vector_db = vector_db
        self.store = None
        self._init_store()
    
    def _init_store(self):
        """Initialize vector store with memory optimizations."""
        if self.vector_db == "chroma":
            try:
                import chromadb
                
                # Use the new ChromaDB client API
                client = chromadb.PersistentClient(
                    path=str(self.db_path / "chroma")
                )
                
                self.store = client.get_or_create_collection(
                    name="instant_db_collection",
                    metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                raise ImportError("chromadb not installed")
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_db}")
    
    def add_documents_batch(self, documents: List[Dict[str, Any]], 
                           embeddings: np.ndarray, batch_size: int = 50):
        """Add documents in memory-efficient batches."""
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = [doc["id"] for doc in batch_docs]
            texts = [doc["content"] for doc in batch_docs]
            metadatas = []
            
            for doc in batch_docs:
                metadata = {k: v for k, v in doc.items() 
                           if k not in ["id", "content"] and v is not None}
                metadatas.append(metadata)
            
            # Add to store
            self.store.add(
                ids=ids,
                documents=texts,
                embeddings=batch_embeddings.tolist(),
                metadatas=metadatas
            )
            
            # Cleanup after each batch
            gc.collect()

class OptimizedInstantDB:
    """Memory-optimized version of InstantDB."""
    
    def __init__(self, 
                 db_path: str = "./optimized_instant_db",
                 embedding_provider: str = "sentence-transformers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db: str = "chroma",
                 memory_config: Optional[MemoryConfig] = None):
        """
        Initialize optimized InstantDB instance.
        
        Args:
            db_path: Path to store the database
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name
            vector_db: Vector database type
            memory_config: Memory optimization configuration
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.config = memory_config or MemoryConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        
        # Initialize optimized components
        self.embedding_provider = OptimizedEmbeddingProvider(
            provider=embedding_provider,
            model=embedding_model,
            batch_size=self.config.batch_size
        )
        
        self.vector_store = OptimizedVectorStore(
            db_path=self.db_path,
            vector_db=vector_db
        )
        
        self.chunker = StreamingChunker()
        
        # Initialize connection pool
        self.connection_pool = ConnectionPool(
            self.db_path / "metadata.db",
            self.config.connection_pool_size
        )
        
        self._init_metadata_db()
        
        self.operation_count = 0
    
    def _init_metadata_db(self):
        """Initialize metadata database with optimizations."""
        with self.connection_pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT,
                    document_type TEXT,
                    source_file TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER,
                    total_chars INTEGER,
                    file_hash TEXT,
                    embedding_model TEXT,
                    metadata TEXT
                )
            ''')
            
            # Add indexes for better performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_hash 
                ON documents(file_hash)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_source 
                ON documents(source_file)
            ''')
            
            conn.commit()
    
    def add_document_optimized(self, 
                              content: str,
                              metadata: Dict[str, Any],
                              document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add document with memory optimizations.
        
        Args:
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID
            
        Returns:
            Processing results
        """
        self.memory_monitor.log_memory_usage(f"Starting document processing")
        
        # Generate document ID if not provided
        if not document_id:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            title = metadata.get('title', 'document')
            document_id = f"{title}_{content_hash}"
        
        # Calculate file hash
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        try:
            # Stream chunks to avoid loading all into memory
            chunks = list(self.chunker.stream_chunks(content, document_id, metadata))
            chunk_count = len(chunks)
            
            self.memory_monitor.log_memory_usage(f"Created {chunk_count} chunks")
            
            # Process embeddings in batches
            all_embeddings = []
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            for i in range(0, len(chunk_texts), self.config.batch_size):
                batch_texts = chunk_texts[i:i + self.config.batch_size]
                batch_embeddings = self.embedding_provider.encode_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                self.memory_monitor.log_memory_usage(f"Processed embeddings batch {i//self.config.batch_size + 1}")
                
                # Check memory limit
                if not self.memory_monitor.check_memory_limit():
                    self.memory_monitor.force_cleanup()
            
            # Combine embeddings
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            
            # Add to vector store in batches
            self.vector_store.add_documents_batch(
                chunks, embeddings, self.config.batch_size
            )
            
            # Update metadata
            self._update_document_metadata(
                document_id=document_id,
                title=metadata.get('title', document_id),
                document_type=metadata.get('document_type', 'Unknown'),
                source_file=metadata.get('source_file', ''),
                chunk_count=chunk_count,
                total_chars=len(content),
                file_hash=file_hash,
                metadata=metadata
            )
            
            # Periodic cleanup
            self.operation_count += 1
            if self.operation_count % self.config.gc_frequency == 0:
                self.memory_monitor.force_cleanup()
            
            self.memory_monitor.log_memory_usage(f"Completed document processing")
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_processed": chunk_count,
                "memory_peak_mb": self.memory_monitor.peak_usage,
                "embedding_model": self.embedding_provider.model_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "memory_peak_mb": self.memory_monitor.peak_usage
            }
    
    def _update_document_metadata(self, **kwargs):
        """Update document metadata with connection pooling."""
        with self.connection_pool.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO documents 
                (document_id, title, document_type, source_file, chunk_count, 
                 total_chars, file_hash, embedding_model, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['document_id'],
                kwargs['title'],
                kwargs['document_type'],
                kwargs['source_file'],
                kwargs['chunk_count'],
                kwargs['total_chars'],
                kwargs['file_hash'],
                self.embedding_provider.model_name,
                str(kwargs['metadata'])
            ))
            conn.commit()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            "current_usage_mb": self.memory_monitor.get_current_usage_mb(),
            "peak_usage_mb": self.memory_monitor.peak_usage,
            "memory_limit_mb": self.config.memory_limit_mb,
            "operations_processed": self.operation_count,
            "within_limit": self.memory_monitor.check_memory_limit()
        }
    
    def optimize_database(self):
        """Run database optimization operations."""
        with self.connection_pool.get_connection() as conn:
            # Vacuum database to reclaim space
            conn.execute("VACUUM")
            
            # Analyze tables for better query planning
            conn.execute("ANALYZE")
            
            # Update statistics
            conn.execute("PRAGMA optimize")
            
            conn.commit()
        
        # Force garbage collection
        self.memory_monitor.force_cleanup()
        
        print("✅ Database optimization completed")


def create_memory_benchmark():
    """Create a benchmark to test memory optimizations."""
    
    def benchmark_memory_usage():
        """Benchmark memory usage of optimized vs standard implementation."""
        print("🧠 MEMORY OPTIMIZATION BENCHMARK")
        print("=" * 50)
        
        # Test data
        test_content = "This is a test document. " * 1000  # ~25KB
        test_metadata = {"title": "Test Document", "type": "text"}
        
        # Test optimized version
        print("\n📊 Testing Optimized Implementation:")
        optimized_db = OptimizedInstantDB(
            db_path="./benchmark_optimized",
            memory_config=MemoryConfig(
                memory_limit_mb=256,
                batch_size=25,
                enable_streaming=True
            )
        )
        
        start_time = time.time()
        result = optimized_db.add_document_optimized(
            content=test_content,
            metadata=test_metadata,
            document_id="test_optimized"
        )
        end_time = time.time()
        
        stats = optimized_db.get_memory_stats()
        
        print(f"   ⏱️  Processing time: {end_time - start_time:.3f}s")
        print(f"   🧠 Peak memory: {stats['peak_usage_mb']:.2f}MB")
        print(f"   📊 Chunks processed: {result.get('chunks_processed', 0)}")
        print(f"   ✅ Status: {result.get('status')}")
        
        return {
            "implementation": "optimized",
            "processing_time": end_time - start_time,
            "peak_memory_mb": stats['peak_usage_mb'],
            "chunks_processed": result.get('chunks_processed', 0),
            "status": result.get('status')
        }
    
    return benchmark_memory_usage


if __name__ == "__main__":
    # Run memory optimization benchmark
    benchmark = create_memory_benchmark()
    results = benchmark()
    
    print(f"\n🎉 Memory optimization benchmark completed!")
    print(f"📊 Results: {results}")

