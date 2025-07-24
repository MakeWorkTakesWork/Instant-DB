"""
Core database management for Instant-DB
Main interface for vector database operations
"""

import json
import sqlite3
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

from .embeddings import EmbeddingProvider
from .search import SearchEngine
from .chunking import ChunkingEngine
from .graph_memory import GraphMemoryEngine


class InstantDB:
    """
    Main interface for Instant-DB vector database operations
    
    Supports multiple vector databases and embedding providers
    """
    
    def __init__(self, 
                 db_path: str = "./instant_db_database",
                 embedding_provider: str = "sentence-transformers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db: str = "chroma"):
        """
        Initialize Instant-DB instance
        
        Args:
            db_path: Path to store the vector database
            embedding_provider: 'sentence-transformers' or 'openai'
            embedding_model: Model name for embeddings
            vector_db: 'chroma', 'faiss', or 'sqlite'
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.embedding_provider_name = embedding_provider
        self.embedding_model_name = embedding_model
        self.vector_db_name = vector_db
        
        # Initialize components
        self.embedding_provider = EmbeddingProvider(
            provider=embedding_provider,
            model=embedding_model
        )
        
        self.search_engine = SearchEngine(
            db_path=self.db_path,
            vector_db=vector_db,
            embedding_provider=self.embedding_provider
        )
        
        self.chunking_engine = ChunkingEngine()
        
        # Initialize graph memory engine
        self.graph_memory = GraphMemoryEngine(
            db_path=self.db_path,
            embedding_provider=self.embedding_provider
        )
        
        # Initialize metadata database
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite metadata tracking database"""
        self.metadata_db_path = self.db_path / "metadata.db"
        conn = sqlite3.connect(self.metadata_db_path)
        
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
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                operation TEXT,
                status TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, 
                    content: str,
                    metadata: Dict[str, Any],
                    document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a document to the vector database
        
        Args:
            content: Document content (markdown or plain text)
            metadata: Document metadata (title, type, source, etc.)
            document_id: Optional custom document ID
            
        Returns:
            Dict with processing results
        """
        # Generate document ID if not provided
        if not document_id:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            title = metadata.get('title', 'document')
            document_id = f"{title}_{content_hash}"
        
        # Calculate file hash for change detection
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if document already exists and unchanged
        if self._document_exists(document_id, file_hash):
            return {"status": "skipped", "document_id": document_id, "reason": "unchanged"}
        
        try:
            # Create chunks
            chunks = self.chunking_engine.chunk_text(content, document_id, metadata)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_provider.encode(chunk_texts)
            
            # Convert chunks to dict format for search engine
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                # Flatten all metadata into the chunk dict
                chunk_dict = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count
                }
                
                # Add section/subsection only if not None
                if chunk.section is not None:
                    chunk_dict["section"] = chunk.section
                if chunk.subsection is not None:
                    chunk_dict["subsection"] = chunk.subsection
                
                # Add any additional metadata from the chunk
                for key, value in chunk.metadata.items():
                    # Only add simple types that vector stores can handle, excluding None
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        chunk_dict[key] = value
                
                chunk_dicts.append(chunk_dict)
            
            # Store in vector database - the search engine expects embeddings to be passed separately
            self.search_engine.vector_store.add_documents(chunk_dicts, embeddings)
            
            # Process for graph memory
            graph_result = self.graph_memory.process_document_for_graph(
                document_id, content, chunks, metadata
            )
            
            # Update metadata
            self._update_document_metadata(
                document_id=document_id,
                title=metadata.get('title', document_id),
                document_type=metadata.get('document_type', 'Unknown'),
                source_file=metadata.get('source_file', ''),
                chunk_count=len(chunks),
                total_chars=len(content),
                file_hash=file_hash,
                metadata=metadata
            )
            
            # Log the operation
            self._log_operation(document_id, "add_document", "success", 
                              f"Added {len(chunks)} chunks")
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_processed": len(chunks),
                "embedding_model": self.embedding_model_name,
                "vector_db": self.vector_db_name
            }
            
        except Exception as e:
            self._log_operation(document_id, "add_document", "error", str(e))
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    def search(self, 
              query: str,
              top_k: int = 5,
              document_type: Optional[str] = None,
              document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant content
        
        Args:
            query: Search query text
            top_k: Number of results to return
            document_type: Filter by document type
            document_id: Filter by specific document
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Build filters if document_type or document_id specified
            filters = {}
            if document_type:
                filters["document_type"] = document_type
            if document_id:
                filters["document_id"] = document_id
            
            results = self.search_engine.search(
                query=query,
                top_k=top_k,
                filters=filters if filters else None
            )
            
            # Log search operation
            self._log_operation(None, "search", "success", 
                              f"Query: '{query}', Results: {len(results)}")
            
            return results
            
        except Exception as e:
            self._log_operation(None, "search", "error", 
                              f"Query: '{query}', Error: {str(e)}")
            return []
    
    def graph_search(self, 
                    query: str,
                    top_k: int = 5,
                    include_relationships: bool = True) -> List[Dict[str, Any]]:
        """
        Perform graph-enhanced search using knowledge graph relationships
        
        Args:
            query: Search query text
            top_k: Number of results to return
            include_relationships: Whether to include relationship context
            
        Returns:
            List of search results enhanced with graph context
        """
        try:
            # Get graph-enhanced results
            graph_results = self.graph_memory.graph_enhanced_search(query, top_k * 2)
            
            # Combine with traditional vector search
            vector_results = self.search(query, top_k)
            
            # Merge and rank results
            combined_results = []
            
            # Add graph results with relationship context
            for result in graph_results:
                enhanced_result = {
                    'content': result['text'],
                    'entity_type': result['type'],
                    'similarity': result['similarity'],
                    'graph_score': result['composite_score'],
                    'search_type': 'graph_enhanced',
                    'frequency': result['frequency']
                }
                
                # Add relationship context if requested
                if include_relationships:
                    relationships = self.graph_memory.get_entity_relationships(result['entity_id'])
                    enhanced_result['relationships'] = relationships
                
                combined_results.append(enhanced_result)
            
            # Add vector results
            for result in vector_results:
                combined_results.append({
                    **result,
                    'search_type': 'vector_search',
                    'graph_score': result.get('similarity', 0.0)
                })
            
            # Sort by combined score and remove duplicates
            seen_content = set()
            final_results = []
            
            for result in sorted(combined_results, key=lambda x: x['graph_score'], reverse=True):
                content_key = result['content'][:100]  # First 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    final_results.append(result)
                    
                    if len(final_results) >= top_k:
                        break
            
            # Log search operation
            self._log_operation(None, "graph_search", "success", 
                              f"Query: '{query}', Results: {len(final_results)}")
            
            return final_results
            
        except Exception as e:
            self._log_operation(None, "graph_search", "error", 
                              f"Query: '{query}', Error: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        conn = sqlite3.connect(self.metadata_db_path)
        
        # Document counts by type
        cursor = conn.execute('''
            SELECT document_type, COUNT(*), SUM(chunk_count), SUM(total_chars)
            FROM documents 
            GROUP BY document_type
        ''')
        by_type = {}
        for doc_type, count, chunks, chars in cursor.fetchall():
            by_type[doc_type] = {
                'count': count,
                'chunks': chunks or 0,
                'chars': chars or 0
            }
        
        # Overall stats
        cursor = conn.execute('''
            SELECT COUNT(*), SUM(chunk_count), SUM(total_chars)
            FROM documents
        ''')
        total_docs, total_chunks, total_chars = cursor.fetchone()
        
        # Recent activity
        cursor = conn.execute('''
            SELECT operation, COUNT(*)
            FROM processing_log
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY operation
        ''')
        recent_activity = dict(cursor.fetchall())
        
        conn.close()
        
        # Get graph statistics
        graph_stats = self.graph_memory.get_graph_stats()
        
        return {
            'total_documents': total_docs or 0,
            'total_chunks': total_chunks or 0,
            'total_chars': total_chars or 0,
            'by_type': by_type,
            'recent_activity': recent_activity,
            'embedding_provider': self.embedding_provider_name,
            'embedding_model': self.embedding_model_name,
            'vector_database': self.vector_db_name,
            'database_path': str(self.db_path),
            'graph_memory': graph_stats
        }
    
    def get_documents(self, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of all documents in the database"""
        conn = sqlite3.connect(self.metadata_db_path)
        
        if document_type:
            cursor = conn.execute('''
                SELECT document_id, title, document_type, source_file, 
                       processed_at, chunk_count, total_chars
                FROM documents
                WHERE document_type = ?
                ORDER BY processed_at DESC
            ''', (document_type,))
        else:
            cursor = conn.execute('''
                SELECT document_id, title, document_type, source_file,
                       processed_at, chunk_count, total_chars
                FROM documents
                ORDER BY processed_at DESC
            ''')
        
        documents = []
        for row in cursor.fetchall():
            doc_id, title, doc_type, source, processed, chunks, chars = row
            documents.append({
                'document_id': doc_id,
                'title': title,
                'document_type': doc_type,
                'source_file': source,
                'processed_at': processed,
                'chunk_count': chunks,
                'total_chars': chars
            })
        
        conn.close()
        return documents
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the database"""
        try:
            # Delete from vector database
            self.search_engine.delete_document(document_id)
            
            # Delete from metadata
            conn = sqlite3.connect(self.metadata_db_path)
            conn.execute('DELETE FROM documents WHERE document_id = ?', (document_id,))
            conn.commit()
            conn.close()
            
            self._log_operation(document_id, "delete_document", "success", "Document deleted")
            return True
            
        except Exception as e:
            self._log_operation(document_id, "delete_document", "error", str(e))
            return False
    
    def update_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing document by re-processing it
        
        Args:
            document_id: ID of the document to update
            content: New document content
            metadata: Updated metadata
            
        Returns:
            Dict with update results
        """
        # Calculate new file hash
        new_file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if document exists and get old hash
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute(
            'SELECT file_hash FROM documents WHERE document_id = ?',
            (document_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {"status": "error", "document_id": document_id, "error": "Document not found"}
        
        old_file_hash = result[0]
        
        # Check if content has changed
        if old_file_hash == new_file_hash:
            return {"status": "skipped", "document_id": document_id, "reason": "Content unchanged"}
        
        try:
            # Delete old document
            self.delete_document(document_id)
            
            # Add updated document
            add_result = self.add_document(content, metadata, document_id)
            
            if add_result["status"] == "success":
                self._log_operation(document_id, "update_document", "success", 
                                  f"Updated with {add_result.get('chunks_processed', 0)} chunks")
                return {
                    "status": "updated",
                    "document_id": document_id,
                    "chunks_processed": add_result.get("chunks_processed", 0),
                    "old_hash": old_file_hash[:8],
                    "new_hash": new_file_hash[:8]
                }
            else:
                return add_result
                
        except Exception as e:
            self._log_operation(document_id, "update_document", "error", str(e))
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    def check_for_updates(self, source_directory: Path) -> Dict[str, Any]:
        """
        Check for documents that need updating based on file changes
        
        Args:
            source_directory: Directory to scan for updates
            
        Returns:
            Dict with files that need updating
        """
        from ..core.discovery import DocumentDiscovery
        
        # Get all documents in database with source files
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute('''
            SELECT document_id, source_file, file_hash 
            FROM documents 
            WHERE source_file IS NOT NULL AND source_file != ''
        ''')
        
        db_documents = {}
        for doc_id, source_file, file_hash in cursor.fetchall():
            db_documents[source_file] = {
                "document_id": doc_id,
                "file_hash": file_hash
            }
        conn.close()
        
        # Scan directory for current files
        discovery = DocumentDiscovery()
        current_files = discovery.scan_directory_for_documents(
            directory=source_directory,
            recursive=True
        )
        
        updates_needed = {
            "new_files": [],
            "modified_files": [],
            "deleted_files": []
        }
        
        # Check for new and modified files
        for doc_metadata in current_files:
            file_path = str(doc_metadata.file_path)
            
            # Calculate current file hash
            try:
                with open(file_path, 'rb') as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
            except Exception:
                continue
            
            if file_path in db_documents:
                # Check if modified
                if db_documents[file_path]["file_hash"] != current_hash:
                    updates_needed["modified_files"].append({
                        "file_path": file_path,
                        "document_id": db_documents[file_path]["document_id"],
                        "old_hash": db_documents[file_path]["file_hash"][:8],
                        "new_hash": current_hash[:8]
                    })
                db_documents.pop(file_path)  # Remove from tracking
            else:
                # New file
                updates_needed["new_files"].append({
                    "file_path": file_path,
                    "file_type": doc_metadata.file_type,
                    "file_size_mb": doc_metadata.file_size / (1024 * 1024)
                })
        
        # Remaining files in db_documents are deleted
        for source_file, doc_info in db_documents.items():
            updates_needed["deleted_files"].append({
                "file_path": source_file,
                "document_id": doc_info["document_id"]
            })
        
        return updates_needed
    
    def export_database(self, export_path: str, format: str = "json") -> str:
        """Export database content in various formats"""
        export_path = Path(export_path)
        export_path.mkdir(exist_ok=True)
        
        if format == "json":
            return self._export_json(export_path)
        elif format == "custom-gpt":
            return self._export_custom_gpt(export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, export_path: Path) -> str:
        """Export database as JSON"""
        # Get all documents and their content
        documents = self.get_documents()
        
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_documents': len(documents),
                'embedding_model': self.embedding_model_name,
                'vector_db': self.vector_db_name
            },
            'documents': []
        }
        
        for doc in documents:
            # Get document chunks
            chunks = self.search_engine.get_document_chunks(doc['document_id'])
            
            export_data['documents'].append({
                'document_id': doc['document_id'],
                'title': doc['title'],
                'document_type': doc['document_type'],
                'source_file': doc['source_file'],
                'processed_at': doc['processed_at'],
                'chunks': chunks
            })
        
        # Save to file
        export_file = export_path / "instant_db_export.json"
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(export_file)
    
    def _export_custom_gpt(self, export_path: Path) -> str:
        """Export database in Custom GPT format"""
        from ..integrations.custom_gpt import CustomGPTExporter
        
        exporter = CustomGPTExporter(self)
        return exporter.export(str(export_path / "custom_gpt_export.md"))
    
    def _document_exists(self, document_id: str, file_hash: str) -> bool:
        """Check if document already exists with the same hash"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute(
            'SELECT file_hash FROM documents WHERE document_id = ?',
            (document_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result is not None and result[0] == file_hash
    
    def _update_document_metadata(self, document_id: str, title: str, document_type: str,
                                 source_file: str, chunk_count: int, total_chars: int,
                                 file_hash: str, metadata: Dict[str, Any]):
        """Update document metadata in tracking database"""
        conn = sqlite3.connect(self.metadata_db_path)
        
        metadata_json = json.dumps(metadata)
        
        conn.execute('''
            INSERT OR REPLACE INTO documents 
            (document_id, title, document_type, source_file, chunk_count, 
             total_chars, file_hash, embedding_model, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (document_id, title, document_type, source_file, chunk_count,
              total_chars, file_hash, self.embedding_model_name, metadata_json))
        
        conn.commit()
        conn.close()
    
    def _log_operation(self, document_id: Optional[str], operation: str, 
                      status: str, details: str):
        """Log an operation to the processing log"""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute('''
            INSERT INTO processing_log (document_id, operation, status, details)
            VALUES (?, ?, ?, ?)
        ''', (document_id, operation, status, details))
        conn.commit()
        conn.close() 