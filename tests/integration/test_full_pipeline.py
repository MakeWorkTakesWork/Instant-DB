"""
Integration tests for full document processing pipeline
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from instant_db.core.database import InstantDB
from instant_db.processors.document import DocumentProcessor
from instant_db.processors.batch import BatchProcessor


class TestFullPipeline:
    """Test complete document processing pipeline"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests"""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_documents(self, temp_workspace):
        """Create sample documents for testing"""
        docs_dir = temp_workspace / "documents"
        docs_dir.mkdir()
        
        # Create various document types
        (docs_dir / "readme.txt").write_text("""
        # Project README
        
        This is a sample project that demonstrates machine learning concepts.
        
        ## Features
        - Data preprocessing
        - Model training
        - Evaluation metrics
        
        ## Installation
        pip install -r requirements.txt
        """)
        
        (docs_dir / "tutorial.md").write_text("""
        # Machine Learning Tutorial
        
        ## Introduction
        Machine learning is a subset of artificial intelligence.
        
        ## Getting Started
        1. Prepare your data
        2. Choose a model
        3. Train the model
        4. Evaluate performance
        
        ### Data Preparation
        Clean and preprocess your dataset.
        
        ### Model Selection
        Consider different algorithms based on your problem type.
        """)
        
        (docs_dir / "notes.txt").write_text("""
        Meeting Notes - January 15, 2024
        
        Topics Discussed:
        - Project timeline
        - Resource allocation  
        - Risk assessment
        
        Action Items:
        1. Complete data collection by end of month
        2. Schedule follow-up meeting
        3. Prepare progress report
        """)
        
        return docs_dir
    
    def test_single_document_processing(self, temp_workspace, sample_documents):
        """Test processing a single document end-to-end"""
        db_path = temp_workspace / "test_db"
        
        # Initialize processor
        processor = DocumentProcessor(
            embedding_provider="sentence-transformers",
            vector_db="sqlite",
            chunk_size=300,
            chunk_overlap=50
        )
        
        # Process single document
        readme_file = sample_documents / "readme.txt"
        result = processor.process_document(readme_file, str(db_path))
        
        # Verify processing results
        assert result['status'] == 'success'
        assert result['chunks_processed'] > 0
        assert result['total_words'] > 0
        assert Path(result['database_path']).exists()
        
        # Verify database was created and can be searched
        db = InstantDB(db_path=str(db_path), vector_db="sqlite")
        
        # Test search functionality
        search_results = db.search("machine learning", top_k=3)
        assert len(search_results) > 0
        
        # Verify search results contain expected content
        found_content = " ".join([r['content'] for r in search_results])
        assert any(term in found_content.lower() for term in ['machine', 'learning', 'project'])
    
    def test_batch_processing(self, temp_workspace, sample_documents):
        """Test batch processing multiple documents"""
        db_path = temp_workspace / "batch_db"
        
        # Initialize batch processor
        processor = DocumentProcessor(
            embedding_provider="sentence-transformers",
            vector_db="sqlite",
            chunk_size=400,
            chunk_overlap=80
        )
        batch_processor = BatchProcessor(
            document_processor=processor,
            max_workers=2,
            skip_errors=True
        )
        
        # Process all documents in directory
        result = batch_processor.process_directory(
            input_dir=sample_documents,
            output_dir=str(db_path),
            recursive=True
        )
        
        # Verify batch processing results
        assert result['status'] == 'completed'
        assert result['files_processed'] >= 3
        assert result['total_chunks'] > 0
        assert result['files_with_errors'] == 0
        
        # Verify all documents are searchable
        db = InstantDB(db_path=str(db_path), vector_db="sqlite")
        
        # Test searches for content from different documents
        search_queries = [
            "machine learning",      # From tutorial.md
            "project readme",        # From readme.txt
            "meeting notes",         # From notes.txt
            "data preparation",      # From tutorial.md
            "action items"           # From notes.txt
        ]
        
        for query in search_queries:
            results = db.search(query, top_k=5)
            assert len(results) > 0, f"No results found for query: {query}"
            
            # Verify relevance scores
            for result in results:
                assert 'similarity' in result
                assert result['similarity'] > 0
    
    def test_search_filtering(self, temp_workspace, sample_documents):
        """Test search with document type filtering"""
        db_path = temp_workspace / "filtered_db"
        
        # Process documents with metadata
        processor = DocumentProcessor(
            embedding_provider="sentence-transformers",
            vector_db="sqlite"
        )
        
        for doc_file in sample_documents.iterdir():
            if doc_file.is_file():
                result = processor.process_document(doc_file, str(db_path))
                assert result['status'] == 'success'
        
        # Test filtering by document type
        db = InstantDB(db_path=str(db_path), vector_db="sqlite")
        
        # Search for markdown documents only
        md_results = db.search(
            "machine learning",
            top_k=10,
            filters={"document_type": "Markdown Document"}
        )
        
        # Verify results are filtered correctly
        for result in md_results:
            assert result.get('document_type') == "Markdown Document" or 'tutorial.md' in result.get('source_file', '')
    
    def test_database_statistics(self, temp_workspace, sample_documents):
        """Test database statistics after processing"""
        db_path = temp_workspace / "stats_db"
        
        # Process documents
        batch_processor = BatchProcessor(
            DocumentProcessor(vector_db="sqlite"),
            max_workers=1
        )
        
        result = batch_processor.process_directory(
            input_dir=sample_documents,
            output_dir=str(db_path)
        )
        assert result['status'] == 'completed'
        
        # Get database statistics
        db = InstantDB(db_path=str(db_path), vector_db="sqlite")
        stats = db.get_stats()
        
        # Verify statistics
        assert stats['document_count'] > 0
        assert stats['embedding_dimension'] > 0
        assert stats['vector_db'] == 'sqlite'
        assert stats['embedding_provider'] == 'sentence-transformers'
    
    def test_error_handling(self, temp_workspace):
        """Test error handling in processing pipeline"""
        docs_dir = temp_workspace / "bad_docs"
        docs_dir.mkdir()
        
        # Create a file with unsupported extension
        (docs_dir / "unsupported.xyz").write_text("This file has unsupported extension")
        
        # Create a valid document
        (docs_dir / "valid.txt").write_text("This is a valid text document")
        
        # Process with error handling
        processor = DocumentProcessor(vector_db="sqlite")
        batch_processor = BatchProcessor(
            document_processor=processor,
            skip_errors=True
        )
        
        result = batch_processor.process_directory(
            input_dir=docs_dir,
            output_dir=str(temp_workspace / "error_db")
        )
        
        # Should complete despite errors
        assert result['status'] == 'completed'
        assert result['files_processed'] >= 1  # Valid file should be processed
        assert result['files_with_errors'] >= 0  # Might have errors from unsupported files
    
    def test_chunking_quality(self, temp_workspace, sample_documents):
        """Test that chunking produces quality results"""
        db_path = temp_workspace / "quality_db"
        
        processor = DocumentProcessor(
            embedding_provider="sentence-transformers",
            vector_db="sqlite",
            chunk_size=200,
            chunk_overlap=40
        )
        
        # Process the tutorial document (has good structure)
        tutorial_file = sample_documents / "tutorial.md"
        result = processor.process_document(tutorial_file, str(db_path))
        
        assert result['status'] == 'success'
        assert 'chunk_stats' in result
        
        stats = result['chunk_stats']
        
        # Verify chunking quality
        assert stats['total_chunks'] > 1
        assert stats['avg_chars_per_chunk'] > 0
        assert stats['sections_detected'] > 0
        
        # Verify sections were properly detected
        assert len(stats['sections']) > 0
        section_names = [s.lower() for s in stats['sections']]
        assert any('introduction' in name for name in section_names)
    
    def test_search_relevance(self, temp_workspace, sample_documents):
        """Test search result relevance and ranking"""
        db_path = temp_workspace / "relevance_db"
        
        # Process all documents
        batch_processor = BatchProcessor(
            DocumentProcessor(vector_db="sqlite"),
            max_workers=1
        )
        
        batch_processor.process_directory(
            input_dir=sample_documents,
            output_dir=str(db_path)
        )
        
        db = InstantDB(db_path=str(db_path), vector_db="sqlite")
        
        # Test specific queries and verify relevance
        test_cases = [
            {
                'query': 'machine learning tutorial',
                'expected_doc': 'tutorial.md',
                'min_similarity': 0.3
            },
            {
                'query': 'project installation requirements',
                'expected_doc': 'readme.txt',
                'min_similarity': 0.2
            },
            {
                'query': 'meeting discussion action items',
                'expected_doc': 'notes.txt',
                'min_similarity': 0.2
            }
        ]
        
        for test_case in test_cases:
            results = db.search(test_case['query'], top_k=5)
            assert len(results) > 0
            
            # Verify top result has good similarity score
            top_result = results[0]
            assert top_result['similarity'] >= test_case['min_similarity']
            
            # Verify results are ranked by similarity (descending)
            similarities = [r['similarity'] for r in results]
            assert similarities == sorted(similarities, reverse=True)
            
            # Check if expected document appears in top results
            source_files = [r.get('source_file', '') for r in results[:3]]
            assert any(test_case['expected_doc'] in sf for sf in source_files) 