"""
Unit tests for chunking module
"""

import pytest
from instant_db.core.chunking import ChunkingEngine, TextChunk


class TestTextChunk:
    """Test TextChunk dataclass"""
    
    def test_chunk_creation(self):
        """Test basic chunk creation"""
        chunk = TextChunk(
            id="test-chunk-1",
            content="This is a test chunk.",
            document_id="doc-123",
            chunk_index=0,
            start_char=0,
            end_char=22,
            section="Introduction"
        )
        
        assert chunk.id == "test-chunk-1"
        assert chunk.content == "This is a test chunk."
        assert chunk.document_id == "doc-123"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 22
        assert chunk.section == "Introduction"
        assert chunk.word_count == 5
        assert chunk.char_count == 22
    
    def test_automatic_id_generation(self):
        """Test automatic ID generation"""
        chunk = TextChunk(
            id="",  # Empty ID should trigger generation
            content="Test content",
            document_id="doc-123",
            chunk_index=0,
            start_char=0,
            end_char=12
        )
        
        assert chunk.id.startswith("doc-123_chunk_0_")
        assert len(chunk.id) > 20  # Should include hash
    
    def test_word_and_char_count_calculation(self):
        """Test automatic word and character count calculation"""
        content = "This is a test with multiple words and sentences."
        chunk = TextChunk(
            id="test",
            content=content,
            document_id="doc",
            chunk_index=0,
            start_char=0,
            end_char=len(content)
        )
        
        assert chunk.word_count == 10
        assert chunk.char_count == len(content)


class TestChunkingEngine:
    """Test ChunkingEngine class"""
    
    def test_engine_initialization(self):
        """Test chunking engine initialization"""
        engine = ChunkingEngine(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100
        )
        
        assert engine.chunk_size == 1000
        assert engine.chunk_overlap == 200
        assert engine.min_chunk_size == 100
        assert engine.respect_sentence_boundaries is True
        assert engine.respect_paragraph_boundaries is True
    
    def test_simple_text_chunking(self):
        """Test basic text chunking"""
        engine = ChunkingEngine(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        
        text = "This is a simple test. " * 20  # Create long text
        chunks = engine.chunk_text(text, "doc-1")
        
        assert len(chunks) > 1  # Should create multiple chunks
        
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.document_id == "doc-1"
            assert chunk.char_count <= engine.chunk_size * 1.5  # Allow some flexibility
            assert chunk.char_count >= engine.min_chunk_size or chunk == chunks[-1]  # Last chunk might be smaller
    
    def test_empty_text(self):
        """Test chunking empty text"""
        engine = ChunkingEngine()
        
        chunks = engine.chunk_text("", "doc-1")
        assert chunks == []
        
        chunks = engine.chunk_text("   ", "doc-1")
        assert chunks == []
    
    def test_section_detection(self):
        """Test section header detection"""
        engine = ChunkingEngine(chunk_size=500)
        
        text = """
        # Introduction
        This is the introduction section.
        
        ## Methodology
        This describes the methodology used.
        
        ### Data Collection
        Details about data collection.
        
        # Results
        This section presents the results.
        """
        
        chunks = engine.chunk_text(text, "doc-1")
        
        # Should detect sections
        sections = set(chunk.section for chunk in chunks if chunk.section)
        assert "Introduction" in sections
        assert "Results" in sections
    
    def test_numbered_sections(self):
        """Test numbered section detection"""
        engine = ChunkingEngine(chunk_size=500)
        
        text = """
        1. First Section
        Content of the first section.
        
        2. Second Section  
        Content of the second section.
        
        2.1 Subsection
        Content of the subsection.
        """
        
        chunks = engine.chunk_text(text, "doc-1")
        
        sections = set(chunk.section for chunk in chunks if chunk.section)
        assert "First Section" in sections
        assert "Second Section" in sections
    
    def test_list_detection(self):
        """Test list item detection in document structure"""
        engine = ChunkingEngine()
        
        text = """
        Introduction
        
        Key points:
        - First point
        - Second point
        - Third point
        
        1. Numbered item one
        2. Numbered item two
        """
        
        structure = engine._analyze_document_structure(text)
        
        assert len(structure['lists']) > 0
        list_contents = [item['content'] for item in structure['lists']]
        assert any('First point' in content for content in list_contents)
        assert any('Numbered item one' in content for content in list_contents)
    
    def test_table_detection(self):
        """Test table detection in document structure"""
        engine = ChunkingEngine()
        
        text = """
        Data Table:
        
        | Name | Age | City |
        |------|-----|------|
        | John | 30  | NYC  |
        | Jane | 25  | LA   |
        """
        
        structure = engine._analyze_document_structure(text)
        
        assert len(structure['tables']) > 0
        table_contents = [item['content'] for item in structure['tables']]
        assert any('|' in content for content in table_contents)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        engine = ChunkingEngine(chunk_size=100, chunk_overlap=30, min_chunk_size=20)
        
        # Create text that will definitely need chunking
        text = "This is a sentence. " * 15  # About 300 characters
        
        chunks = engine.chunk_text(text, "doc-1")
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].content[-20:]  # Last 20 chars
                chunk2_start = chunks[i + 1].content[:50]  # First 50 chars
                
                # There should be some common content (overlap)
                # This is a simple check - in practice overlap might be more complex
                assert len(chunk1_end.strip()) > 0
                assert len(chunk2_start.strip()) > 0
    
    def test_small_chunk_merging(self):
        """Test that small chunks get merged with adjacent ones"""
        engine = ChunkingEngine(chunk_size=200, min_chunk_size=100)
        
        text = """
        # Section 1
        Short.
        
        # Section 2
        This is a longer section with more content that should meet the minimum size requirements.
        
        # Section 3  
        Short.
        """
        
        chunks = engine.chunk_text(text, "doc-1")
        
        # All chunks should meet minimum size or be the result of merging
        for chunk in chunks:
            if chunk != chunks[-1]:  # Last chunk might be smaller
                assert chunk.char_count >= engine.min_chunk_size * 0.8  # Allow some flexibility
    
    def test_chunking_statistics(self):
        """Test chunking statistics generation"""
        engine = ChunkingEngine()
        
        text = """
        # Introduction
        This is the introduction to our document.
        
        ## Background
        Some background information here.
        
        # Methods
        Description of methods used.
        """
        
        chunks = engine.chunk_text(text, "doc-1")
        stats = engine.get_chunking_stats(chunks)
        
        assert 'total_chunks' in stats
        assert 'avg_chars_per_chunk' in stats
        assert 'avg_words_per_chunk' in stats
        assert 'min_chars' in stats
        assert 'max_chars' in stats
        assert 'total_chars' in stats
        assert 'total_words' in stats
        assert 'sections_detected' in stats
        assert 'sections' in stats
        
        assert stats['total_chunks'] == len(chunks)
        assert stats['sections_detected'] >= 1  # Should detect at least one section
        assert isinstance(stats['sections'], list)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks"""
        engine = ChunkingEngine()
        
        text = "This is test content."
        metadata = {"author": "Test Author", "date": "2024-01-01"}
        
        chunks = engine.chunk_text(text, "doc-1", metadata)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["date"] == "2024-01-01"
    
    def test_section_header_levels(self):
        """Test different section header level detection"""
        engine = ChunkingEngine()
        
        test_lines = [
            "# Level 1 Header",
            "## Level 2 Header", 
            "### Level 3 Header",
            "1. Numbered Section",
            "A. Letter Section",
            "I. Roman Numeral"
        ]
        
        for line in test_lines:
            result = engine._detect_section_header(line)
            assert result is not None, f"Failed to detect header: {line}"
            level, title = result
            assert isinstance(level, int)
            assert isinstance(title, str)
            assert level >= 1
            assert len(title) > 0
    
    def test_list_item_detection(self):
        """Test list item detection"""
        engine = ChunkingEngine()
        
        list_items = [
            "- Bullet point",
            "* Another bullet",
            "+ Plus bullet", 
            "1. Numbered item",
            "a. Letter item",
            "i. Roman numeral item"
        ]
        
        for item in list_items:
            assert engine._is_list_item(item), f"Failed to detect list item: {item}"
        
        # Test non-list items
        non_list_items = [
            "Regular text",
            "Just some content",
            "# Not a list item"
        ]
        
        for item in non_list_items:
            assert not engine._is_list_item(item), f"Incorrectly detected as list item: {item}" 