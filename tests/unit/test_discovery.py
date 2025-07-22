"""
Unit tests for the auto-discovery functionality
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from instant_db.core.discovery import DocumentDiscovery, DocumentMetadata, scan_directory_for_documents


class TestDocumentDiscovery:
    """Test cases for DocumentDiscovery class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test files"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        (temp_dir / "document.pdf").write_text("PDF content")
        (temp_dir / "presentation.pptx").write_text("PowerPoint content")
        (temp_dir / "spreadsheet.xlsx").write_text("Excel content")
        (temp_dir / "text_file.txt").write_text("Text content")
        (temp_dir / "markdown.md").write_text("# Markdown content")
        (temp_dir / "webpage.html").write_text("<html><body>HTML content</body></html>")
        
        # Create unsupported files
        (temp_dir / "image.jpg").write_bytes(b"fake image data")
        (temp_dir / "archive.zip").write_bytes(b"fake zip data")
        (temp_dir / "no_extension").write_text("File without extension")
        
        # Create subdirectory
        subdir = temp_dir / "subdirectory"
        subdir.mkdir()
        (subdir / "nested_document.docx").write_text("Word content")
        
        # Create hidden file
        (temp_dir / ".hidden_file.txt").write_text("Hidden content")
        
        # Create large file (simulate)
        large_file = temp_dir / "large_file.txt"
        large_file.write_text("x" * 1000)  # 1KB file
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_discovery_basic(self, temp_dir):
        """Test basic document discovery"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(temp_dir)
        
        # Should find supported document types
        filenames = [doc.filename for doc in docs]
        expected_files = [
            "document.pdf", "presentation.pptx", "spreadsheet.xlsx", 
            "text_file.txt", "markdown.md", "webpage.html", "nested_document.docx"
        ]
        
        for expected in expected_files:
            assert expected in filenames, f"Expected to find {expected}"
        
        # Should not find unsupported files
        unsupported_files = ["image.jpg", "archive.zip", "no_extension"]
        for unsupported in unsupported_files:
            assert unsupported not in filenames, f"Should not find {unsupported}"
    
    def test_discovery_non_recursive(self, temp_dir):
        """Test non-recursive discovery"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(temp_dir, recursive=False)
        
        filenames = [doc.filename for doc in docs]
        
        # Should not find nested files
        assert "nested_document.docx" not in filenames
        
        # Should find files in root directory
        assert "document.pdf" in filenames
        assert "text_file.txt" in filenames
    
    def test_discovery_with_extensions_filter(self, temp_dir):
        """Test discovery with specific file extensions"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(
            temp_dir, 
            file_extensions=['.pdf', '.txt']
        )
        
        filenames = [doc.filename for doc in docs]
        
        # Should only find PDF and TXT files
        assert "document.pdf" in filenames
        assert "text_file.txt" in filenames
        
        # Should not find other types
        assert "presentation.pptx" not in filenames
        assert "markdown.md" not in filenames
    
    def test_discovery_with_exclude_patterns(self, temp_dir):
        """Test discovery with exclude patterns"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(
            temp_dir,
            exclude_patterns=["large_", "presentation"]
        )
        
        filenames = [doc.filename for doc in docs]
        
        # Should exclude files matching patterns
        assert "large_file.txt" not in filenames
        assert "presentation.pptx" not in filenames
        
        # Should include other files
        assert "document.pdf" in filenames
        assert "text_file.txt" in filenames
    
    def test_discovery_file_size_limit(self, temp_dir):
        """Test discovery with file size limits"""
        # Create a file that's "too large" for testing
        large_file = temp_dir / "very_large.txt"
        large_file.write_text("x" * 2000)  # 2KB file
        
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(
            temp_dir,
            max_file_size_mb=0.001  # 1KB limit
        )
        
        filenames = [doc.filename for doc in docs]
        
        # Should exclude the large file
        assert "very_large.txt" not in filenames
        
        # Should include smaller files
        assert "text_file.txt" in filenames
    
    def test_discovery_include_hidden(self, temp_dir):
        """Test discovery with hidden files included"""
        discovery = DocumentDiscovery(include_hidden=True)
        docs = discovery.scan_directory_for_documents(temp_dir)
        
        filenames = [doc.filename for doc in docs]
        
        # Should include hidden files when enabled
        assert ".hidden_file.txt" in filenames
    
    def test_discovery_exclude_hidden(self, temp_dir):
        """Test discovery with hidden files excluded (default)"""
        discovery = DocumentDiscovery(include_hidden=False)
        docs = discovery.scan_directory_for_documents(temp_dir)
        
        filenames = [doc.filename for doc in docs]
        
        # Should exclude hidden files by default
        assert ".hidden_file.txt" not in filenames
    
    def test_document_metadata_creation(self, temp_dir):
        """Test DocumentMetadata creation and content"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(temp_dir)
        
        # Find a specific document
        pdf_doc = next(doc for doc in docs if doc.filename == "document.pdf")
        
        # Check metadata fields
        assert pdf_doc.filename == "document.pdf"
        assert pdf_doc.file_type == "pdf"
        assert pdf_doc.file_path == temp_dir / "document.pdf"
        assert pdf_doc.file_size > 0
        assert isinstance(pdf_doc.creation_date, datetime)
        assert isinstance(pdf_doc.modification_date, datetime)
        assert pdf_doc.mime_type is not None
    
    def test_discovery_summary(self, temp_dir):
        """Test discovery summary generation"""
        discovery = DocumentDiscovery()
        docs = discovery.scan_directory_for_documents(temp_dir)
        summary = discovery.get_discovery_summary(docs)
        
        # Check summary structure
        assert "total_documents" in summary
        assert "total_size_mb" in summary
        assert "file_types" in summary
        assert "mime_types" in summary
        assert "largest_file" in summary
        assert "oldest_file" in summary
        assert "newest_file" in summary
        
        # Check summary content
        assert summary["total_documents"] > 0
        assert summary["total_size_mb"] > 0
        assert len(summary["file_types"]) > 0
    
    def test_convenience_function(self, temp_dir):
        """Test the convenience function"""
        docs = scan_directory_for_documents(temp_dir)
        
        # Should work the same as the class method
        assert len(docs) > 0
        assert all(isinstance(doc, DocumentMetadata) for doc in docs)
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directory"""
        discovery = DocumentDiscovery()
        
        with pytest.raises(FileNotFoundError):
            discovery.scan_directory_for_documents("/nonexistent/path")
    
    def test_file_instead_of_directory(self, temp_dir):
        """Test handling when path is a file, not directory"""
        discovery = DocumentDiscovery()
        file_path = temp_dir / "document.pdf"
        
        with pytest.raises(ValueError):
            discovery.scan_directory_for_documents(file_path)
    
    def test_empty_directory(self):
        """Test discovery in empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            discovery = DocumentDiscovery()
            docs = discovery.scan_directory_for_documents(temp_dir)
            
            assert len(docs) == 0
            
            # Test summary with empty results
            summary = discovery.get_discovery_summary(docs)
            assert summary["total_documents"] == 0
            assert summary["total_size_mb"] == 0
            assert summary["largest_file"] is None


class TestSupportedFormats:
    """Test supported file format detection"""
    
    def test_supported_extensions(self):
        """Test that all expected extensions are supported"""
        discovery = DocumentDiscovery()
        
        expected_extensions = {
            '.txt', '.md', '.rst', '.rtf',  # Text
            '.pdf',  # PDF
            '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',  # Office
            '.odt', '.ods', '.odp',  # OpenDocument
            '.html', '.htm', '.xml',  # Web
            '.json', '.yaml', '.yml', '.csv', '.tsv'  # Structured
        }
        
        supported = set(discovery.SUPPORTED_EXTENSIONS.keys())
        
        for ext in expected_extensions:
            assert ext in supported, f"Extension {ext} should be supported"
    
    def test_mime_type_validation(self):
        """Test MIME type validation"""
        discovery = DocumentDiscovery()
        
        # Should have corresponding MIME types for major formats
        assert 'application/pdf' in discovery.SUPPORTED_MIME_TYPES
        assert 'text/plain' in discovery.SUPPORTED_MIME_TYPES
        assert 'text/html' in discovery.SUPPORTED_MIME_TYPES 