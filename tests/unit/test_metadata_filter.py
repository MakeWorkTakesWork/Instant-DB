"""
Unit tests for the metadata filtering system
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path

from instant_db.core.metadata_filter import (
    MetadataFilterEngine, MetadataFilter, FilterCriteria, FilterOperator,
    parse_filter_string, create_file_type_filter, create_size_filter, create_date_filter,
    create_filter_examples
)
from instant_db.core.discovery import DocumentMetadata


@pytest.fixture
def sample_documents():
    """Create sample DocumentMetadata objects for testing"""
    base_date = datetime(2024, 1, 15, 10, 30, 0)
    
    docs = [
        DocumentMetadata(
            filename="report.pdf",
            file_path=Path("/docs/report.pdf"),
            file_type="pdf",
            mime_type="application/pdf",
            file_size=5 * 1024 * 1024,  # 5MB
            creation_date=base_date,
            modification_date=base_date + timedelta(days=1),
            author="John Doe",
            tags=["quarterly", "financial"]
        ),
        DocumentMetadata(
            filename="presentation.pptx",
            file_path=Path("/docs/presentation.pptx"),
            file_type="powerpoint",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            file_size=15 * 1024 * 1024,  # 15MB
            creation_date=base_date - timedelta(days=30),
            modification_date=base_date - timedelta(days=29),
            author="Jane Smith",
            tags=["sales", "demo"]
        ),
        DocumentMetadata(
            filename="draft_notes.txt",
            file_path=Path("/docs/draft_notes.txt"),
            file_type="text",
            mime_type="text/plain",
            file_size=500 * 1024,  # 500KB
            creation_date=base_date + timedelta(days=5),
            modification_date=base_date + timedelta(days=6),
            author="Bob Wilson",
            tags=["draft", "meeting"]
        ),
        DocumentMetadata(
            filename="analysis.docx",
            file_path=Path("/docs/analysis.docx"),
            file_type="word",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size=2 * 1024 * 1024,  # 2MB
            creation_date=base_date - timedelta(days=10),
            modification_date=base_date - timedelta(days=9),
            author="Alice Brown",
            tags=["analysis", "quarterly"]
        )
    ]
    
    return docs


class TestMetadataFilterEngine:
    """Test the MetadataFilterEngine class"""
    
    def test_filter_by_file_type(self, sample_documents):
        """Test filtering by file type"""
        engine = MetadataFilterEngine()
        
        # Create filter for PDF files only
        filter_criteria = FilterCriteria(
            field="file_type",
            operator=FilterOperator.EQUALS,
            value="pdf"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the PDF document
        assert len(result) == 1
        assert result[0].filename == "report.pdf"
        assert result[0].file_type == "pdf"
    
    def test_filter_by_file_size(self, sample_documents):
        """Test filtering by file size"""
        engine = MetadataFilterEngine()
        
        # Create filter for files larger than 10MB
        filter_criteria = FilterCriteria(
            field="file_size_mb",
            operator=FilterOperator.GREATER_THAN,
            value=10
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the presentation (15MB)
        assert len(result) == 1
        assert result[0].filename == "presentation.pptx"
        assert result[0].file_size > 10 * 1024 * 1024
    
    def test_filter_by_filename_contains(self, sample_documents):
        """Test filtering by filename contains"""
        engine = MetadataFilterEngine()
        
        # Create filter for files containing "draft"
        filter_criteria = FilterCriteria(
            field="filename",
            operator=FilterOperator.CONTAINS,
            value="draft"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the draft notes
        assert len(result) == 1
        assert result[0].filename == "draft_notes.txt"
    
    def test_filter_by_filename_starts_with(self, sample_documents):
        """Test filtering by filename starts with"""
        engine = MetadataFilterEngine()
        
        # Create filter for files starting with "report"
        filter_criteria = FilterCriteria(
            field="filename",
            operator=FilterOperator.STARTS_WITH,
            value="report"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the report file
        assert len(result) == 1
        assert result[0].filename == "report.pdf"
    
    def test_filter_by_creation_year(self, sample_documents):
        """Test filtering by creation year"""
        engine = MetadataFilterEngine()
        
        # Create filter for files created in 2024
        filter_criteria = FilterCriteria(
            field="creation_year",
            operator=FilterOperator.EQUALS,
            value=2024
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return all documents (all created in 2024)
        assert len(result) == 4
    
    def test_filter_by_age_days(self, sample_documents):
        """Test filtering by document age"""
        engine = MetadataFilterEngine()
        
        # Create filter for documents older than 20 days
        filter_criteria = FilterCriteria(
            field="age_days",
            operator=FilterOperator.GREATER_THAN,
            value=20
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return documents created more than 20 days ago
        assert len(result) >= 1  # At least the presentation
        older_docs = [doc for doc in result if "presentation" in doc.filename or "analysis" in doc.filename]
        assert len(older_docs) >= 1
    
    def test_filter_multiple_criteria_and(self, sample_documents):
        """Test filtering with multiple criteria (AND logic)"""
        engine = MetadataFilterEngine()
        
        # Create filter for PDF files larger than 1MB
        criteria = [
            FilterCriteria(
                field="file_type",
                operator=FilterOperator.EQUALS,
                value="pdf"
            ),
            FilterCriteria(
                field="file_size_mb",
                operator=FilterOperator.GREATER_THAN,
                value=1
            )
        ]
        metadata_filter = MetadataFilter(criteria=criteria, logical_operator="AND")
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the PDF file (5MB)
        assert len(result) == 1
        assert result[0].filename == "report.pdf"
        assert result[0].file_type == "pdf"
        assert result[0].file_size > 1024 * 1024
    
    def test_filter_multiple_criteria_or(self, sample_documents):
        """Test filtering with multiple criteria (OR logic)"""
        engine = MetadataFilterEngine()
        
        # Create filter for PDF files OR files larger than 10MB
        criteria = [
            FilterCriteria(
                field="file_type",
                operator=FilterOperator.EQUALS,
                value="pdf"
            ),
            FilterCriteria(
                field="file_size_mb",
                operator=FilterOperator.GREATER_THAN,
                value=10
            )
        ]
        metadata_filter = MetadataFilter(criteria=criteria, logical_operator="OR")
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return PDF file AND large files
        assert len(result) == 2  # report.pdf and presentation.pptx
        filenames = [doc.filename for doc in result]
        assert "report.pdf" in filenames
        assert "presentation.pptx" in filenames
    
    def test_filter_by_file_extension(self, sample_documents):
        """Test filtering by file extension"""
        engine = MetadataFilterEngine()
        
        # Create filter for .txt files
        filter_criteria = FilterCriteria(
            field="file_extension",
            operator=FilterOperator.EQUALS,
            value=".txt"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        # Apply filter
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return only the text file
        assert len(result) == 1
        assert result[0].filename == "draft_notes.txt"
        assert result[0].file_path.suffix == ".txt"
    
    def test_filter_case_sensitivity(self, sample_documents):
        """Test case sensitivity in filtering"""
        engine = MetadataFilterEngine()
        
        # Case-insensitive search
        filter_criteria = FilterCriteria(
            field="filename",
            operator=FilterOperator.CONTAINS,
            value="REPORT",
            case_sensitive=False
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        assert len(result) == 1
        assert result[0].filename == "report.pdf"
        
        # Case-sensitive search (should not match)
        filter_criteria.case_sensitive = True
        result = engine.apply_filter(sample_documents, metadata_filter)
        assert len(result) == 0
    
    def test_filter_in_operator(self, sample_documents):
        """Test the IN operator"""
        engine = MetadataFilterEngine()
        
        # Filter for specific file types
        filter_criteria = FilterCriteria(
            field="file_type",
            operator=FilterOperator.IN,
            value=["pdf", "text"]
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return PDF and text files
        assert len(result) == 2
        file_types = [doc.file_type for doc in result]
        assert "pdf" in file_types
        assert "text" in file_types
    
    def test_filter_regex_operator(self, sample_documents):
        """Test the REGEX operator"""
        engine = MetadataFilterEngine()
        
        # Filter for files ending with .pdf or .txt
        filter_criteria = FilterCriteria(
            field="filename",
            operator=FilterOperator.REGEX,
            value=r"\.(pdf|txt)$"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return PDF and text files
        assert len(result) == 2
        filenames = [doc.filename for doc in result]
        assert "report.pdf" in filenames
        assert "draft_notes.txt" in filenames
    
    def test_empty_filter(self, sample_documents):
        """Test filtering with empty criteria"""
        engine = MetadataFilterEngine()
        metadata_filter = MetadataFilter(criteria=[])
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should return all documents
        assert len(result) == len(sample_documents)


class TestFilterParsing:
    """Test filter string parsing"""
    
    def test_parse_simple_filter(self):
        """Test parsing simple filter syntax"""
        filter_string = "file_type:pdf"
        metadata_filter = parse_filter_string(filter_string)
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "file_type"
        assert criterion.operator == FilterOperator.EQUALS
        assert criterion.value == "pdf"
    
    def test_parse_comparison_filter(self):
        """Test parsing comparison operators"""
        filter_string = "file_size_mb>10"
        metadata_filter = parse_filter_string(filter_string)
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "file_size_mb"
        assert criterion.operator == FilterOperator.GREATER_THAN
        assert criterion.value == 10
    
    def test_parse_contains_filter(self):
        """Test parsing contains operator"""
        filter_string = "filename~report"
        metadata_filter = parse_filter_string(filter_string)
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "filename"
        assert criterion.operator == FilterOperator.CONTAINS
        assert criterion.value == "report"
    
    def test_parse_json_filter_single(self):
        """Test parsing JSON format for single criterion"""
        filter_dict = {
            "field": "file_type",
            "operator": "eq",
            "value": "pdf",
            "case_sensitive": True
        }
        filter_string = json.dumps(filter_dict)
        metadata_filter = parse_filter_string(filter_string)
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "file_type"
        assert criterion.operator == FilterOperator.EQUALS
        assert criterion.value == "pdf"
        assert criterion.case_sensitive == True
    
    def test_parse_json_filter_multiple(self):
        """Test parsing JSON format for multiple criteria"""
        filter_list = [
            {"field": "file_type", "operator": "eq", "value": "pdf"},
            {"field": "file_size_mb", "operator": "gt", "value": 5}
        ]
        filter_string = json.dumps(filter_list)
        metadata_filter = parse_filter_string(filter_string)
        
        assert len(metadata_filter.criteria) == 2
        
        criterion1 = metadata_filter.criteria[0]
        assert criterion1.field == "file_type"
        assert criterion1.operator == FilterOperator.EQUALS
        assert criterion1.value == "pdf"
        
        criterion2 = metadata_filter.criteria[1]
        assert criterion2.field == "file_size_mb"
        assert criterion2.operator == FilterOperator.GREATER_THAN
        assert criterion2.value == 5
    
    def test_parse_empty_filter(self):
        """Test parsing empty filter string"""
        metadata_filter = parse_filter_string("")
        assert len(metadata_filter.criteria) == 0
        
        metadata_filter = parse_filter_string("   ")
        assert len(metadata_filter.criteria) == 0
    
    def test_parse_invalid_filter(self):
        """Test parsing invalid filter string"""
        with pytest.raises(ValueError):
            parse_filter_string("invalid_filter_format")
        
        with pytest.raises(ValueError):
            parse_filter_string("field_without_operator")


class TestConvenienceFunctions:
    """Test convenience functions for creating filters"""
    
    def test_create_file_type_filter_single(self):
        """Test creating file type filter for single type"""
        metadata_filter = create_file_type_filter("pdf")
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "file_type"
        assert criterion.operator == FilterOperator.EQUALS
        assert criterion.value == "pdf"
    
    def test_create_file_type_filter_multiple(self):
        """Test creating file type filter for multiple types"""
        metadata_filter = create_file_type_filter(["pdf", "word"])
        
        assert len(metadata_filter.criteria) == 1
        criterion = metadata_filter.criteria[0]
        assert criterion.field == "file_type"
        assert criterion.operator == FilterOperator.IN
        assert criterion.value == ["pdf", "word"]
    
    def test_create_size_filter(self):
        """Test creating size filter"""
        metadata_filter = create_size_filter(min_mb=1, max_mb=10)
        
        assert len(metadata_filter.criteria) == 2
        
        # Check min size criterion
        min_criterion = metadata_filter.criteria[0]
        assert min_criterion.field == "file_size_mb"
        assert min_criterion.operator == FilterOperator.GREATER_EQUAL
        assert min_criterion.value == 1
        
        # Check max size criterion
        max_criterion = metadata_filter.criteria[1]
        assert max_criterion.field == "file_size_mb"
        assert max_criterion.operator == FilterOperator.LESS_EQUAL
        assert max_criterion.value == 10
    
    def test_create_date_filter(self):
        """Test creating date filter"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        metadata_filter = create_date_filter(after_date=start_date, before_date=end_date)
        
        assert len(metadata_filter.criteria) == 2
        
        # Check after date criterion
        after_criterion = metadata_filter.criteria[0]
        assert after_criterion.field == "creation_date"
        assert after_criterion.operator == FilterOperator.GREATER_EQUAL
        assert after_criterion.value == start_date
        
        # Check before date criterion
        before_criterion = metadata_filter.criteria[1]
        assert before_criterion.field == "creation_date"
        assert before_criterion.operator == FilterOperator.LESS_EQUAL
        assert before_criterion.value == end_date


class TestFilterExamples:
    """Test filter examples functionality"""
    
    def test_create_filter_examples(self):
        """Test that filter examples are created correctly"""
        examples = create_filter_examples()
        
        assert isinstance(examples, dict)
        assert len(examples) > 0
        
        # Check that some expected examples exist
        assert any("PDF" in key for key in examples.keys())
        assert any("large" in key.lower() for key in examples.keys())
        assert any("file_type:pdf" in value for value in examples.values())
    
    def test_filter_examples_are_valid(self):
        """Test that all filter examples can be parsed"""
        examples = create_filter_examples()
        
        for description, filter_string in examples.items():
            try:
                metadata_filter = parse_filter_string(filter_string)
                assert isinstance(metadata_filter, MetadataFilter)
            except Exception as e:
                pytest.fail(f"Filter example '{description}' failed to parse: {e}")


class TestIntegrationScenarios:
    """Test realistic filtering scenarios"""
    
    def test_quarterly_reports_scenario(self, sample_documents):
        """Test filtering for quarterly reports"""
        engine = MetadataFilterEngine()
        
        # Filter for documents containing "quarterly" in tags
        filter_criteria = FilterCriteria(
            field="tags",
            operator=FilterOperator.CONTAINS,
            value="quarterly"
        )
        metadata_filter = MetadataFilter(criteria=[filter_criteria])
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should find documents tagged with "quarterly"
        assert len(result) >= 1
        for doc in result:
            assert "quarterly" in doc.tags
    
    def test_recent_large_files_scenario(self, sample_documents):
        """Test filtering for recent large files"""
        engine = MetadataFilterEngine()
        
        # Filter for files larger than 1MB created in the last 20 days
        criteria = [
            FilterCriteria(
                field="file_size_mb",
                operator=FilterOperator.GREATER_THAN,
                value=1
            ),
            FilterCriteria(
                field="age_days",
                operator=FilterOperator.LESS_THAN,
                value=20
            )
        ]
        metadata_filter = MetadataFilter(criteria=criteria, logical_operator="AND")
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should find recent large files
        assert len(result) >= 1
        for doc in result:
            assert doc.file_size > 1024 * 1024  # > 1MB
    
    def test_draft_documents_scenario(self, sample_documents):
        """Test filtering for draft documents"""
        engine = MetadataFilterEngine()
        
        # Filter for files with "draft" in filename OR tags
        criteria = [
            FilterCriteria(
                field="filename",
                operator=FilterOperator.CONTAINS,
                value="draft"
            ),
            FilterCriteria(
                field="tags",
                operator=FilterOperator.CONTAINS,
                value="draft"
            )
        ]
        metadata_filter = MetadataFilter(criteria=criteria, logical_operator="OR")
        
        result = engine.apply_filter(sample_documents, metadata_filter)
        
        # Should find draft documents
        assert len(result) >= 1
        for doc in result:
            assert "draft" in doc.filename.lower() or "draft" in doc.tags 