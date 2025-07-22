"""
Advanced metadata filtering system for Instant-DB
Enables powerful document filtering based on metadata criteria
"""

import re
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .discovery import DocumentMetadata


class FilterOperator(Enum):
    """Operators for metadata filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class FilterCriteria:
    """Single filter criterion"""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class MetadataFilter:
    """Collection of filter criteria with logical operations"""
    criteria: List[FilterCriteria] = field(default_factory=list)
    logical_operator: str = "AND"  # "AND" or "OR"


class MetadataFilterEngine:
    """Engine for applying metadata filters to document collections"""
    
    def __init__(self):
        """Initialize the filter engine"""
        self.operator_functions = {
            FilterOperator.EQUALS: self._equals,
            FilterOperator.NOT_EQUALS: self._not_equals,
            FilterOperator.CONTAINS: self._contains,
            FilterOperator.NOT_CONTAINS: self._not_contains,
            FilterOperator.STARTS_WITH: self._starts_with,
            FilterOperator.ENDS_WITH: self._ends_with,
            FilterOperator.GREATER_THAN: self._greater_than,
            FilterOperator.LESS_THAN: self._less_than,
            FilterOperator.GREATER_EQUAL: self._greater_equal,
            FilterOperator.LESS_EQUAL: self._less_equal,
            FilterOperator.IN: self._in,
            FilterOperator.NOT_IN: self._not_in,
            FilterOperator.REGEX: self._regex,
            FilterOperator.EXISTS: self._exists,
            FilterOperator.NOT_EXISTS: self._not_exists,
        }
    
    def apply_filter(
        self,
        documents: List[DocumentMetadata],
        metadata_filter: MetadataFilter
    ) -> List[DocumentMetadata]:
        """
        Apply metadata filter to a list of documents
        
        Args:
            documents: List of documents to filter
            metadata_filter: Filter criteria to apply
            
        Returns:
            Filtered list of documents
        """
        if not metadata_filter.criteria:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            if self._document_matches_filter(doc, metadata_filter):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _document_matches_filter(
        self,
        doc: DocumentMetadata,
        metadata_filter: MetadataFilter
    ) -> bool:
        """Check if a document matches the filter criteria"""
        
        results = []
        
        for criterion in metadata_filter.criteria:
            result = self._evaluate_criterion(doc, criterion)
            results.append(result)
        
        # Apply logical operator
        if metadata_filter.logical_operator.upper() == "OR":
            return any(results)
        else:  # Default to AND
            return all(results)
    
    def _evaluate_criterion(
        self,
        doc: DocumentMetadata,
        criterion: FilterCriteria
    ) -> bool:
        """Evaluate a single filter criterion against a document"""
        
        # Get the field value from the document
        field_value = self._get_field_value(doc, criterion.field)
        
        # Get the operator function
        operator_func = self.operator_functions.get(criterion.operator)
        if not operator_func:
            raise ValueError(f"Unsupported operator: {criterion.operator}")
        
        # Apply the operator
        return operator_func(field_value, criterion.value, criterion.case_sensitive)
    
    def _get_field_value(self, doc: DocumentMetadata, field: str) -> Any:
        """Get the value of a field from a document"""
        
        # Handle nested field access (e.g., "file_path.name")
        if "." in field:
            parts = field.split(".")
            value = doc
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
            return value
        
        # Handle direct field access
        if hasattr(doc, field):
            return getattr(doc, field)
        
        # Handle special computed fields
        if field == "file_extension":
            return doc.file_path.suffix.lower() if doc.file_path else None
        elif field == "file_size_mb":
            return doc.file_size / (1024 * 1024) if doc.file_size else None
        elif field == "creation_year":
            return doc.creation_date.year if doc.creation_date else None
        elif field == "creation_month":
            return doc.creation_date.month if doc.creation_date else None
        elif field == "modification_year":
            return doc.modification_date.year if doc.modification_date else None
        elif field == "age_days":
            if doc.creation_date:
                return (datetime.now() - doc.creation_date).days
            return None
        
        return None
    
    # Operator implementation methods
    def _equals(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return filter_value is None
        
        if isinstance(field_value, str) and isinstance(filter_value, str):
            if not case_sensitive:
                return field_value.lower() == filter_value.lower()
        
        return field_value == filter_value
    
    def _not_equals(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        return not self._equals(field_value, filter_value, case_sensitive)
    
    def _contains(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        field_str = str(field_value)
        filter_str = str(filter_value)
        
        if not case_sensitive:
            return filter_str.lower() in field_str.lower()
        
        return filter_str in field_str
    
    def _not_contains(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        return not self._contains(field_value, filter_value, case_sensitive)
    
    def _starts_with(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        field_str = str(field_value)
        filter_str = str(filter_value)
        
        if not case_sensitive:
            return field_str.lower().startswith(filter_str.lower())
        
        return field_str.startswith(filter_str)
    
    def _ends_with(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        field_str = str(field_value)
        filter_str = str(filter_value)
        
        if not case_sensitive:
            return field_str.lower().endswith(filter_str.lower())
        
        return field_str.endswith(filter_str)
    
    def _greater_than(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        try:
            return field_value > filter_value
        except TypeError:
            return False
    
    def _less_than(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        try:
            return field_value < filter_value
        except TypeError:
            return False
    
    def _greater_equal(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        try:
            return field_value >= filter_value
        except TypeError:
            return False
    
    def _less_equal(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        try:
            return field_value <= filter_value
        except TypeError:
            return False
    
    def _in(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        if not isinstance(filter_value, (list, tuple, set)):
            filter_value = [filter_value]
        
        if isinstance(field_value, str) and not case_sensitive:
            field_value = field_value.lower()
            filter_value = [str(v).lower() for v in filter_value]
        
        return field_value in filter_value
    
    def _not_in(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        return not self._in(field_value, filter_value, case_sensitive)
    
    def _regex(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        if field_value is None:
            return False
        
        field_str = str(field_value)
        pattern = str(filter_value)
        
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            return bool(re.search(pattern, field_str, flags))
        except re.error:
            return False
    
    def _exists(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        return field_value is not None
    
    def _not_exists(self, field_value: Any, filter_value: Any, case_sensitive: bool) -> bool:
        return field_value is None


def parse_filter_string(filter_string: str) -> MetadataFilter:
    """
    Parse a filter string into MetadataFilter object
    
    Supports JSON format and simplified syntax:
    
    JSON format:
    '{"field": "file_type", "operator": "eq", "value": "pdf"}'
    '[{"field": "file_type", "operator": "eq", "value": "pdf"}, {"field": "file_size_mb", "operator": "lt", "value": 10}]'
    
    Simplified syntax:
    'file_type:pdf'
    'file_size_mb<10'
    'filename~report'
    
    Args:
        filter_string: String representation of filter
        
    Returns:
        MetadataFilter object
    """
    
    if not filter_string.strip():
        return MetadataFilter()
    
    # Try JSON format first
    try:
        filter_data = json.loads(filter_string)
        
        if isinstance(filter_data, dict):
            # Single criterion
            criterion = _parse_criterion_dict(filter_data)
            return MetadataFilter(criteria=[criterion])
        
        elif isinstance(filter_data, list):
            # Multiple criteria
            criteria = [_parse_criterion_dict(item) for item in filter_data]
            return MetadataFilter(criteria=criteria)
        
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    # Try simplified syntax
    try:
        return _parse_simplified_filter(filter_string)
    except ValueError:
        raise ValueError(f"Invalid filter format: {filter_string}")


def _parse_criterion_dict(data: Dict[str, Any]) -> FilterCriteria:
    """Parse a dictionary into FilterCriteria"""
    
    field = data["field"]
    operator_str = data["operator"]
    value = data["value"]
    case_sensitive = data.get("case_sensitive", False)
    
    # Convert operator string to enum
    operator = FilterOperator(operator_str)
    
    return FilterCriteria(
        field=field,
        operator=operator,
        value=value,
        case_sensitive=case_sensitive
    )


def _parse_simplified_filter(filter_string: str) -> MetadataFilter:
    """Parse simplified filter syntax"""
    
    # Map of operators to FilterOperator enums
    operator_map = {
        ":": FilterOperator.EQUALS,
        "=": FilterOperator.EQUALS,
        "!=": FilterOperator.NOT_EQUALS,
        "~": FilterOperator.CONTAINS,
        "!~": FilterOperator.NOT_CONTAINS,
        "^": FilterOperator.STARTS_WITH,
        "$": FilterOperator.ENDS_WITH,
        ">": FilterOperator.GREATER_THAN,
        "<": FilterOperator.LESS_THAN,
        ">=": FilterOperator.GREATER_EQUAL,
        "<=": FilterOperator.LESS_EQUAL,
    }
    
    # Find the operator
    for op_str, op_enum in operator_map.items():
        if op_str in filter_string:
            field, value = filter_string.split(op_str, 1)
            field = field.strip()
            value = value.strip()
            
            # Try to convert value to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            
            criterion = FilterCriteria(
                field=field,
                operator=op_enum,
                value=value
            )
            
            return MetadataFilter(criteria=[criterion])
    
    raise ValueError(f"No valid operator found in filter: {filter_string}")


def create_filter_examples() -> Dict[str, str]:
    """Create examples of filter usage for documentation"""
    
    examples = {
        "PDF documents only": 'file_type:pdf',
        "Large files (>10MB)": 'file_size_mb>10',
        "Recent documents (created this year)": f'creation_year:{datetime.now().year}',
        "Documents containing 'report'": 'filename~report',
        "Word documents from last month": '[{"field": "file_type", "operator": "eq", "value": "word"}, {"field": "creation_month", "operator": "eq", "value": 12}]',
        "Small text files": '[{"field": "file_type", "operator": "eq", "value": "text"}, {"field": "file_size_mb", "operator": "lt", "value": 1}]',
        "Documents with specific extensions": 'file_extension:.docx',
        "Old documents (>365 days)": 'age_days>365',
        "PowerPoint presentations": 'file_type:powerpoint',
        "Documents starting with 'draft'": 'filename^draft'
    }
    
    return examples


# Convenience functions for common filters
def create_file_type_filter(file_types: Union[str, List[str]]) -> MetadataFilter:
    """Create a filter for specific file types"""
    if isinstance(file_types, str):
        file_types = [file_types]
    
    if len(file_types) == 1:
        criterion = FilterCriteria(
            field="file_type",
            operator=FilterOperator.EQUALS,
            value=file_types[0]
        )
    else:
        criterion = FilterCriteria(
            field="file_type",
            operator=FilterOperator.IN,
            value=file_types
        )
    
    return MetadataFilter(criteria=[criterion])


def create_size_filter(min_mb: Optional[float] = None, max_mb: Optional[float] = None) -> MetadataFilter:
    """Create a filter for file size range"""
    criteria = []
    
    if min_mb is not None:
        criteria.append(FilterCriteria(
            field="file_size_mb",
            operator=FilterOperator.GREATER_EQUAL,
            value=min_mb
        ))
    
    if max_mb is not None:
        criteria.append(FilterCriteria(
            field="file_size_mb",
            operator=FilterOperator.LESS_EQUAL,
            value=max_mb
        ))
    
    return MetadataFilter(criteria=criteria)


def create_date_filter(
    after_date: Optional[datetime] = None,
    before_date: Optional[datetime] = None,
    field: str = "creation_date"
) -> MetadataFilter:
    """Create a filter for date range"""
    criteria = []
    
    if after_date is not None:
        criteria.append(FilterCriteria(
            field=field,
            operator=FilterOperator.GREATER_EQUAL,
            value=after_date
        ))
    
    if before_date is not None:
        criteria.append(FilterCriteria(
            field=field,
            operator=FilterOperator.LESS_EQUAL,
            value=before_date
        ))
    
    return MetadataFilter(criteria=criteria) 