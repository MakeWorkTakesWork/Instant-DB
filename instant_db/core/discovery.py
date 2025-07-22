"""
Auto-discovery functionality for Instant-DB
Scans directories and automatically detects processable documents
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import magic

# Initialize libmagic for better file type detection
try:
    mime = magic.Magic(mime=True)
    MAGIC_AVAILABLE = True
except:
    MAGIC_AVAILABLE = False


@dataclass
class DocumentMetadata:
    """Enhanced metadata for discovered documents"""
    filename: str
    file_path: Path
    file_type: str
    mime_type: str
    file_size: int
    creation_date: datetime
    modification_date: datetime
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    encoding: Optional[str] = None


class DocumentDiscovery:
    """Auto-discovery system for documents"""
    
    # Supported file extensions and their categories
    SUPPORTED_EXTENSIONS = {
        # Text documents
        '.txt': 'text',
        '.md': 'markdown', 
        '.rst': 'text',
        '.rtf': 'text',
        
        # PDF documents
        '.pdf': 'pdf',
        
        # Microsoft Office
        '.docx': 'word',
        '.doc': 'word',
        '.pptx': 'powerpoint',
        '.ppt': 'powerpoint',
        '.xlsx': 'excel',
        '.xls': 'excel',
        
        # OpenDocument
        '.odt': 'opendocument',
        '.ods': 'opendocument',
        '.odp': 'opendocument',
        
        # Web documents
        '.html': 'html',
        '.htm': 'html',
        '.xml': 'xml',
        
        # Code and structured text
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.csv': 'csv',
        '.tsv': 'csv'
    }
    
    # Mime types for additional validation
    SUPPORTED_MIME_TYPES = {
        'text/plain',
        'text/markdown', 
        'text/x-rst',
        'text/rtf',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/msword',
        'application/vnd.ms-powerpoint',
        'application/vnd.ms-excel',
        'application/vnd.oasis.opendocument.text',
        'application/vnd.oasis.opendocument.spreadsheet',
        'application/vnd.oasis.opendocument.presentation',
        'text/html',
        'application/xml',
        'text/xml',
        'application/json',
        'text/x-yaml',
        'text/csv'
    }
    
    def __init__(self, include_hidden: bool = False):
        """
        Initialize document discovery
        
        Args:
            include_hidden: Whether to include hidden files/directories
        """
        self.include_hidden = include_hidden
        self._init_mime_types()
    
    def _init_mime_types(self):
        """Initialize additional mime type mappings"""
        # Add common extensions to mimetypes
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/x-rst', '.rst')
        mimetypes.add_type('text/x-yaml', '.yaml')
        mimetypes.add_type('text/x-yaml', '.yml')
    
    def scan_directory_for_documents(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size_mb: int = 100
    ) -> List[DocumentMetadata]:
        """
        Scan directory and return list of processable documents
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories
            file_extensions: Specific extensions to include (None = all supported)
            exclude_patterns: Patterns to exclude from scanning
            max_file_size_mb: Maximum file size in MB
            
        Returns:
            List of DocumentMetadata objects for discovered documents
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        discovered_docs = []
        max_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Determine extensions to look for
        target_extensions = set(file_extensions or self.SUPPORTED_EXTENSIONS.keys())
        
        # Normalize exclude patterns
        exclude_patterns = exclude_patterns or []
        exclude_patterns.extend([
            '.*',  # Hidden files (unless include_hidden=True)
            '__pycache__',
            '.git',
            '.svn',
            'node_modules',
            '.DS_Store',
            'Thumbs.db'
        ])
        
        # Walk directory tree
        if recursive:
            file_paths = directory.rglob('*')
        else:
            file_paths = directory.glob('*')
        
        for file_path in file_paths:
            if not file_path.is_file():
                continue
            
            # Skip if excluded
            if self._should_exclude(file_path, exclude_patterns):
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in target_extensions:
                continue
            
            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    continue
                    
                if file_size == 0:  # Skip empty files
                    continue
            except OSError:
                continue
            
            # Create metadata
            try:
                metadata = self._create_document_metadata(file_path)
                if metadata:
                    discovered_docs.append(metadata)
            except Exception:
                # Skip files that can't be processed
                continue
        
        return discovered_docs
    
    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded based on patterns"""
        
        # Check for hidden files
        if not self.include_hidden and any(part.startswith('.') for part in file_path.parts):
            return True
        
        # Check exclude patterns
        for pattern in exclude_patterns:
            if pattern in str(file_path) or file_path.name.startswith(pattern):
                return True
        
        return False
    
    def _create_document_metadata(self, file_path: Path) -> Optional[DocumentMetadata]:
        """Create DocumentMetadata for a file"""
        
        try:
            stat = file_path.stat()
            
            # Determine mime type
            mime_type = None
            if MAGIC_AVAILABLE:
                try:
                    mime_type = mime.from_file(str(file_path))
                except:
                    pass
            
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or 'application/octet-stream'
            
            # Validate mime type
            if mime_type not in self.SUPPORTED_MIME_TYPES and file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                return None
            
            # Determine file type
            file_extension = file_path.suffix.lower()
            file_type = self.SUPPORTED_EXTENSIONS.get(file_extension, 'unknown')
            
            # Get timestamps
            creation_date = datetime.fromtimestamp(stat.st_ctime)
            modification_date = datetime.fromtimestamp(stat.st_mtime)
            
            # Try to detect encoding for text files
            encoding = None
            if 'text' in mime_type:
                encoding = self._detect_encoding(file_path)
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=file_path,
                file_type=file_type,
                mime_type=mime_type,
                file_size=stat.st_size,
                creation_date=creation_date,
                modification_date=modification_date,
                encoding=encoding
            )
            
            return metadata
            
        except Exception:
            return None
    
    def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """Detect file encoding for text files"""
        try:
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)  # Read first 1KB
            
            # Try common encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    raw_data.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return 'utf-8'  # Default fallback
            
        except Exception:
            return None
    
    def get_discovery_summary(self, documents: List[DocumentMetadata]) -> Dict:
        """Generate summary statistics for discovered documents"""
        
        if not documents:
            return {
                'total_documents': 0,
                'total_size_mb': 0,
                'file_types': {},
                'mime_types': {},
                'largest_file': None,
                'oldest_file': None,
                'newest_file': None
            }
        
        # Calculate statistics
        total_size = sum(doc.file_size for doc in documents)
        
        # Group by type
        file_types = {}
        mime_types = {}
        
        for doc in documents:
            file_types[doc.file_type] = file_types.get(doc.file_type, 0) + 1
            mime_types[doc.mime_type] = mime_types.get(doc.mime_type, 0) + 1
        
        # Find extremes
        largest_file = max(documents, key=lambda d: d.file_size)
        oldest_file = min(documents, key=lambda d: d.creation_date)
        newest_file = max(documents, key=lambda d: d.creation_date)
        
        return {
            'total_documents': len(documents),
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': file_types,
            'mime_types': mime_types,
            'largest_file': {
                'name': largest_file.filename,
                'size_mb': largest_file.file_size / (1024 * 1024)
            },
            'oldest_file': {
                'name': oldest_file.filename,
                'date': oldest_file.creation_date.isoformat()
            },
            'newest_file': {
                'name': newest_file.filename,
                'date': newest_file.creation_date.isoformat()
            }
        }


def scan_directory_for_documents(
    directory: Union[str, Path],
    recursive: bool = True,
    file_extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_file_size_mb: int = 100
) -> List[DocumentMetadata]:
    """
    Convenience function for document discovery
    
    Args:
        directory: Directory path to scan
        recursive: Whether to scan subdirectories
        file_extensions: Specific extensions to include (None = all supported)
        exclude_patterns: Patterns to exclude from scanning
        max_file_size_mb: Maximum file size in MB
        
    Returns:
        List of DocumentMetadata objects for discovered documents
    """
    discovery = DocumentDiscovery()
    return discovery.scan_directory_for_documents(
        directory=directory,
        recursive=recursive,
        file_extensions=file_extensions,
        exclude_patterns=exclude_patterns,
        max_file_size_mb=max_file_size_mb
    ) 