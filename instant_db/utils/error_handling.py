"""
Enhanced error handling and recovery patterns for Instant-DB
Provides graceful error recovery for corrupted files and unsupported formats
"""

import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger


class ErrorType(Enum):
    """Types of errors that can occur during processing"""
    CORRUPTED_FILE = "corrupted_file"
    UNSUPPORTED_FORMAT = "unsupported_format"
    PERMISSION_DENIED = "permission_denied"
    FILE_NOT_FOUND = "file_not_found"
    FILE_TOO_LARGE = "file_too_large"
    ENCODING_ERROR = "encoding_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ProcessingError:
    """Structured error information"""
    error_type: ErrorType
    file_path: Optional[Path]
    error_message: str
    original_exception: Optional[Exception] = None
    recovery_suggestion: Optional[str] = None
    is_recoverable: bool = True
    context: Dict[str, Any] = None


class CorruptedFileError(Exception):
    """Raised when a file is corrupted or malformed"""
    pass


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported"""
    pass


class FileTooLargeError(Exception):
    """Raised when file exceeds size limits"""
    pass


class ProcessingTimeoutError(Exception):
    """Raised when processing times out"""
    pass


class ErrorRecoveryHandler:
    """Handles errors gracefully with recovery strategies"""
    
    def __init__(self, skip_errors: bool = True, max_retries: int = 3):
        """
        Initialize error recovery handler
        
        Args:
            skip_errors: Whether to skip errors and continue processing
            max_retries: Maximum number of retry attempts
        """
        self.skip_errors = skip_errors
        self.max_retries = max_retries
        self.logger = get_logger()
        self.error_log: List[ProcessingError] = []
        
        # Error type mappings for better categorization
        self.error_mappings = {
            FileNotFoundError: ErrorType.FILE_NOT_FOUND,
            PermissionError: ErrorType.PERMISSION_DENIED,
            CorruptedFileError: ErrorType.CORRUPTED_FILE,
            UnsupportedFormatError: ErrorType.UNSUPPORTED_FORMAT,
            FileTooLargeError: ErrorType.FILE_TOO_LARGE,
            UnicodeDecodeError: ErrorType.ENCODING_ERROR,
            MemoryError: ErrorType.MEMORY_ERROR,
            ProcessingTimeoutError: ErrorType.TIMEOUT_ERROR,
            ConnectionError: ErrorType.NETWORK_ERROR,
            TimeoutError: ErrorType.NETWORK_ERROR,
        }
    
    def handle_error(
        self,
        error: Exception,
        file_path: Optional[Path] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingError:
        """
        Handle an error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            file_path: Path to the file being processed
            context: Additional context information
            
        Returns:
            ProcessingError object with details and recovery suggestions
        """
        # Determine error type
        error_type = self.error_mappings.get(type(error), ErrorType.UNKNOWN_ERROR)
        
        # Create processing error
        processing_error = ProcessingError(
            error_type=error_type,
            file_path=file_path,
            error_message=str(error),
            original_exception=error,
            context=context or {}
        )
        
        # Add recovery suggestions
        processing_error.recovery_suggestion = self._get_recovery_suggestion(error_type, file_path)
        processing_error.is_recoverable = self._is_recoverable(error_type)
        
        # Log the error
        self._log_error(processing_error)
        
        # Add to error log
        self.error_log.append(processing_error)
        
        return processing_error
    
    def safe_process_file(
        self,
        processor_func: Callable,
        file_path: Path,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Safely process a file with error recovery
        
        Args:
            processor_func: Function to process the file
            file_path: Path to the file
            **kwargs: Additional arguments for processor_func
            
        Returns:
            Processing result or None if failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                return processor_func(file_path, **kwargs)
                
            except Exception as e:
                processing_error = self.handle_error(e, file_path)
                
                # If not recoverable or on final attempt, handle based on skip_errors setting
                if not processing_error.is_recoverable or attempt >= self.max_retries:
                    if self.skip_errors:
                        self.logger.warning(f"⚠️  Skipping {file_path.name}: {processing_error.error_message}")
                        if processing_error.recovery_suggestion:
                            self.logger.info(f"💡 Suggestion: {processing_error.recovery_suggestion}")
                        return None
                    else:
                        self.logger.error(f"❌ Failed to process {file_path.name}: {processing_error.error_message}")
                        raise e
                
                # Wait before retry for certain error types
                if error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
                    import time
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10s
                
                self.logger.debug(f"🔄 Retrying {file_path.name} (attempt {attempt + 2}/{self.max_retries + 1})")
        
        return None
    
    def _get_recovery_suggestion(self, error_type: ErrorType, file_path: Optional[Path] = None) -> str:
        """Get recovery suggestion based on error type"""
        suggestions = {
            ErrorType.CORRUPTED_FILE: "File may be corrupted. Try re-downloading or using a different version.",
            ErrorType.UNSUPPORTED_FORMAT: f"File format not supported. Supported types: .pdf, .docx, .txt, .md, .html",
            ErrorType.PERMISSION_DENIED: "Check file permissions or run with appropriate privileges.",
            ErrorType.FILE_NOT_FOUND: "Verify the file path exists and is accessible.",
            ErrorType.FILE_TOO_LARGE: "Consider splitting the file or increasing the --max-file-size limit.",
            ErrorType.ENCODING_ERROR: "File may have unusual encoding. Try converting to UTF-8.",
            ErrorType.NETWORK_ERROR: "Check internet connection and retry.",
            ErrorType.PROCESSING_ERROR: "File content may be complex. Try processing smaller sections.",
            ErrorType.MEMORY_ERROR: "File too large for available memory. Close other applications or use smaller files.",
            ErrorType.TIMEOUT_ERROR: "Processing timed out. Try smaller files or increase timeout limits.",
            ErrorType.UNKNOWN_ERROR: "Unexpected error occurred. Check logs for details."
        }
        
        suggestion = suggestions.get(error_type, "No specific suggestion available.")
        
        # Add file-specific suggestions
        if file_path:
            if error_type == ErrorType.UNSUPPORTED_FORMAT:
                ext = file_path.suffix.lower()
                if ext in ['.exe', '.zip', '.rar']:
                    suggestion += " This appears to be a binary/archive file."
                elif ext == '':
                    suggestion += " File has no extension - add appropriate extension."
        
        return suggestion
    
    def _is_recoverable(self, error_type: ErrorType) -> bool:
        """Determine if error type is recoverable with retry"""
        recoverable_errors = {
            ErrorType.NETWORK_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.PROCESSING_ERROR
        }
        return error_type in recoverable_errors
    
    def _log_error(self, error: ProcessingError):
        """Log error with appropriate level"""
        file_name = error.file_path.name if error.file_path else "Unknown file"
        
        if error.error_type in [ErrorType.CORRUPTED_FILE, ErrorType.UNSUPPORTED_FORMAT]:
            # These are common and expected - log as warning
            self.logger.warning(f"⚠️  {error.error_type.value}: {file_name} - {error.error_message}")
        elif error.error_type == ErrorType.PERMISSION_DENIED:
            self.logger.warning(f"🔒 {error.error_type.value}: {file_name} - {error.error_message}")
        elif error.error_type in [ErrorType.MEMORY_ERROR, ErrorType.FILE_TOO_LARGE]:
            self.logger.warning(f"💾 {error.error_type.value}: {file_name} - {error.error_message}")
        else:
            # More serious errors - log as error
            self.logger.error(f"❌ {error.error_type.value}: {file_name} - {error.error_message}")
            if error.original_exception:
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        if not self.error_log:
            return {"total_errors": 0, "error_types": {}, "recoverable_errors": 0}
        
        error_types = {}
        recoverable_count = 0
        
        for error in self.error_log:
            error_type_name = error.error_type.value
            error_types[error_type_name] = error_types.get(error_type_name, 0) + 1
            
            if error.is_recoverable:
                recoverable_count += 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recoverable_errors": recoverable_count,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def clear_errors(self):
        """Clear the error log"""
        self.error_log.clear()


def create_safe_processor(
    processor_func: Callable,
    skip_errors: bool = True,
    max_retries: int = 3
) -> Callable:
    """
    Create a safe version of a processor function with error handling
    
    Args:
        processor_func: Function to wrap with error handling
        skip_errors: Whether to skip errors
        max_retries: Maximum retry attempts
        
    Returns:
        Wrapped function with error handling
    """
    error_handler = ErrorRecoveryHandler(skip_errors=skip_errors, max_retries=max_retries)
    
    def safe_wrapper(file_path: Path, **kwargs):
        return error_handler.safe_process_file(processor_func, file_path, **kwargs)
    
    safe_wrapper.error_handler = error_handler
    return safe_wrapper


# Pre-built error patterns for common file processing issues
def validate_file_for_processing(file_path: Path, max_size_mb: int = 100) -> Optional[ProcessingError]:
    """
    Validate file before processing and return error if invalid
    
    Args:
        file_path: Path to validate
        max_size_mb: Maximum file size in MB
        
    Returns:
        ProcessingError if invalid, None if valid
    """
    error_handler = ErrorRecoveryHandler()
    
    # Check if file exists
    if not file_path.exists():
        return error_handler.handle_error(FileNotFoundError(f"File not found: {file_path}"), file_path)
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1)
    except PermissionError as e:
        return error_handler.handle_error(e, file_path)
    except Exception as e:
        return error_handler.handle_error(CorruptedFileError(f"Cannot read file: {e}"), file_path)
    
    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size_mb * 1024 * 1024:
            return error_handler.handle_error(
                FileTooLargeError(f"File size {file_size / (1024*1024):.1f}MB exceeds limit {max_size_mb}MB"), 
                file_path
            )
    except Exception as e:
        return error_handler.handle_error(e, file_path)
    
    return None  # File is valid 