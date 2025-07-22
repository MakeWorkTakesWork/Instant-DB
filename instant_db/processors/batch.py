"""
Batch document processor for Instant-DB
Handles processing multiple documents in directories
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .document import DocumentProcessor


class BatchProcessor:
    """
    Process multiple documents in batch for Instant-DB
    Supports directory traversal and parallel processing
    """
    
    def __init__(self, 
                 document_processor: Optional[DocumentProcessor] = None,
                 max_workers: int = 4,
                 skip_errors: bool = True):
        """
        Initialize batch processor
        
        Args:
            document_processor: DocumentProcessor instance to use
            max_workers: Maximum number of worker threads
            skip_errors: Whether to skip files that cause errors
        """
        self.document_processor = document_processor or DocumentProcessor()
        self.max_workers = max_workers
        self.skip_errors = skip_errors
        
        # Thread-safe counters
        self._lock = threading.Lock()
        self._processed_count = 0
        self._error_count = 0
        self._total_files = 0
    
    def process_directory(self, 
                         input_dir: Union[str, Path],
                         output_dir: Optional[str] = None,
                         file_extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process all documents in a directory
        
        Args:
            input_dir: Input directory containing documents
            output_dir: Output directory for database
            file_extensions: File extensions to process (e.g., ['.pdf', '.txt'])
            recursive: Whether to search subdirectories
            exclude_patterns: Filename patterns to exclude
            
        Returns:
            Processing results dictionary
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            return {
                "status": "error",
                "error": f"Input directory not found: {input_dir}"
            }
        
        if not input_dir.is_dir():
            return {
                "status": "error", 
                "error": f"Path is not a directory: {input_dir}"
            }
        
        # Find all files to process
        files_to_process = self._find_files(
            input_dir, 
            file_extensions, 
            recursive, 
            exclude_patterns
        )
        
        if not files_to_process:
            return {
                "status": "warning",
                "message": "No files found to process",
                "input_dir": str(input_dir),
                "file_extensions": file_extensions
            }
        
        # Initialize counters
        with self._lock:
            self._processed_count = 0
            self._error_count = 0
            self._total_files = len(files_to_process)
        
        start_time = datetime.now()
        results = []
        errors = []
        
        # Process files in parallel
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, output_dir): file_path
                    for file_path in files_to_process
                }
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        with self._lock:
                            if result["status"] == "success":
                                self._processed_count += 1
                            else:
                                self._error_count += 1
                                if not self.skip_errors:
                                    errors.append(result)
                    
                    except Exception as e:
                        error_result = {
                            "status": "error",
                            "file_path": str(file_path),
                            "error": str(e),
                            "exception_type": type(e).__name__
                        }
                        errors.append(error_result)
                        
                        with self._lock:
                            self._error_count += 1
        
        except KeyboardInterrupt:
            return {
                "status": "interrupted",
                "message": "Processing interrupted by user",
                "processed": self._processed_count,
                "errors": self._error_count,
                "total": self._total_files
            }
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Compile statistics
        successful_results = [r for r in results if r["status"] == "success"]
        total_chunks = sum(r.get("chunks_processed", 0) for r in successful_results)
        total_chars = sum(r.get("total_chars", 0) for r in successful_results)
        
        return {
            "status": "completed",
            "input_dir": str(input_dir),
            "output_dir": output_dir,
            "files_processed": self._processed_count,
            "files_with_errors": self._error_count,
            "total_files": self._total_files,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "processing_time_seconds": processing_time,
            "files_per_second": self._processed_count / processing_time if processing_time > 0 else 0,
            "results": results if len(results) <= 100 else results[:100],  # Limit output size
            "errors": errors[:50],  # Limit error output
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
    
    def process_file_list(self, 
                         file_paths: List[Union[str, Path]],
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a specific list of files
        
        Args:
            file_paths: List of file paths to process
            output_dir: Output directory for database
            
        Returns:
            Processing results dictionary
        """
        file_paths = [Path(p) for p in file_paths]
        
        # Filter existing files
        existing_files = [p for p in file_paths if p.exists()]
        
        if len(existing_files) != len(file_paths):
            missing_files = [p for p in file_paths if not p.exists()]
            print(f"Warning: {len(missing_files)} files not found:")
            for f in missing_files[:10]:  # Show first 10
                print(f"  - {f}")
        
        if not existing_files:
            return {
                "status": "error",
                "error": "No valid files found to process",
                "file_count": len(file_paths)
            }
        
        # Initialize counters
        with self._lock:
            self._processed_count = 0
            self._error_count = 0
            self._total_files = len(existing_files)
        
        start_time = datetime.now()
        results = []
        errors = []
        
        # Process files
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file_path, output_dir): file_path
                for file_path in existing_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self._lock:
                        if result["status"] == "success":
                            self._processed_count += 1
                        else:
                            self._error_count += 1
                            if not self.skip_errors:
                                errors.append(result)
                
                except Exception as e:
                    error_result = {
                        "status": "error",
                        "file_path": str(file_path),
                        "error": str(e),
                        "exception_type": type(e).__name__
                    }
                    errors.append(error_result)
                    
                    with self._lock:
                        self._error_count += 1
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        successful_results = [r for r in results if r["status"] == "success"]
        total_chunks = sum(r.get("chunks_processed", 0) for r in successful_results)
        
        return {
            "status": "completed",
            "files_processed": self._processed_count,
            "files_with_errors": self._error_count,
            "total_files": self._total_files,
            "total_chunks": total_chunks,
            "processing_time_seconds": processing_time,
            "results": results,
            "errors": errors,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
    
    def _find_files(self, 
                   directory: Path,
                   file_extensions: Optional[List[str]] = None,
                   recursive: bool = True,
                   exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Find all files to process in directory"""
        files = []
        
        # Use supported extensions if none specified
        if file_extensions is None:
            file_extensions = list(self.document_processor.get_supported_formats().keys())
        
        # Normalize extensions
        file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in file_extensions]
        
        exclude_patterns = exclude_patterns or []
        
        # Find files
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                root_path = Path(root)
                
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in filenames:
                    file_path = root_path / filename
                    
                    # Check extension
                    if file_path.suffix.lower() not in file_extensions:
                        continue
                    
                    # Check exclude patterns
                    if any(pattern in filename for pattern in exclude_patterns):
                        continue
                    
                    # Skip hidden files
                    if filename.startswith('.'):
                        continue
                    
                    files.append(file_path)
        else:
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                
                if file_path.suffix.lower() not in file_extensions:
                    continue
                
                if any(pattern in file_path.name for pattern in exclude_patterns):
                    continue
                
                if file_path.name.startswith('.'):
                    continue
                
                files.append(file_path)
        
        return sorted(files)
    
    def _process_single_file(self, file_path: Path, output_dir: Optional[str]) -> Dict[str, Any]:
        """Process a single file"""
        try:
            return self.document_processor.process_document(file_path, output_dir)
        except Exception as e:
            return {
                "status": "error",
                "file_path": str(file_path),
                "error": str(e),
                "exception_type": type(e).__name__
            }
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress"""
        with self._lock:
            total = self._total_files
            processed = self._processed_count
            errors = self._error_count
            
            progress_pct = (processed + errors) / total * 100 if total > 0 else 0
            
            return {
                "total_files": total,
                "processed": processed,
                "errors": errors,
                "remaining": total - processed - errors,
                "progress_percent": progress_pct
            } 