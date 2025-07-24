#!/usr/bin/env python3
"""
Production Logging System for Instant-DB

This module provides comprehensive logging capabilities for production environments:
- Structured JSON logging
- Multiple log levels and handlers
- Performance metrics logging
- Error tracking and alerting
- Log rotation and management
- Centralized logging configuration

Usage:
    from production_logging import ProductionLogger, LogConfig
    
    # Initialize logger
    logger = ProductionLogger("instant_db", LogConfig())
    
    # Use structured logging
    logger.info("Document processed", {
        "document_id": "doc_123",
        "chunks": 15,
        "processing_time": 2.5
    })
"""

import json
import logging
import logging.handlers
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from contextlib import contextmanager
import sys
import os

@dataclass
class LogConfig:
    """Configuration for production logging."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = True
    enable_performance: bool = True
    enable_error_tracking: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]
    
    @classmethod
    def from_record(cls, record: logging.LogRecord, extra_data: Dict[str, Any] = None):
        """Create LogEntry from logging record."""
        return cls(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            extra_data=extra_data or {}
        )

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract extra data
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                extra_data[key] = value
        
        # Create structured log entry
        log_entry = LogEntry.from_record(record, extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry.extra_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(asdict(log_entry), default=str)

class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
        self.lock = threading.Lock()
    
    def log_metric(self, metric_name: str, value: float, 
                   tags: Dict[str, str] = None):
        """Log a performance metric."""
        with self.lock:
            timestamp = time.time()
            metric_data = {
                "metric_name": metric_name,
                "value": value,
                "timestamp": timestamp,
                "tags": tags or {}
            }
            
            self.logger.info(f"METRIC: {metric_name}", extra={
                "metric_type": "performance",
                "metric_data": metric_data
            })
    
    def log_timing(self, operation: str, duration: float, 
                   success: bool = True, **kwargs):
        """Log operation timing."""
        self.log_metric(
            f"{operation}_duration",
            duration,
            {
                "operation": operation,
                "success": str(success),
                **kwargs
            }
        )
    
    def log_counter(self, counter_name: str, increment: int = 1,
                   tags: Dict[str, str] = None):
        """Log counter metric."""
        self.log_metric(f"{counter_name}_count", increment, tags)

class ErrorTracker:
    """Track and analyze errors."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_counts = {}
        self.lock = threading.Lock()
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track an error occurrence."""
        error_type = type(error).__name__
        error_message = str(error)
        
        with self.lock:
            key = f"{error_type}:{error_message}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self.logger.error(f"ERROR: {error_type}", extra={
            "error_type": error_type,
            "error_message": error_message,
            "error_count": self.error_counts[key],
            "context": context or {},
            "traceback": traceback.format_exc()
        }, exc_info=True)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        with self.lock:
            return self.error_counts.copy()

class ProductionLogger:
    """Main production logging class."""
    
    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self.performance_logger = PerformanceLogger(self.logger)
        self.error_tracker = ErrorTracker(self.logger)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.config.enable_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(logging.Formatter(self.config.log_format))
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.enable_file:
            file_path = log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            
            if self.config.enable_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(self.config.log_format))
            
            self.logger.addHandler(file_handler)
        
        # Performance log file
        if self.config.enable_performance:
            perf_file = log_dir / f"{self.name}_performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            perf_handler.setFormatter(JSONFormatter())
            perf_handler.addFilter(lambda record: hasattr(record, 'metric_type'))
            self.logger.addHandler(perf_handler)
        
        # Error log file
        if self.config.enable_error_tracking:
            error_file = log_dir / f"{self.name}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            error_handler.setFormatter(JSONFormatter())
            error_handler.setLevel(logging.ERROR)
            self.logger.addHandler(error_handler)
    
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Log debug message."""
        self.logger.debug(message, extra=extra_data or {})
    
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message."""
        self.logger.info(message, extra=extra_data or {})
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Log warning message."""
        self.logger.warning(message, extra=extra_data or {})
    
    def error(self, message: str, extra_data: Dict[str, Any] = None, 
              exception: Exception = None):
        """Log error message."""
        if exception:
            self.error_tracker.track_error(exception, extra_data)
        else:
            self.logger.error(message, extra=extra_data or {})
    
    def critical(self, message: str, extra_data: Dict[str, Any] = None):
        """Log critical message."""
        self.logger.critical(message, extra=extra_data or {})
    
    @contextmanager
    def log_operation(self, operation_name: str, **context):
        """Context manager to log operation timing and success."""
        start_time = time.time()
        success = True
        error = None
        
        self.info(f"Starting operation: {operation_name}", {
            "operation": operation_name,
            "context": context
        })
        
        try:
            yield
        except Exception as e:
            success = False
            error = e
            self.error_tracker.track_error(e, {
                "operation": operation_name,
                "context": context
            })
            raise
        finally:
            duration = time.time() - start_time
            
            self.performance_logger.log_timing(
                operation_name, duration, success, **context
            )
            
            if success:
                self.info(f"Completed operation: {operation_name}", {
                    "operation": operation_name,
                    "duration": duration,
                    "success": success,
                    "context": context
                })
            else:
                self.error(f"Failed operation: {operation_name}", {
                    "operation": operation_name,
                    "duration": duration,
                    "success": success,
                    "error": str(error) if error else None,
                    "context": context
                })
    
    def log_metric(self, metric_name: str, value: float, 
                   tags: Dict[str, str] = None):
        """Log performance metric."""
        self.performance_logger.log_metric(metric_name, value, tags)
    
    def log_counter(self, counter_name: str, increment: int = 1,
                   tags: Dict[str, str] = None):
        """Log counter metric."""
        self.performance_logger.log_counter(counter_name, increment, tags)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary."""
        return self.error_tracker.get_error_summary()

class LogAnalyzer:
    """Analyze log files for insights."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
    
    def analyze_performance_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance logs for the last N hours."""
        perf_file = self.log_dir / "instant_db_performance.log"
        
        if not perf_file.exists():
            return {"error": "Performance log file not found"}
        
        cutoff_time = time.time() - (hours * 3600)
        metrics = []
        
        try:
            with open(perf_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if (log_entry.get('extra_data', {}).get('metric_data', {}).get('timestamp', 0) > cutoff_time):
                            metrics.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {"error": "Performance log file not found"}
        
        # Analyze metrics
        operation_times = {}
        metric_counts = {}
        
        for metric in metrics:
            metric_data = metric.get('extra_data', {}).get('metric_data', {})
            metric_name = metric_data.get('metric_name', '')
            value = metric_data.get('value', 0)
            
            if metric_name.endswith('_duration'):
                operation = metric_name.replace('_duration', '')
                if operation not in operation_times:
                    operation_times[operation] = []
                operation_times[operation].append(value)
            
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        # Calculate statistics
        operation_stats = {}
        for operation, times in operation_times.items():
            operation_stats[operation] = {
                "count": len(times),
                "avg_duration": sum(times) / len(times),
                "min_duration": min(times),
                "max_duration": max(times),
                "total_duration": sum(times)
            }
        
        return {
            "analysis_period_hours": hours,
            "total_metrics": len(metrics),
            "operation_statistics": operation_stats,
            "metric_counts": metric_counts
        }
    
    def analyze_error_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error logs for the last N hours."""
        error_file = self.log_dir / "instant_db_errors.log"
        
        if not error_file.exists():
            return {"error": "Error log file not found"}
        
        cutoff_time = time.time() - (hours * 3600)
        errors = []
        
        try:
            with open(error_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00')).timestamp()
                        if timestamp > cutoff_time:
                            errors.append(log_entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            return {"error": "Error log file not found"}
        
        # Analyze errors
        error_types = {}
        error_modules = {}
        error_timeline = {}
        
        for error in errors:
            error_type = error.get('extra_data', {}).get('error_type', 'Unknown')
            module = error.get('module', 'Unknown')
            hour = datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')).hour
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_modules[module] = error_modules.get(module, 0) + 1
            error_timeline[hour] = error_timeline.get(hour, 0) + 1
        
        return {
            "analysis_period_hours": hours,
            "total_errors": len(errors),
            "error_types": error_types,
            "error_modules": error_modules,
            "error_timeline": error_timeline,
            "error_rate_per_hour": len(errors) / hours
        }

def create_production_logger_demo():
    """Create a demo of the production logging system."""
    
    def demo_production_logging():
        """Demonstrate production logging capabilities."""
        print("🔧 PRODUCTION LOGGING SYSTEM DEMO")
        print("=" * 50)
        
        # Initialize logger
        config = LogConfig(
            log_level="DEBUG",
            log_dir="./demo_logs",
            enable_console=True,
            enable_file=True,
            enable_json=True
        )
        
        logger = ProductionLogger("instant_db_demo", config)
        
        # Demo basic logging
        logger.info("Application started", {
            "version": "1.1.0",
            "environment": "production"
        })
        
        # Demo operation logging
        with logger.log_operation("document_processing", document_id="doc_123"):
            time.sleep(0.1)  # Simulate work
            logger.info("Processing document", {
                "document_id": "doc_123",
                "chunks": 15
            })
        
        # Demo performance metrics
        logger.log_metric("memory_usage_mb", 256.5, {"component": "embeddings"})
        logger.log_counter("documents_processed", 1, {"type": "pdf"})
        
        # Demo error tracking
        try:
            raise ValueError("Test error for demonstration")
        except ValueError as e:
            logger.error("Document processing failed", {
                "document_id": "doc_456",
                "error_context": "embedding_generation"
            }, exception=e)
        
        # Demo warning
        logger.warning("High memory usage detected", {
            "memory_usage_mb": 512,
            "threshold_mb": 400
        })
        
        # Analyze logs
        time.sleep(0.1)  # Ensure logs are written
        
        analyzer = LogAnalyzer("./demo_logs")
        perf_analysis = analyzer.analyze_performance_logs(1)
        error_analysis = analyzer.analyze_error_logs(1)
        
        print("\n📊 PERFORMANCE ANALYSIS:")
        if "error" not in perf_analysis:
            print(f"   Total metrics: {perf_analysis['total_metrics']}")
            print(f"   Operations tracked: {len(perf_analysis['operation_statistics'])}")
            for op, stats in perf_analysis['operation_statistics'].items():
                print(f"   {op}: {stats['avg_duration']:.3f}s avg, {stats['count']} runs")
        else:
            print(f"   {perf_analysis['error']}")
        
        print("\n🚨 ERROR ANALYSIS:")
        if "error" not in error_analysis:
            print(f"   Total errors: {error_analysis['total_errors']}")
            print(f"   Error rate: {error_analysis['error_rate_per_hour']:.1f}/hour")
            print(f"   Error types: {error_analysis['error_types']}")
        else:
            print(f"   {error_analysis['error']}")
        
        print("\n✅ Production logging demo completed!")
        print(f"📁 Log files created in: ./demo_logs/")
    
    return demo_production_logging


if __name__ == "__main__":
    # Run production logging demo
    demo = create_production_logger_demo()
    demo()

