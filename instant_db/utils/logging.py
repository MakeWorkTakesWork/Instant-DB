"""
Logging utilities for Instant-DB
Provides structured logging with different levels and output formats
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import traceback


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: Union[str, int] = logging.INFO,
                 log_file: Optional[Union[str, Path]] = None,
                 json_format: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True,
                 logger_name: str = 'instant_db') -> logging.Logger:
    """
    Setup logging configuration for Instant-DB
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or numeric)
        log_file: Optional file path for log output
        json_format: Whether to use JSON format for logs
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
        console_formatter = ColoredFormatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'instant_db') -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(f"instant_db.{self.__class__.__name__}")


class ProgressLogger:
    """Logger for tracking progress of long-running operations"""
    
    def __init__(self, 
                 total: int,
                 description: str = "Processing",
                 logger: Optional[logging.Logger] = None,
                 log_interval: int = 10):
        """
        Initialize progress logger
        
        Args:
            total: Total number of items to process
            description: Description of the operation
            logger: Logger instance (will create if None)
            log_interval: Log progress every N percent
        """
        self.total = total
        self.description = description
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        
        self.current = 0
        self.last_logged_percent = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        
        if self.total > 0:
            percent = int((self.current / self.total) * 100)
            
            if percent >= self.last_logged_percent + self.log_interval or self.current >= self.total:
                elapsed = datetime.now() - self.start_time
                
                if self.current < self.total and elapsed.total_seconds() > 0:
                    # Estimate remaining time
                    rate = self.current / elapsed.total_seconds()
                    remaining = (self.total - self.current) / rate if rate > 0 else 0
                    eta_str = f" (ETA: {int(remaining)}s)"
                else:
                    eta_str = ""
                
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} ({percent}%){eta_str}"
                )
                
                self.last_logged_percent = percent
    
    def complete(self):
        """Mark as complete"""
        if self.current < self.total:
            self.current = self.total
        
        elapsed = datetime.now() - self.start_time
        self.logger.info(
            f"{self.description} completed: {self.total} items in {elapsed.total_seconds():.1f}s"
        )


class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = datetime.now()
    
    def end_timer(self, name: str, log_result: bool = True) -> float:
        """End a named timer and return elapsed seconds"""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        start_time = self.timers.pop(name)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if log_result:
            self.logger.info(f"Timer '{name}': {elapsed:.3f}s")
        
        return elapsed
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


def log_function_call(func):
    """Decorator to log function calls with timing"""
    def wrapper(*args, **kwargs):
        logger = get_logger(f"instant_db.{func.__module__}")
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


def log_exceptions(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(f"instant_db.{func.__module__}")
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.exception(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    
    return decorator


# Global performance logger instance
performance_logger = PerformanceLogger()


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message"""
    get_logger().critical(message, **kwargs) 