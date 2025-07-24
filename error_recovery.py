#!/usr/bin/env python3
"""
Enhanced Error Recovery System for Instant-DB

This module provides robust error recovery mechanisms:
- Automatic retry with exponential backoff
- Circuit breaker pattern for failing services
- Graceful degradation strategies
- Data corruption detection and recovery
- Service failover mechanisms
- Recovery state management

Usage:
    from error_recovery import RetryManager, CircuitBreaker, RecoveryManager
    
    # Use retry decorator
    @RetryManager.retry(max_attempts=3, backoff_factor=2.0)
    def risky_operation():
        # Your code here
        pass
    
    # Use circuit breaker
    circuit_breaker = CircuitBreaker("embedding_service")
    with circuit_breaker:
        result = embedding_service.encode(text)
"""

import time
import json
import threading
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import shutil
import traceback

class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exceptions: tuple = (Exception,)
    jitter: bool = True

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0

@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    timestamp: datetime
    operation: str
    success: bool
    error_message: str
    recovery_time_ms: float
    details: Dict[str, Any]

class RetryManager:
    """Manages retry logic with various strategies."""
    
    @staticmethod
    def retry(max_attempts: int = 3, 
              backoff_factor: float = 2.0,
              base_delay: float = 1.0,
              max_delay: float = 60.0,
              strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
              exceptions: tuple = (Exception,),
              jitter: bool = True):
        """Retry decorator with configurable strategy."""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            # Last attempt failed
                            break
                        
                        # Calculate delay
                        if strategy == RetryStrategy.FIXED:
                            delay = base_delay
                        elif strategy == RetryStrategy.LINEAR:
                            delay = base_delay * (attempt + 1)
                        else:  # EXPONENTIAL
                            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        
                        # Add jitter to prevent thundering herd
                        if jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        print(f"🔄 Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} in {delay:.2f}s")
                        time.sleep(delay)
                
                # All attempts failed
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    def retry_with_config(config: RetryConfig):
        """Retry decorator using RetryConfig."""
        return RetryManager.retry(
            max_attempts=config.max_attempts,
            backoff_factor=config.backoff_factor,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            strategy=config.strategy,
            exceptions=config.exceptions,
            jitter=config.jitter
        )

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def __enter__(self):
        """Context manager entry."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    print(f"🔄 Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        with self.lock:
            if exc_type is None:
                # Success
                self._record_success()
            else:
                # Failure
                self._record_failure()
    
    def _record_success(self):
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print(f"✅ Circuit breaker {self.name} recovered, transitioning to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"❌ Circuit breaker {self.name} OPEN due to {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            print(f"❌ Circuit breaker {self.name} back to OPEN after failure during recovery")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "config": asdict(self.config)
            }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class DatabaseRecovery:
    """Database recovery mechanisms."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.backup_dir = db_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self) -> Path:
        """Create database backup."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{self.db_path.stem}_backup_{timestamp}.db"
        
        shutil.copy2(self.db_path, backup_path)
        print(f"💾 Database backup created: {backup_path}")
        
        return backup_path
    
    def restore_from_backup(self, backup_path: Path = None) -> bool:
        """Restore database from backup."""
        if backup_path is None:
            # Find most recent backup
            backups = list(self.backup_dir.glob(f"{self.db_path.stem}_backup_*.db"))
            if not backups:
                print("❌ No backups found")
                return False
            
            backup_path = max(backups, key=lambda p: p.stat().st_mtime)
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_path}")
            return False
        
        try:
            # Test backup integrity first
            with sqlite3.connect(backup_path) as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                if result != "ok":
                    print(f"❌ Backup integrity check failed: {result}")
                    return False
            
            # Restore backup
            shutil.copy2(backup_path, self.db_path)
            print(f"✅ Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def repair_database(self) -> bool:
        """Attempt to repair corrupted database."""
        if not self.db_path.exists():
            return False
        
        try:
            # Create backup before repair
            backup_path = self.create_backup()
            
            # Try to repair using SQLite recovery
            temp_path = self.db_path.with_suffix('.tmp')
            
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(temp_path) as target:
                    # Copy schema
                    source.backup(target)
            
            # Replace original with repaired version
            shutil.move(temp_path, self.db_path)
            
            # Verify repair
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                
                if result == "ok":
                    print("✅ Database repair successful")
                    return True
                else:
                    print(f"❌ Database repair failed: {result}")
                    # Restore from backup
                    return self.restore_from_backup(backup_path)
        
        except Exception as e:
            print(f"❌ Database repair error: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Clean up old backup files."""
        backups = list(self.backup_dir.glob(f"{self.db_path.stem}_backup_*.db"))
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        for backup in backups[keep_count:]:
            backup.unlink()
            print(f"🗑️  Removed old backup: {backup.name}")

class ServiceRecovery:
    """Service recovery mechanisms."""
    
    def __init__(self):
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None):
        """Register a circuit breaker for a service."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.register_circuit_breaker(name)
        return self.circuit_breakers[name]
    
    def recover_embedding_service(self) -> bool:
        """Attempt to recover embedding service."""
        start_time = time.time()
        
        try:
            # Clear any cached models
            import gc
            gc.collect()
            
            # Try to reinitialize sentence transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Test encoding
            test_embedding = model.encode(["test"])
            
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_attempts.append(RecoveryAttempt(
                timestamp=datetime.now(),
                operation="embedding_service_recovery",
                success=True,
                error_message="",
                recovery_time_ms=recovery_time,
                details={"model": "all-MiniLM-L6-v2"}
            ))
            
            print(f"✅ Embedding service recovered in {recovery_time:.1f}ms")
            return True
            
        except Exception as e:
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_attempts.append(RecoveryAttempt(
                timestamp=datetime.now(),
                operation="embedding_service_recovery",
                success=False,
                error_message=str(e),
                recovery_time_ms=recovery_time,
                details={"error_type": type(e).__name__}
            ))
            
            print(f"❌ Embedding service recovery failed: {e}")
            return False
    
    def recover_vector_store(self, db_path: Path) -> bool:
        """Attempt to recover vector store."""
        start_time = time.time()
        
        try:
            # Ensure directory exists
            chroma_path = db_path / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Try to initialize ChromaDB
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            
            # Create default collection if none exists
            collections = client.list_collections()
            if not collections:
                client.create_collection("instant_db_collection")
                print("📁 Created default vector store collection")
            
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_attempts.append(RecoveryAttempt(
                timestamp=datetime.now(),
                operation="vector_store_recovery",
                success=True,
                error_message="",
                recovery_time_ms=recovery_time,
                details={"chroma_path": str(chroma_path)}
            ))
            
            print(f"✅ Vector store recovered in {recovery_time:.1f}ms")
            return True
            
        except Exception as e:
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_attempts.append(RecoveryAttempt(
                timestamp=datetime.now(),
                operation="vector_store_recovery",
                success=False,
                error_message=str(e),
                recovery_time_ms=recovery_time,
                details={"error_type": type(e).__name__}
            ))
            
            print(f"❌ Vector store recovery failed: {e}")
            return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_attempts:
            return {"total_attempts": 0}
        
        successful = [a for a in self.recovery_attempts if a.success]
        failed = [a for a in self.recovery_attempts if not a.success]
        
        avg_recovery_time = sum(a.recovery_time_ms for a in successful) / len(successful) if successful else 0
        
        return {
            "total_attempts": len(self.recovery_attempts),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.recovery_attempts) * 100,
            "avg_recovery_time_ms": avg_recovery_time,
            "recent_attempts": [asdict(a) for a in self.recovery_attempts[-10:]]
        }

class RecoveryManager:
    """Main recovery management system."""
    
    def __init__(self, db_path: str = "./instant_db_database"):
        self.db_path = Path(db_path)
        self.database_recovery = DatabaseRecovery(self.db_path / "metadata.db")
        self.service_recovery = ServiceRecovery()
        
        # Setup default circuit breakers
        self.service_recovery.register_circuit_breaker("embedding_service")
        self.service_recovery.register_circuit_breaker("vector_store")
        self.service_recovery.register_circuit_breaker("database")
    
    def auto_recover(self) -> Dict[str, bool]:
        """Attempt automatic recovery of all services."""
        print("🔧 Starting automatic recovery...")
        
        results = {}
        
        # Recover embedding service
        try:
            results["embedding_service"] = self.service_recovery.recover_embedding_service()
        except Exception as e:
            print(f"❌ Embedding service recovery error: {e}")
            results["embedding_service"] = False
        
        # Recover vector store
        try:
            results["vector_store"] = self.service_recovery.recover_vector_store(self.db_path)
        except Exception as e:
            print(f"❌ Vector store recovery error: {e}")
            results["vector_store"] = False
        
        # Check database integrity
        try:
            metadata_db = self.db_path / "metadata.db"
            if metadata_db.exists():
                with sqlite3.connect(metadata_db) as conn:
                    cursor = conn.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()[0]
                    
                    if result == "ok":
                        results["database"] = True
                    else:
                        print(f"🔧 Database integrity issue detected: {result}")
                        results["database"] = self.database_recovery.repair_database()
            else:
                results["database"] = True  # No database to check
        except Exception as e:
            print(f"❌ Database recovery error: {e}")
            results["database"] = False
        
        successful_recoveries = sum(results.values())
        total_recoveries = len(results)
        
        print(f"🎯 Recovery completed: {successful_recoveries}/{total_recoveries} services recovered")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system recovery status."""
        circuit_breaker_states = {
            name: cb.get_state() 
            for name, cb in self.service_recovery.circuit_breakers.items()
        }
        
        recovery_stats = self.service_recovery.get_recovery_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": circuit_breaker_states,
            "recovery_statistics": recovery_stats,
            "database_path": str(self.db_path)
        }

def create_error_recovery_demo():
    """Create a demo of the error recovery system."""
    
    def demo_error_recovery():
        """Demonstrate error recovery capabilities."""
        print("🔧 ERROR RECOVERY SYSTEM DEMO")
        print("=" * 50)
        
        # Initialize recovery manager
        recovery_manager = RecoveryManager("./demo_recovery_db")
        
        # Demo retry mechanism
        print("🔄 Testing retry mechanism...")
        
        @RetryManager.retry(max_attempts=3, backoff_factor=1.5)
        def flaky_function():
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise ValueError("Simulated failure")
            return "Success!"
        
        try:
            result = flaky_function()
            print(f"   ✅ Retry succeeded: {result}")
        except Exception as e:
            print(f"   ❌ Retry failed: {e}")
        
        # Demo circuit breaker
        print("\n⚡ Testing circuit breaker...")
        
        circuit_breaker = recovery_manager.service_recovery.get_circuit_breaker("test_service")
        
        # Simulate failures to open circuit
        for i in range(6):
            try:
                with circuit_breaker:
                    raise Exception(f"Simulated failure {i+1}")
            except Exception as e:
                print(f"   Attempt {i+1}: {e}")
        
        # Try to use service when circuit is open
        try:
            with circuit_breaker:
                print("   This should not execute")
        except CircuitBreakerOpenError as e:
            print(f"   ⚡ Circuit breaker blocked request: {e}")
        
        # Demo auto recovery
        print("\n🔧 Testing automatic recovery...")
        recovery_results = recovery_manager.auto_recover()
        
        for service, recovered in recovery_results.items():
            status = "✅" if recovered else "❌"
            print(f"   {status} {service}: {'Recovered' if recovered else 'Failed'}")
        
        # Show system status
        print("\n📊 System recovery status:")
        status = recovery_manager.get_system_status()
        
        print(f"   Circuit breakers: {len(status['circuit_breakers'])}")
        for name, cb_state in status['circuit_breakers'].items():
            print(f"     {name}: {cb_state['state']} (failures: {cb_state['failure_count']})")
        
        recovery_stats = status['recovery_statistics']
        if recovery_stats['total_attempts'] > 0:
            print(f"   Recovery attempts: {recovery_stats['total_attempts']}")
            print(f"   Success rate: {recovery_stats['success_rate']:.1f}%")
            print(f"   Avg recovery time: {recovery_stats['avg_recovery_time_ms']:.1f}ms")
        
        print("\n✅ Error recovery demo completed!")
    
    return demo_error_recovery


if __name__ == "__main__":
    # Run error recovery demo
    demo = create_error_recovery_demo()
    demo()

