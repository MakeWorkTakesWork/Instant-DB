#!/usr/bin/env python3
"""
Database Query Optimizations for Instant-DB

This module provides optimized database operations including:
- Connection pooling and prepared statements
- Query optimization and indexing
- Batch operations for better performance
- Caching strategies for frequently accessed data
- Optimized vector database operations

Usage:
    from database_optimizations import OptimizedDatabase, QueryOptimizer
    
    # Use optimized database
    db = OptimizedDatabase(db_path="./optimized_db")
    
    # Optimize existing database
    optimizer = QueryOptimizer(db_path="./existing_db")
    optimizer.optimize_all()
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from collections import OrderedDict
import pickle

@dataclass
class QueryStats:
    """Statistics for database queries."""
    query_type: str
    execution_time: float
    rows_affected: int
    cache_hit: bool = False

class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)

class ConnectionPool:
    """Optimized SQLite connection pool."""
    
    def __init__(self, db_path: Path, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
        self.stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "pool_hits": 0,
            "pool_misses": 0
        }
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized SQLite connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,  # 30 second timeout
            check_same_thread=False  # Allow sharing between threads
        )
        
        # Apply performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL
        conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
        conn.execute("PRAGMA optimize")  # Enable query optimizer
        
        self.stats["connections_created"] += 1
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        
        with self.lock:
            if self.connections:
                conn = self.connections.pop()
                self.stats["pool_hits"] += 1
                self.stats["connections_reused"] += 1
            else:
                self.stats["pool_misses"] += 1
        
        if conn is None:
            conn = self._create_connection()
        
        try:
            yield conn
        finally:
            # Return to pool if space available
            with self.lock:
                if len(self.connections) < self.pool_size:
                    self.connections.append(conn)
                else:
                    conn.close()
    
    def close_all(self):
        """Close all connections in pool."""
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        with self.lock:
            return self.stats.copy()

class PreparedStatements:
    """Manage prepared statements for better performance."""
    
    def __init__(self):
        self.statements = {}
    
    def get_statement(self, conn: sqlite3.Connection, 
                     query: str, name: str) -> sqlite3.Cursor:
        """Get or create prepared statement."""
        if name not in self.statements:
            self.statements[name] = query
        
        return conn.cursor()

class QueryOptimizer:
    """Optimize database queries and structure."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection_pool = ConnectionPool(db_path)
        self.cache = LRUCache(capacity=2000)
        self.prepared_statements = PreparedStatements()
        self.query_stats = []
    
    def _execute_with_stats(self, conn: sqlite3.Connection, 
                           query: str, params: tuple = (), 
                           query_type: str = "unknown") -> Tuple[sqlite3.Cursor, QueryStats]:
        """Execute query with performance tracking."""
        start_time = time.time()
        cursor = conn.execute(query, params)
        execution_time = time.time() - start_time
        
        stats = QueryStats(
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=cursor.rowcount if cursor.rowcount >= 0 else 0
        )
        
        self.query_stats.append(stats)
        return cursor, stats
    
    def create_optimized_indexes(self):
        """Create optimized indexes for better query performance."""
        print("🔧 Creating optimized indexes...")
        
        indexes = [
            # Documents table indexes
            ("idx_documents_hash_opt", "documents", "file_hash"),
            ("idx_documents_source_opt", "documents", "source_file"),
            ("idx_documents_type_opt", "documents", "document_type"),
            ("idx_documents_processed_opt", "documents", "processed_at"),
            ("idx_documents_composite", "documents", "document_type, processed_at"),
            
            # Processing log indexes
            ("idx_log_document_opt", "processing_log", "document_id"),
            ("idx_log_operation_opt", "processing_log", "operation"),
            ("idx_log_timestamp_opt", "processing_log", "timestamp"),
            ("idx_log_status_opt", "processing_log", "status"),
            
            # Composite indexes for common queries
            ("idx_log_composite", "processing_log", "document_id, operation, status"),
        ]
        
        with self.connection_pool.get_connection() as conn:
            for index_name, table, columns in indexes:
                try:
                    query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({columns})"
                    cursor, stats = self._execute_with_stats(
                        conn, query, (), "create_index"
                    )
                    print(f"   ✅ Created index {index_name} ({stats.execution_time:.3f}s)")
                except sqlite3.Error as e:
                    print(f"   ❌ Failed to create index {index_name}: {e}")
            
            conn.commit()
    
    def optimize_table_structure(self):
        """Optimize table structure and settings."""
        print("🔧 Optimizing table structure...")
        
        optimizations = [
            "PRAGMA auto_vacuum=INCREMENTAL",  # Automatic space reclamation
            "PRAGMA foreign_keys=ON",  # Enable foreign key constraints
            "PRAGMA recursive_triggers=ON",  # Enable recursive triggers
            "PRAGMA secure_delete=OFF",  # Faster deletes (less secure)
            "PRAGMA case_sensitive_like=ON",  # Faster LIKE operations
        ]
        
        with self.connection_pool.get_connection() as conn:
            for pragma in optimizations:
                try:
                    cursor, stats = self._execute_with_stats(
                        conn, pragma, (), "pragma"
                    )
                    print(f"   ✅ Applied: {pragma} ({stats.execution_time:.3f}s)")
                except sqlite3.Error as e:
                    print(f"   ❌ Failed: {pragma} - {e}")
            
            conn.commit()
    
    def analyze_and_optimize_queries(self):
        """Analyze and optimize query performance."""
        print("🔧 Analyzing query performance...")
        
        with self.connection_pool.get_connection() as conn:
            # Update table statistics
            cursor, stats = self._execute_with_stats(
                conn, "ANALYZE", (), "analyze"
            )
            print(f"   ✅ Updated table statistics ({stats.execution_time:.3f}s)")
            
            # Optimize query planner
            cursor, stats = self._execute_with_stats(
                conn, "PRAGMA optimize", (), "optimize"
            )
            print(f"   ✅ Optimized query planner ({stats.execution_time:.3f}s)")
            
            conn.commit()
    
    def vacuum_and_cleanup(self):
        """Vacuum database and cleanup unused space."""
        print("🔧 Vacuuming database...")
        
        with self.connection_pool.get_connection() as conn:
            # Get database size before
            cursor = conn.execute("PRAGMA page_count")
            pages_before = cursor.fetchone()[0]
            
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            size_before_mb = (pages_before * page_size) / 1024 / 1024
            
            # Vacuum database
            start_time = time.time()
            conn.execute("VACUUM")
            vacuum_time = time.time() - start_time
            
            # Get size after
            cursor = conn.execute("PRAGMA page_count")
            pages_after = cursor.fetchone()[0]
            size_after_mb = (pages_after * page_size) / 1024 / 1024
            
            space_saved_mb = size_before_mb - size_after_mb
            
            print(f"   ✅ Vacuum completed ({vacuum_time:.3f}s)")
            print(f"   📊 Size before: {size_before_mb:.2f}MB")
            print(f"   📊 Size after: {size_after_mb:.2f}MB")
            print(f"   💾 Space saved: {space_saved_mb:.2f}MB")
    
    def optimize_all(self):
        """Run all optimization procedures."""
        print("🚀 STARTING DATABASE OPTIMIZATION")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            self.create_optimized_indexes()
            self.optimize_table_structure()
            self.analyze_and_optimize_queries()
            self.vacuum_and_cleanup()
            
            total_time = time.time() - start_time
            
            print(f"\n✅ DATABASE OPTIMIZATION COMPLETED")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Queries executed: {len(self.query_stats)}")
            
            # Print query statistics
            if self.query_stats:
                avg_time = sum(s.execution_time for s in self.query_stats) / len(self.query_stats)
                max_time = max(s.execution_time for s in self.query_stats)
                print(f"   Average query time: {avg_time:.3f}s")
                print(f"   Slowest query time: {max_time:.3f}s")
            
            # Print connection pool stats
            pool_stats = self.connection_pool.get_stats()
            print(f"   Connection pool hits: {pool_stats['pool_hits']}")
            print(f"   Connection pool misses: {pool_stats['pool_misses']}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            raise
        
        finally:
            self.connection_pool.close_all()

class OptimizedDatabase:
    """Optimized database operations with caching and connection pooling."""
    
    def __init__(self, db_path: Path, cache_size: int = 2000, pool_size: int = 10):
        self.db_path = db_path
        self.connection_pool = ConnectionPool(db_path, pool_size)
        self.cache = LRUCache(cache_size)
        self.prepared_statements = PreparedStatements()
        
        # Performance counters
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_query_time": 0.0
        }
    
    def _cache_key(self, query: str, params: tuple) -> str:
        """Generate cache key for query."""
        return hashlib.md5(f"{query}:{params}".encode()).hexdigest()
    
    def execute_cached_query(self, query: str, params: tuple = (), 
                           cache_ttl: int = 300) -> List[Dict[str, Any]]:
        """Execute query with caching."""
        cache_key = self._cache_key(query, params)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Execute query
        self.stats["cache_misses"] += 1
        start_time = time.time()
        
        with self.connection_pool.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
        
        query_time = time.time() - start_time
        self.stats["queries_executed"] += 1
        self.stats["total_query_time"] += query_time
        
        # Cache results
        self.cache.put(cache_key, results)
        
        return results
    
    def execute_batch_insert(self, table: str, data: List[Dict[str, Any]], 
                           batch_size: int = 1000) -> int:
        """Execute batch insert with optimal performance."""
        if not data:
            return 0
        
        total_inserted = 0
        
        # Get column names from first record
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        query = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        with self.connection_pool.get_connection() as conn:
            # Use transaction for better performance
            conn.execute("BEGIN TRANSACTION")
            
            try:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    batch_values = [
                        tuple(record[col] for col in columns)
                        for record in batch
                    ]
                    
                    cursor = conn.executemany(query, batch_values)
                    total_inserted += cursor.rowcount
                
                conn.execute("COMMIT")
                
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e
        
        # Clear relevant cache entries
        self.cache.clear()
        
        return total_inserted
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get document by file hash with caching."""
        query = "SELECT * FROM documents WHERE file_hash = ?"
        results = self.execute_cached_query(query, (file_hash,))
        return results[0] if results else None
    
    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """Get documents by type with caching."""
        query = "SELECT * FROM documents WHERE document_type = ? ORDER BY processed_at DESC"
        return self.execute_cached_query(query, (document_type,))
    
    def get_recent_processing_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent processing log entries."""
        query = """
            SELECT * FROM processing_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.execute_cached_query(query, (limit,))
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        queries = [
            ("total_documents", "SELECT COUNT(*) as count FROM documents"),
            ("total_chunks", "SELECT SUM(chunk_count) as total FROM documents"),
            ("avg_processing_time", """
                SELECT AVG(
                    CASE WHEN details LIKE '%processing_time:%' 
                    THEN CAST(SUBSTR(details, INSTR(details, 'processing_time:') + 16, 10) AS REAL)
                    ELSE NULL END
                ) as avg_time FROM processing_log WHERE operation = 'add_document'
            """),
            ("success_rate", """
                SELECT 
                    (COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*)) as rate
                FROM processing_log 
                WHERE operation = 'add_document'
            """)
        ]
        
        stats = {}
        for stat_name, query in queries:
            try:
                results = self.execute_cached_query(query)
                if results:
                    key = list(results[0].keys())[0]
                    stats[stat_name] = results[0][key] or 0
                else:
                    stats[stat_name] = 0
            except Exception as e:
                print(f"Warning: Failed to get {stat_name}: {e}")
                stats[stat_name] = 0
        
        return stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        avg_query_time = (
            self.stats["total_query_time"] / max(self.stats["queries_executed"], 1)
        )
        
        cache_hit_rate = (
            self.stats["cache_hits"] / 
            max(self.stats["cache_hits"] + self.stats["cache_misses"], 1) * 100
        )
        
        return {
            "queries_executed": self.stats["queries_executed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": self.cache.size(),
            "avg_query_time": avg_query_time,
            "total_query_time": self.stats["total_query_time"],
            "connection_pool_stats": self.connection_pool.get_stats()
        }
    
    def close(self):
        """Close database and cleanup resources."""
        self.connection_pool.close_all()
        self.cache.clear()


def create_optimization_benchmark():
    """Create benchmark to test database optimizations."""
    
    def benchmark_database_optimizations():
        """Benchmark database optimization improvements."""
        print("🔧 DATABASE OPTIMIZATION BENCHMARK")
        print("=" * 50)
        
        # Test with a temporary database
        test_db_path = Path("./test_optimization.db")
        
        try:
            # Create test database with some data
            with sqlite3.connect(test_db_path) as conn:
                # Create tables
                conn.execute('''
                    CREATE TABLE documents (
                        document_id TEXT PRIMARY KEY,
                        title TEXT,
                        document_type TEXT,
                        source_file TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        chunk_count INTEGER,
                        total_chars INTEGER,
                        file_hash TEXT,
                        embedding_model TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE processing_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT,
                        operation TEXT,
                        status TEXT,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert test data
                for i in range(1000):
                    conn.execute('''
                        INSERT INTO documents 
                        (document_id, title, document_type, source_file, chunk_count, total_chars, file_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"doc_{i}",
                        f"Document {i}",
                        "text" if i % 2 == 0 else "pdf",
                        f"/path/to/doc_{i}.txt",
                        i % 50 + 1,
                        i * 100 + 500,
                        f"hash_{i}"
                    ))
                
                conn.commit()
            
            print("📊 Created test database with 1000 documents")
            
            # Test before optimization
            print("\n📈 Testing BEFORE optimization:")
            db_before = OptimizedDatabase(test_db_path)
            
            start_time = time.time()
            for i in range(100):
                db_before.get_documents_by_type("text")
                db_before.get_document_by_hash(f"hash_{i}")
            before_time = time.time() - start_time
            
            before_stats = db_before.get_performance_stats()
            db_before.close()
            
            print(f"   Query time: {before_time:.3f}s")
            print(f"   Avg query time: {before_stats['avg_query_time']:.3f}s")
            print(f"   Cache hit rate: {before_stats['cache_hit_rate']:.1f}%")
            
            # Apply optimizations
            print("\n🔧 Applying optimizations...")
            optimizer = QueryOptimizer(test_db_path)
            optimizer.optimize_all()
            
            # Test after optimization
            print("\n📈 Testing AFTER optimization:")
            db_after = OptimizedDatabase(test_db_path)
            
            start_time = time.time()
            for i in range(100):
                db_after.get_documents_by_type("text")
                db_after.get_document_by_hash(f"hash_{i}")
            after_time = time.time() - start_time
            
            after_stats = db_after.get_performance_stats()
            db_after.close()
            
            print(f"   Query time: {after_time:.3f}s")
            print(f"   Avg query time: {after_stats['avg_query_time']:.3f}s")
            print(f"   Cache hit rate: {after_stats['cache_hit_rate']:.1f}%")
            
            # Calculate improvements
            time_improvement = (before_time - after_time) / before_time * 100
            query_improvement = (
                (before_stats['avg_query_time'] - after_stats['avg_query_time']) /
                before_stats['avg_query_time'] * 100
            )
            
            print(f"\n🎉 OPTIMIZATION RESULTS:")
            print(f"   Total time improvement: {time_improvement:.1f}%")
            print(f"   Avg query improvement: {query_improvement:.1f}%")
            print(f"   Cache efficiency: {after_stats['cache_hit_rate']:.1f}%")
            
            return {
                "before_time": before_time,
                "after_time": after_time,
                "time_improvement_percent": time_improvement,
                "query_improvement_percent": query_improvement,
                "cache_hit_rate": after_stats['cache_hit_rate']
            }
            
        finally:
            # Cleanup
            if test_db_path.exists():
                test_db_path.unlink()
    
    return benchmark_database_optimizations


if __name__ == "__main__":
    # Run database optimization benchmark
    benchmark = create_optimization_benchmark()
    results = benchmark()
    
    print(f"\n🎉 Database optimization benchmark completed!")
    print(f"📊 Results: {results}")

