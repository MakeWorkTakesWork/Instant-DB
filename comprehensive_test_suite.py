#!/usr/bin/env python3
"""
Comprehensive Test Suite for Instant-DB Release Validation

This test suite validates all major components and optimizations implemented
in the Instant-DB optimization project. It ensures that all features work
correctly and performance targets are met before release.

Test Categories:
- Core functionality tests
- Performance optimization validation
- Production hardening verification
- Error recovery testing
- Documentation and API validation
- Integration tests

Usage:
    python comprehensive_test_suite.py --all
    python comprehensive_test_suite.py --category performance
    python comprehensive_test_suite.py --category production
"""

import sys
import os
import time
import json
import tempfile
import shutil
import unittest
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import traceback

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "Instant-DB"))
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class TestResult:
    """Test result container."""
    name: str
    category: str
    status: str  # "passed", "failed", "skipped"
    duration: float
    message: str
    details: Dict[str, Any]

class TestRunner:
    """Main test runner for comprehensive validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.temp_dirs: List[Path] = []
        
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def run_test(self, test_func, category: str = "general"):
        """Run a single test function."""
        test_name = test_func.__name__
        start_time = time.time()
        
        try:
            print(f"🔍 Running {test_name}...")
            result = test_func()
            duration = time.time() - start_time
            
            if result is True or (isinstance(result, dict) and result.get('success', True)):
                status = "passed"
                message = "Test passed successfully"
                details = result if isinstance(result, dict) else {}
            else:
                status = "failed"
                message = str(result) if result else "Test failed"
                details = {}
            
        except Exception as e:
            duration = time.time() - start_time
            status = "failed"
            message = f"Test failed with exception: {str(e)}"
            details = {"exception": str(e), "traceback": traceback.format_exc()}
        
        test_result = TestResult(
            name=test_name,
            category=category,
            status=status,
            duration=duration,
            message=message,
            details=details
        )
        
        self.results.append(test_result)
        
        status_emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        print(f"   {status_emoji} {test_name}: {status} ({duration:.3f}s)")
        
        return test_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        
        total_duration = sum(r.duration for r in self.results)
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "failed": 0, "skipped": 0}
            categories[result.category][result.status] += 1
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": total_duration,
            "categories": categories,
            "failed_tests": [r.name for r in self.results if r.status == "failed"]
        }

class CoreFunctionalityTests:
    """Test core Instant-DB functionality."""
    
    def __init__(self, test_runner: TestRunner):
        self.runner = test_runner
    
    def test_basic_database_operations(self):
        """Test basic database initialization and operations."""
        try:
            from instant_db import InstantDB
            
            temp_dir = self.runner.create_temp_dir("core_test_")
            
            # Initialize database
            db = InstantDB(
                db_path=str(temp_dir),
                embedding_provider="sentence-transformers",
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Test document addition
            result = db.add_document(
                content="This is a test document for validation.",
                metadata={"title": "Test Document", "category": "test"},
                document_id="test_doc_1"
            )
            
            assert result["status"] == "success"
            assert result["chunks_processed"] > 0
            
            # Test search
            search_results = db.search("test document", top_k=5)
            assert len(search_results) > 0
            assert search_results[0]["score"] > 0.5
            
            # Test document retrieval
            doc = db.get_document("test_doc_1")
            assert doc is not None
            assert doc["metadata"]["title"] == "Test Document"
            
            return {
                "success": True,
                "chunks_processed": result["chunks_processed"],
                "search_results": len(search_results),
                "top_score": search_results[0]["score"]
            }
            
        except ImportError:
            return {"success": False, "error": "instant_db module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_optimized_implementation(self):
        """Test optimized InstantDB implementation."""
        try:
            from memory_optimizations import OptimizedInstantDB, MemoryConfig
            
            temp_dir = self.runner.create_temp_dir("optimized_test_")
            
            # Configure memory optimization
            config = MemoryConfig(
                memory_limit_mb=256,
                batch_size=25,
                enable_streaming=True
            )
            
            # Initialize optimized database
            db = OptimizedInstantDB(
                db_path=str(temp_dir),
                memory_config=config
            )
            
            # Test optimized document processing
            result = db.add_document_optimized(
                content="This is a test document for optimized processing validation.",
                metadata={"title": "Optimized Test", "type": "validation"},
                document_id="optimized_test_1"
            )
            
            assert result["status"] == "success"
            
            # Get memory statistics
            memory_stats = db.get_memory_stats()
            assert "peak_usage_mb" in memory_stats
            assert memory_stats["peak_usage_mb"] < config.memory_limit_mb
            
            return {
                "success": True,
                "memory_efficiency": memory_stats.get("efficiency_percent", 0),
                "peak_memory_mb": memory_stats.get("peak_usage_mb", 0)
            }
            
        except ImportError:
            return {"success": False, "error": "memory_optimizations module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_database_optimizations(self):
        """Test database optimization features."""
        try:
            from database_optimizations import OptimizedDatabase, QueryOptimizer
            
            temp_dir = self.runner.create_temp_dir("db_opt_test_")
            db_path = temp_dir / "test.db"
            
            # Create test database
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    CREATE TABLE test_table (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        category TEXT
                    )
                ''')
                
                # Insert test data
                for i in range(100):
                    conn.execute(
                        "INSERT INTO test_table (name, category) VALUES (?, ?)",
                        (f"item_{i}", f"category_{i % 5}")
                    )
                conn.commit()
            
            # Test query optimizer
            optimizer = QueryOptimizer(db_path)
            optimizer.optimize_all()
            
            # Test optimized database operations
            opt_db = OptimizedDatabase(db_path)
            
            # Test cached queries
            start_time = time.time()
            results = opt_db.execute_cached_query(
                "SELECT * FROM test_table WHERE category = ?",
                ("category_1",)
            )
            query_time = time.time() - start_time
            
            assert len(results) > 0
            
            # Test cache hit
            start_time = time.time()
            cached_results = opt_db.execute_cached_query(
                "SELECT * FROM test_table WHERE category = ?",
                ("category_1",)
            )
            cached_time = time.time() - start_time
            
            # Cached query should be faster
            assert cached_time < query_time
            
            # Get performance stats
            perf_stats = opt_db.get_performance_stats()
            opt_db.close()
            
            return {
                "success": True,
                "query_time": query_time,
                "cached_time": cached_time,
                "cache_hit_rate": perf_stats.get("cache_hit_rate", 0),
                "speedup": query_time / cached_time if cached_time > 0 else 1
            }
            
        except ImportError:
            return {"success": False, "error": "database_optimizations module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class PerformanceTests:
    """Test performance optimizations and benchmarks."""
    
    def __init__(self, test_runner: TestRunner):
        self.runner = test_runner
    
    def test_performance_improvements(self):
        """Validate that performance improvements meet targets."""
        try:
            from performance_comparison import PerformanceComparator
            
            # Create test dataset
            temp_dir = self.runner.create_temp_dir("perf_test_")
            test_dataset = temp_dir / "test_dataset"
            test_dataset.mkdir()
            
            # Create test files
            test_files = [
                "Machine learning is a subset of artificial intelligence.",
                "Vector databases store high-dimensional vectors for similarity search.",
                "Document processing involves chunking and embedding generation.",
                "Semantic search uses embeddings to find relevant content."
            ]
            
            for i, content in enumerate(test_files):
                (test_dataset / f"test_file_{i}.txt").write_text(content)
            
            # Run performance comparison
            comparator = PerformanceComparator(str(test_dataset))
            
            # Test optimized implementation only (original may not be available)
            optimized_results = comparator.test_optimized_implementation()
            
            if not optimized_results:
                return {"success": False, "error": "Failed to test optimized implementation"}
            
            # Validate performance metrics
            processing_time = optimized_results.document_processing["avg_file_time"]
            memory_usage = optimized_results.memory_usage["delta_mb"]
            overall_score = optimized_results.overall_score
            
            # Performance targets (based on achieved improvements)
            targets = {
                "max_processing_time": 2.0,  # seconds per file
                "max_memory_usage": 100.0,   # MB
                "min_overall_score": 30.0    # performance score
            }
            
            meets_targets = (
                processing_time <= targets["max_processing_time"] and
                memory_usage <= targets["max_memory_usage"] and
                overall_score >= targets["min_overall_score"]
            )
            
            return {
                "success": meets_targets,
                "processing_time": processing_time,
                "memory_usage": memory_usage,
                "overall_score": overall_score,
                "targets_met": meets_targets,
                "targets": targets
            }
            
        except ImportError:
            return {"success": False, "error": "performance_comparison module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_memory_optimizations(self):
        """Test memory optimization effectiveness."""
        try:
            from memory_optimizations import OptimizedInstantDB, MemoryConfig
            import psutil
            
            temp_dir = self.runner.create_temp_dir("memory_test_")
            
            # Test with strict memory limits
            config = MemoryConfig(
                memory_limit_mb=128,
                batch_size=10,
                enable_streaming=True,
                enable_compression=True
            )
            
            db = OptimizedInstantDB(str(temp_dir), memory_config=config)
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Process a larger document
            large_content = "This is a test document. " * 1000  # ~25KB
            
            result = db.add_document_optimized(
                content=large_content,
                metadata={"title": "Large Test Document"},
                document_id="large_test"
            )
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            # Get internal memory stats
            memory_stats = db.get_memory_stats()
            
            return {
                "success": result["status"] == "success" and memory_increase < 200,  # Less than 200MB increase
                "memory_increase_mb": memory_increase,
                "internal_peak_mb": memory_stats.get("peak_usage_mb", 0),
                "efficiency_percent": memory_stats.get("efficiency_percent", 0),
                "chunks_processed": result.get("chunks_processed", 0)
            }
            
        except ImportError:
            return {"success": False, "error": "memory_optimizations module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class ProductionHardeningTests:
    """Test production hardening features."""
    
    def __init__(self, test_runner: TestRunner):
        self.runner = test_runner
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        try:
            from health_monitoring import HealthMonitor, HealthCheckServer
            
            temp_dir = self.runner.create_temp_dir("health_test_")
            
            # Initialize health monitor
            monitor = HealthMonitor(str(temp_dir))
            
            # Run health checks
            status = monitor.get_overall_status()
            
            # Validate health check structure
            required_fields = ["overall_status", "timestamp", "status_counts", "total_checks", "checks"]
            for field in required_fields:
                assert field in status, f"Missing field: {field}"
            
            # Check that we have some health checks
            assert status["total_checks"] > 0
            assert len(status["checks"]) > 0
            
            # Validate individual checks
            for check_name, check_data in status["checks"].items():
                assert "status" in check_data
                assert "message" in check_data
                assert "response_time_ms" in check_data
                assert check_data["status"] in ["healthy", "degraded", "unhealthy"]
            
            return {
                "success": True,
                "overall_status": status["overall_status"],
                "total_checks": status["total_checks"],
                "healthy_checks": status["status_counts"].get("healthy", 0)
            }
            
        except ImportError:
            return {"success": False, "error": "health_monitoring module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        try:
            from error_recovery import RetryManager, CircuitBreaker, RecoveryManager
            
            # Test retry mechanism
            attempt_count = 0
            
            @RetryManager.retry(max_attempts=3, base_delay=0.1)
            def flaky_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError("Simulated failure")
                return "success"
            
            result = flaky_function()
            assert result == "success"
            assert attempt_count == 3
            
            # Test circuit breaker
            circuit_breaker = CircuitBreaker("test_service")
            
            # Test normal operation
            with circuit_breaker:
                pass  # Should succeed
            
            state = circuit_breaker.get_state()
            assert state["state"] == "closed"
            
            # Test recovery manager
            temp_dir = self.runner.create_temp_dir("recovery_test_")
            recovery_manager = RecoveryManager(str(temp_dir))
            
            # Test auto recovery
            recovery_results = recovery_manager.auto_recover()
            
            return {
                "success": True,
                "retry_attempts": attempt_count,
                "circuit_breaker_state": state["state"],
                "recovery_services": len(recovery_results),
                "successful_recoveries": sum(recovery_results.values())
            }
            
        except ImportError:
            return {"success": False, "error": "error_recovery module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_production_logging(self):
        """Test production logging system."""
        try:
            from production_logging import ProductionLogger, LogConfig
            
            temp_dir = self.runner.create_temp_dir("logging_test_")
            
            # Configure logging
            config = LogConfig(
                log_level="INFO",
                log_dir=str(temp_dir / "logs"),
                enable_console=False,  # Disable console for testing
                enable_file=True,
                enable_json=True
            )
            
            logger = ProductionLogger("test_logger", config)
            
            # Test different log levels
            logger.info("Test info message", {"test_data": "info"})
            logger.warning("Test warning message", {"test_data": "warning"})
            logger.error("Test error message", {"test_data": "error"})
            
            # Test operation logging
            with logger.log_operation("test_operation", test_param="value"):
                time.sleep(0.1)  # Simulate work
            
            # Test metrics logging
            logger.log_metric("test_metric", 42.5, {"component": "test"})
            
            # Check that log files were created
            log_dir = Path(config.log_dir)
            log_files = list(log_dir.glob("*.log"))
            
            return {
                "success": len(log_files) > 0,
                "log_files_created": len(log_files),
                "log_directory": str(log_dir)
            }
            
        except ImportError:
            return {"success": False, "error": "production_logging module not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class IntegrationTests:
    """Test integration between components."""
    
    def __init__(self, test_runner: TestRunner):
        self.runner = test_runner
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        try:
            from instant_db import InstantDB
            from production_logging import ProductionLogger, LogConfig
            from health_monitoring import HealthMonitor
            
            temp_dir = self.runner.create_temp_dir("e2e_test_")
            
            # Setup logging
            log_config = LogConfig(
                log_dir=str(temp_dir / "logs"),
                enable_console=False,
                enable_file=True
            )
            logger = ProductionLogger("e2e_test", log_config)
            
            # Initialize database with logging
            logger.info("Initializing database")
            db = InstantDB(db_path=str(temp_dir / "database"))
            
            # Add multiple documents
            documents = [
                {"content": "Machine learning algorithms for data analysis", "title": "ML Guide"},
                {"content": "Vector databases and similarity search", "title": "Vector DB"},
                {"content": "Production deployment best practices", "title": "Deployment"},
                {"content": "Monitoring and observability in production", "title": "Monitoring"}
            ]
            
            for i, doc in enumerate(documents):
                logger.info(f"Processing document {i+1}")
                result = db.add_document(
                    content=doc["content"],
                    metadata={"title": doc["title"], "index": i},
                    document_id=f"doc_{i}"
                )
                assert result["status"] == "success"
            
            # Test search functionality
            logger.info("Testing search functionality")
            search_results = db.search("machine learning", top_k=3)
            assert len(search_results) > 0
            
            # Test health monitoring
            logger.info("Testing health monitoring")
            health_monitor = HealthMonitor(str(temp_dir / "database"))
            health_status = health_monitor.get_overall_status()
            
            # Validate workflow completion
            return {
                "success": True,
                "documents_processed": len(documents),
                "search_results": len(search_results),
                "health_status": health_status["overall_status"],
                "top_search_score": search_results[0]["score"] if search_results else 0
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Import error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Suite")
    parser.add_argument("--category", choices=["core", "performance", "production", "integration", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🚀 STARTING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    runner = TestRunner()
    
    try:
        # Initialize test classes
        core_tests = CoreFunctionalityTests(runner)
        perf_tests = PerformanceTests(runner)
        prod_tests = ProductionHardeningTests(runner)
        integration_tests = IntegrationTests(runner)
        
        # Define test categories
        test_categories = {
            "core": [
                (core_tests.test_basic_database_operations, "core"),
                (core_tests.test_optimized_implementation, "core"),
                (core_tests.test_database_optimizations, "core")
            ],
            "performance": [
                (perf_tests.test_performance_improvements, "performance"),
                (perf_tests.test_memory_optimizations, "performance")
            ],
            "production": [
                (prod_tests.test_health_monitoring, "production"),
                (prod_tests.test_error_recovery, "production"),
                (prod_tests.test_production_logging, "production")
            ],
            "integration": [
                (integration_tests.test_end_to_end_workflow, "integration")
            ]
        }
        
        # Run selected tests
        if args.category == "all":
            tests_to_run = []
            for category_tests in test_categories.values():
                tests_to_run.extend(category_tests)
        else:
            tests_to_run = test_categories.get(args.category, [])
        
        print(f"📋 Running {len(tests_to_run)} tests in category: {args.category}")
        print()
        
        # Execute tests
        for test_func, category in tests_to_run:
            runner.run_test(test_func, category)
        
        # Print summary
        summary = runner.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️ Skipped: {summary['skipped']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {summary['total_duration']:.3f}s")
        
        print(f"\n📋 Results by Category:")
        for category, stats in summary['categories'].items():
            total_cat = sum(stats.values())
            success_rate = (stats['passed'] / total_cat * 100) if total_cat > 0 else 0
            print(f"   {category}: {stats['passed']}/{total_cat} passed ({success_rate:.1f}%)")
        
        if summary['failed_tests']:
            print(f"\n❌ Failed Tests:")
            for test_name in summary['failed_tests']:
                print(f"   - {test_name}")
        
        # Save detailed results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_results": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "status": r.status,
                        "duration": r.duration,
                        "message": r.message,
                        "details": r.details
                    }
                    for r in runner.results
                ]
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        exit_code = 0 if summary['failed'] == 0 else 1
        
        if exit_code == 0:
            print("\n🎉 ALL TESTS PASSED! Ready for release.")
        else:
            print(f"\n⚠️ {summary['failed']} tests failed. Please review and fix issues before release.")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

