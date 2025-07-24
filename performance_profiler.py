#!/usr/bin/env python3
"""
Comprehensive Performance Profiler for Instant-DB

This script profiles various aspects of Instant-DB performance:
- Document processing speed and memory usage
- Search latency and throughput
- Memory optimization opportunities
- Database operation performance
- Embedding generation performance

Usage:
    python performance_profiler.py [--dataset-path PATH] [--output-dir DIR]
"""

import sys
import os
import json
import time
import psutil
import tracemalloc
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Instant-DB"))

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    duration: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    additional_data: Dict[str, Any]

class PerformanceProfiler:
    """Comprehensive performance profiler for Instant-DB."""
    
    def __init__(self, output_dir: str = "./performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
    def start_memory_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()
        gc.collect()  # Clean up before measurement
        
    def stop_memory_tracking(self) -> tuple:
        """Stop memory tracking and return peak memory usage."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return current / 1024 / 1024, peak / 1024 / 1024  # Convert to MB
        
    def measure_operation(self, operation_name: str, func, *args, **kwargs):
        """Measure performance of a specific operation."""
        print(f"📊 Measuring: {operation_name}")
        
        # Start measurements
        self.start_memory_tracking()
        start_time = time.time()
        cpu_before = self.process.cpu_percent()
        
        # Execute operation
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            print(f"   ❌ Error: {error}")
        
        # Stop measurements
        end_time = time.time()
        duration = end_time - start_time
        memory_current, memory_peak = self.stop_memory_tracking()
        cpu_after = self.process.cpu_percent()
        cpu_percent = max(cpu_after - cpu_before, 0)
        
        # Create metrics
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            memory_peak=memory_peak,
            memory_current=memory_current,
            cpu_percent=cpu_percent,
            additional_data={
                "success": success,
                "error": error,
                "result_size": len(str(result)) if result else 0
            }
        )
        
        self.metrics.append(metrics)
        
        print(f"   ⏱️  Duration: {duration:.3f}s")
        print(f"   🧠 Memory Peak: {memory_peak:.2f}MB")
        print(f"   💾 Memory Current: {memory_current:.2f}MB")
        print(f"   🔥 CPU: {cpu_percent:.1f}%")
        
        return result, metrics
    
    def profile_document_processing(self, dataset_path: str):
        """Profile document processing performance."""
        print("\n🔍 PROFILING DOCUMENT PROCESSING")
        print("=" * 50)
        
        try:
            from instant_db import InstantDB
        except ImportError:
            print("❌ Cannot import instant_db. Make sure it's installed.")
            return
        
        # Create temporary database
        temp_db_path = tempfile.mkdtemp(prefix="perf_test_")
        
        try:
            # Initialize database
            db, init_metrics = self.measure_operation(
                "Database Initialization",
                lambda: InstantDB(
                    db_path=temp_db_path,
                    embedding_provider="sentence-transformers",
                    embedding_model="all-MiniLM-L6-v2"
                )
            )
            
            if not db:
                print("❌ Failed to initialize database")
                return
            
            # Find test files
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                print(f"❌ Dataset path not found: {dataset_path}")
                return
                
            test_files = list(dataset_path.glob("*.txt"))
            if not test_files:
                print(f"❌ No .txt files found in {dataset_path}")
                return
                
            print(f"📄 Found {len(test_files)} test files")
            
            # Profile individual file processing
            processing_times = []
            file_sizes = []
            
            for i, file_path in enumerate(test_files[:5]):  # Limit to 5 files for profiling
                file_size = file_path.stat().st_size / 1024  # KB
                file_sizes.append(file_size)
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Profile document addition
                result, metrics = self.measure_operation(
                    f"Process File {i+1}: {file_path.name} ({file_size:.1f}KB)",
                    lambda: db.add_document(
                        content=content,
                        metadata={
                            "title": file_path.stem.replace('_', ' ').title(),
                            "source": str(file_path),
                            "type": "text",
                            "file_name": file_path.name,
                            "file_size_kb": file_size
                        },
                        document_id=f"test_doc_{i}"
                    )
                )
                
                processing_times.append(metrics.duration)
                
                if result and result.get('status') == 'success':
                    chunks = result.get('chunks_processed', 0)
                    print(f"   ✅ Processed {chunks} chunks")
                    metrics.additional_data['chunks_processed'] = chunks
                else:
                    print(f"   ❌ Processing failed")
            
            # Calculate processing statistics
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
                avg_file_size = sum(file_sizes) / len(file_sizes)
                processing_rate = avg_file_size / avg_processing_time if avg_processing_time > 0 else 0
                
                print(f"\n📈 PROCESSING STATISTICS:")
                print(f"   Average processing time: {avg_processing_time:.3f}s")
                print(f"   Average file size: {avg_file_size:.1f}KB")
                print(f"   Processing rate: {processing_rate:.1f}KB/s")
                
                # Store summary metrics
                summary_metrics = PerformanceMetrics(
                    operation="Document Processing Summary",
                    duration=avg_processing_time,
                    memory_peak=max(m.memory_peak for m in self.metrics[-len(test_files):]),
                    memory_current=0,
                    cpu_percent=0,
                    additional_data={
                        "avg_processing_time": avg_processing_time,
                        "avg_file_size_kb": avg_file_size,
                        "processing_rate_kb_per_sec": processing_rate,
                        "files_processed": len(processing_times)
                    }
                )
                self.metrics.append(summary_metrics)
            
        finally:
            # Cleanup
            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path)
    
    def profile_search_performance(self, dataset_path: str):
        """Profile search performance."""
        print("\n🔍 PROFILING SEARCH PERFORMANCE")
        print("=" * 50)
        
        try:
            from instant_db import InstantDB
        except ImportError:
            print("❌ Cannot import instant_db. Make sure it's installed.")
            return
        
        # Create temporary database with test data
        temp_db_path = tempfile.mkdtemp(prefix="search_perf_")
        
        try:
            # Initialize and populate database
            db = InstantDB(
                db_path=temp_db_path,
                embedding_provider="sentence-transformers",
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Add test documents
            dataset_path = Path(dataset_path)
            test_files = list(dataset_path.glob("*.txt"))[:3]  # Use 3 files for search testing
            
            for i, file_path in enumerate(test_files):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                db.add_document(
                    content=content,
                    metadata={"title": file_path.stem, "source": str(file_path)},
                    document_id=f"search_test_{i}"
                )
            
            # Define test queries
            test_queries = [
                "pricing objections",
                "security features", 
                "action items",
                "CloudSync Pro",
                "data migration"
            ]
            
            # Profile search operations
            search_times = []
            
            for query in test_queries:
                result, metrics = self.measure_operation(
                    f"Search: '{query}'",
                    lambda q=query: db.search(q, top_k=5)
                )
                
                search_times.append(metrics.duration)
                
                if result:
                    print(f"   📊 Found {len(result)} results")
                    if result:
                        print(f"   🎯 Top score: {result[0].get('similarity', 0):.3f}")
                        metrics.additional_data.update({
                            'num_results': len(result),
                            'top_score': result[0].get('similarity', 0) if result else 0
                        })
            
            # Search performance statistics
            if search_times:
                avg_search_time = sum(search_times) / len(search_times)
                max_search_time = max(search_times)
                min_search_time = min(search_times)
                
                print(f"\n📈 SEARCH STATISTICS:")
                print(f"   Average search time: {avg_search_time:.3f}s")
                print(f"   Min search time: {min_search_time:.3f}s")
                print(f"   Max search time: {max_search_time:.3f}s")
                print(f"   Search throughput: {1/avg_search_time:.1f} queries/sec")
                
                # Store summary metrics
                summary_metrics = PerformanceMetrics(
                    operation="Search Performance Summary",
                    duration=avg_search_time,
                    memory_peak=max(m.memory_peak for m in self.metrics[-len(test_queries):]),
                    memory_current=0,
                    cpu_percent=0,
                    additional_data={
                        "avg_search_time": avg_search_time,
                        "min_search_time": min_search_time,
                        "max_search_time": max_search_time,
                        "search_throughput": 1/avg_search_time,
                        "queries_tested": len(search_times)
                    }
                )
                self.metrics.append(summary_metrics)
                
        finally:
            # Cleanup
            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path)
    
    def profile_memory_usage(self):
        """Profile memory usage patterns."""
        print("\n🧠 PROFILING MEMORY USAGE")
        print("=" * 50)
        
        # Get current memory info
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        print(f"Current RSS: {memory_info.rss / 1024 / 1024:.2f}MB")
        print(f"Current VMS: {memory_info.vms / 1024 / 1024:.2f}MB")
        print(f"Memory percent: {memory_percent:.2f}%")
        
        # System memory info
        system_memory = psutil.virtual_memory()
        print(f"System total: {system_memory.total / 1024 / 1024 / 1024:.2f}GB")
        print(f"System available: {system_memory.available / 1024 / 1024 / 1024:.2f}GB")
        print(f"System used: {system_memory.percent:.1f}%")
        
        # Store memory metrics
        memory_metrics = PerformanceMetrics(
            operation="Memory Usage Analysis",
            duration=0,
            memory_peak=memory_info.rss / 1024 / 1024,
            memory_current=memory_info.rss / 1024 / 1024,
            cpu_percent=self.process.cpu_percent(),
            additional_data={
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "memory_percent": memory_percent,
                "system_total_gb": system_memory.total / 1024 / 1024 / 1024,
                "system_available_gb": system_memory.available / 1024 / 1024 / 1024,
                "system_used_percent": system_memory.percent
            }
        )
        self.metrics.append(memory_metrics)
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n📊 GENERATING PERFORMANCE REPORT")
        print("=" * 50)
        
        # Calculate overall statistics
        processing_metrics = [m for m in self.metrics if "Process File" in m.operation]
        search_metrics = [m for m in self.metrics if "Search:" in m.operation]
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            "performance_summary": {
                "total_operations": len(self.metrics),
                "total_duration": sum(m.duration for m in self.metrics),
                "peak_memory_mb": max(m.memory_peak for m in self.metrics) if self.metrics else 0,
                "avg_cpu_percent": sum(m.cpu_percent for m in self.metrics) / len(self.metrics) if self.metrics else 0
            },
            "document_processing": {
                "operations_count": len(processing_metrics),
                "avg_duration": sum(m.duration for m in processing_metrics) / len(processing_metrics) if processing_metrics else 0,
                "avg_memory_peak": sum(m.memory_peak for m in processing_metrics) / len(processing_metrics) if processing_metrics else 0,
                "total_chunks": sum(m.additional_data.get('chunks_processed', 0) for m in processing_metrics)
            },
            "search_performance": {
                "operations_count": len(search_metrics),
                "avg_duration": sum(m.duration for m in search_metrics) / len(search_metrics) if search_metrics else 0,
                "avg_memory_peak": sum(m.memory_peak for m in search_metrics) / len(search_metrics) if search_metrics else 0,
                "throughput_qps": 1 / (sum(m.duration for m in search_metrics) / len(search_metrics)) if search_metrics else 0
            },
            "detailed_metrics": [asdict(m) for m in self.metrics]
        }
        
        # Save report
        report_file = self.output_dir / f"performance_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Print summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Total operations: {report['performance_summary']['total_operations']}")
        print(f"   Total duration: {report['performance_summary']['total_duration']:.3f}s")
        print(f"   Peak memory: {report['performance_summary']['peak_memory_mb']:.2f}MB")
        print(f"   Avg CPU: {report['performance_summary']['avg_cpu_percent']:.1f}%")
        
        if processing_metrics:
            print(f"   Document processing avg: {report['document_processing']['avg_duration']:.3f}s")
            
        if search_metrics:
            print(f"   Search avg: {report['search_performance']['avg_duration']:.3f}s")
            print(f"   Search throughput: {report['search_performance']['throughput_qps']:.1f} queries/sec")
        
        return report
    
    def run_full_profile(self, dataset_path: str = "./Instant-DB/demo_dataset"):
        """Run complete performance profiling suite."""
        print("🚀 STARTING COMPREHENSIVE PERFORMANCE PROFILING")
        print("=" * 60)
        
        # Profile memory baseline
        self.profile_memory_usage()
        
        # Profile document processing
        self.profile_document_processing(dataset_path)
        
        # Profile search performance
        self.profile_search_performance(dataset_path)
        
        # Generate final report
        report = self.generate_report()
        
        print("\n✅ PROFILING COMPLETE!")
        return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Instant-DB Performance Profiler")
    parser.add_argument("--dataset-path", default="./Instant-DB/demo_dataset", 
                       help="Path to test dataset")
    parser.add_argument("--output-dir", default="./performance_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create profiler and run
    profiler = PerformanceProfiler(args.output_dir)
    
    try:
        report = profiler.run_full_profile(args.dataset_path)
        print(f"\n🎉 Performance profiling completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

