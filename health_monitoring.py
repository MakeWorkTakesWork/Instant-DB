#!/usr/bin/env python3
"""
Health Check and Monitoring System for Instant-DB

This module provides comprehensive health monitoring capabilities:
- Health check endpoints for load balancers
- System resource monitoring
- Database health checks
- Service dependency monitoring
- Alerting and notification system
- Metrics collection and reporting

Usage:
    from health_monitoring import HealthMonitor, HealthCheckServer
    
    # Initialize health monitor
    monitor = HealthMonitor()
    
    # Start health check server
    server = HealthCheckServer(monitor, port=8080)
    server.start()
"""

import time
import json
import psutil
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from contextlib import contextmanager
import requests
import socket

@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    last_check: datetime
    response_time_ms: float
    details: Dict[str, Any]

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_connections: int
    load_average: List[float]

class HealthChecker:
    """Base class for health checkers."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
    
    def check(self) -> HealthStatus:
        """Perform health check."""
        start_time = time.time()
        
        try:
            result = self._perform_check()
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                name=self.name,
                status="healthy" if result["healthy"] else "unhealthy",
                message=result.get("message", "OK"),
                last_check=datetime.now(),
                response_time_ms=response_time,
                details=result.get("details", {})
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                name=self.name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check."""
        raise NotImplementedError

class DatabaseHealthChecker(HealthChecker):
    """Health checker for database connectivity."""
    
    def __init__(self, db_path: Path, name: str = "database"):
        super().__init__(name)
        self.db_path = db_path
    
    def _perform_check(self) -> Dict[str, Any]:
        """Check database health."""
        if not self.db_path.exists():
            return {
                "healthy": False,
                "message": "Database file not found",
                "details": {"db_path": str(self.db_path)}
            }
        
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                # Test basic query
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master")
                table_count = cursor.fetchone()[0]
                
                # Check database size
                db_size_mb = self.db_path.stat().st_size / 1024 / 1024
                
                # Check for corruption
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                
                is_healthy = integrity_result == "ok"
                
                return {
                    "healthy": is_healthy,
                    "message": "Database accessible" if is_healthy else f"Integrity check failed: {integrity_result}",
                    "details": {
                        "table_count": table_count,
                        "size_mb": db_size_mb,
                        "integrity_check": integrity_result
                    }
                }
                
        except sqlite3.Error as e:
            return {
                "healthy": False,
                "message": f"Database error: {str(e)}",
                "details": {"error_type": type(e).__name__}
            }

class SystemResourceChecker(HealthChecker):
    """Health checker for system resources."""
    
    def __init__(self, name: str = "system_resources", 
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        super().__init__(name)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def _perform_check(self) -> Dict[str, Any]:
        """Check system resource health."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / 1024 / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Load average
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # Network connections
        network_connections = len(psutil.net_connections())
        
        # Determine health status
        issues = []
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory_percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory_percent:.1f}%")
        
        if disk_percent > self.disk_threshold:
            issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        is_healthy = len(issues) == 0
        status = "healthy" if is_healthy else "degraded" if len(issues) == 1 else "unhealthy"
        
        return {
            "healthy": is_healthy,
            "message": "System resources OK" if is_healthy else f"Issues: {', '.join(issues)}",
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available_gb,
                "disk_percent": disk_percent,
                "disk_free_gb": disk_free_gb,
                "load_average": load_avg,
                "network_connections": network_connections,
                "thresholds": {
                    "cpu": self.cpu_threshold,
                    "memory": self.memory_threshold,
                    "disk": self.disk_threshold
                }
            }
        }

class VectorStoreHealthChecker(HealthChecker):
    """Health checker for vector store connectivity."""
    
    def __init__(self, db_path: Path, name: str = "vector_store"):
        super().__init__(name)
        self.db_path = db_path
    
    def _perform_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            # Check if ChromaDB directory exists
            chroma_path = self.db_path / "chroma"
            if not chroma_path.exists():
                return {
                    "healthy": False,
                    "message": "Vector store directory not found",
                    "details": {"chroma_path": str(chroma_path)}
                }
            
            # Try to initialize ChromaDB client
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            
            # List collections
            collections = client.list_collections()
            collection_count = len(collections)
            
            # Get collection info if available
            collection_info = {}
            if collections:
                collection = collections[0]
                collection_info = {
                    "name": collection.name,
                    "count": collection.count()
                }
            
            return {
                "healthy": True,
                "message": f"Vector store accessible with {collection_count} collections",
                "details": {
                    "collection_count": collection_count,
                    "collection_info": collection_info,
                    "chroma_path": str(chroma_path)
                }
            }
            
        except ImportError:
            return {
                "healthy": False,
                "message": "ChromaDB not available",
                "details": {"error": "chromadb module not found"}
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Vector store error: {str(e)}",
                "details": {"error_type": type(e).__name__}
            }

class EmbeddingServiceChecker(HealthChecker):
    """Health checker for embedding service."""
    
    def __init__(self, name: str = "embedding_service"):
        super().__init__(name)
    
    def _perform_check(self) -> Dict[str, Any]:
        """Check embedding service health."""
        try:
            # Try to load sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Test with a small model load (cached)
            model_name = "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            
            # Test encoding
            test_text = "Health check test"
            embedding = model.encode([test_text])
            
            return {
                "healthy": True,
                "message": "Embedding service operational",
                "details": {
                    "model": model_name,
                    "embedding_dimension": len(embedding[0]),
                    "test_successful": True
                }
            }
            
        except ImportError:
            return {
                "healthy": False,
                "message": "Sentence transformers not available",
                "details": {"error": "sentence_transformers module not found"}
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Embedding service error: {str(e)}",
                "details": {"error_type": type(e).__name__}
            }

class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self, db_path: str = "./instant_db_database"):
        self.db_path = Path(db_path)
        self.checkers: List[HealthChecker] = []
        self.last_results: Dict[str, HealthStatus] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 30  # seconds
        
        # Initialize default health checkers
        self._setup_default_checkers()
    
    def _setup_default_checkers(self):
        """Setup default health checkers."""
        self.add_checker(SystemResourceChecker())
        
        # Add database checker if metadata DB exists
        metadata_db = self.db_path / "metadata.db"
        if metadata_db.exists():
            self.add_checker(DatabaseHealthChecker(metadata_db))
        
        # Add vector store checker
        self.add_checker(VectorStoreHealthChecker(self.db_path))
        
        # Add embedding service checker
        self.add_checker(EmbeddingServiceChecker())
    
    def add_checker(self, checker: HealthChecker):
        """Add a health checker."""
        self.checkers.append(checker)
    
    def remove_checker(self, name: str):
        """Remove a health checker by name."""
        self.checkers = [c for c in self.checkers if c.name != name]
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for checker in self.checkers:
            try:
                result = checker.check()
                results[checker.name] = result
                self.last_results[checker.name] = result
            except Exception as e:
                # Create error status if checker itself fails
                error_status = HealthStatus(
                    name=checker.name,
                    status="unhealthy",
                    message=f"Health checker failed: {str(e)}",
                    last_check=datetime.now(),
                    response_time_ms=0,
                    details={"checker_error": str(e)}
                )
                results[checker.name] = error_status
                self.last_results[checker.name] = error_status
        
        return results
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.check_all()
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        # Count statuses
        status_counts = {
            "healthy": sum(1 for s in statuses if s == "healthy"),
            "degraded": sum(1 for s in statuses if s == "degraded"),
            "unhealthy": sum(1 for s in statuses if s == "unhealthy")
        }
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "status_counts": status_counts,
            "total_checks": len(results),
            "checks": {name: asdict(status) for name, status in results.items()}
        }
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"🔍 Health monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        print("🔍 Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                results = self.check_all()
                
                # Log any unhealthy services
                for name, status in results.items():
                    if status.status != "healthy":
                        print(f"⚠️  Health check warning: {name} is {status.status} - {status.message}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ Health monitoring error: {e}")
                time.sleep(self.check_interval)

class HealthCheckServer:
    """Flask server for health check endpoints."""
    
    def __init__(self, health_monitor: HealthMonitor, port: int = 8080):
        self.health_monitor = health_monitor
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Basic health check endpoint."""
            status = self.health_monitor.get_overall_status()
            
            # Return appropriate HTTP status code
            http_status = 200 if status['overall_status'] == 'healthy' else 503
            
            return jsonify(status), http_status
        
        @self.app.route('/health/live', methods=['GET'])
        def liveness_check():
            """Kubernetes liveness probe endpoint."""
            return jsonify({
                "status": "alive",
                "timestamp": datetime.now().isoformat()
            }), 200
        
        @self.app.route('/health/ready', methods=['GET'])
        def readiness_check():
            """Kubernetes readiness probe endpoint."""
            status = self.health_monitor.get_overall_status()
            
            # Ready if all critical services are healthy
            critical_services = ["database", "vector_store", "system_resources"]
            critical_healthy = all(
                status['checks'].get(service, {}).get('status') == 'healthy'
                for service in critical_services
                if service in status['checks']
            )
            
            if critical_healthy:
                return jsonify({
                    "status": "ready",
                    "timestamp": datetime.now().isoformat()
                }), 200
            else:
                return jsonify({
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "details": status
                }), 503
        
        @self.app.route('/health/detailed', methods=['GET'])
        def detailed_health():
            """Detailed health information."""
            return jsonify(self.health_monitor.get_overall_status()), 200
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus-style metrics endpoint."""
            status = self.health_monitor.get_overall_status()
            
            metrics_text = []
            
            # Health status metrics
            for check_name, check_data in status['checks'].items():
                status_value = 1 if check_data['status'] == 'healthy' else 0
                metrics_text.append(f'instant_db_health_status{{service="{check_name}"}} {status_value}')
                
                # Response time metric
                response_time = check_data.get('response_time_ms', 0)
                metrics_text.append(f'instant_db_health_response_time_ms{{service="{check_name}"}} {response_time}')
            
            # System metrics
            system_check = status['checks'].get('system_resources', {})
            if system_check:
                details = system_check.get('details', {})
                if 'cpu_percent' in details:
                    metrics_text.append(f'instant_db_cpu_percent {details["cpu_percent"]}')
                if 'memory_percent' in details:
                    metrics_text.append(f'instant_db_memory_percent {details["memory_percent"]}')
                if 'disk_percent' in details:
                    metrics_text.append(f'instant_db_disk_percent {details["disk_percent"]}')
            
            return '\n'.join(metrics_text), 200, {'Content-Type': 'text/plain'}
    
    def start(self, debug: bool = False):
        """Start the health check server."""
        print(f"🌐 Starting health check server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

def create_health_monitoring_demo():
    """Create a demo of the health monitoring system."""
    
    def demo_health_monitoring():
        """Demonstrate health monitoring capabilities."""
        print("🔍 HEALTH MONITORING SYSTEM DEMO")
        print("=" * 50)
        
        # Initialize health monitor
        monitor = HealthMonitor("./demo_health_db")
        
        # Run health checks
        print("📊 Running health checks...")
        status = monitor.get_overall_status()
        
        print(f"\n🎯 Overall Status: {status['overall_status'].upper()}")
        print(f"📅 Timestamp: {status['timestamp']}")
        print(f"📈 Total Checks: {status['total_checks']}")
        
        print(f"\n📊 Status Summary:")
        for status_type, count in status['status_counts'].items():
            print(f"   {status_type.title()}: {count}")
        
        print(f"\n🔍 Individual Check Results:")
        for name, check_data in status['checks'].items():
            status_emoji = "✅" if check_data['status'] == 'healthy' else "⚠️" if check_data['status'] == 'degraded' else "❌"
            print(f"   {status_emoji} {name}: {check_data['status']} ({check_data['response_time_ms']:.1f}ms)")
            print(f"      {check_data['message']}")
        
        # Test monitoring alerts
        print(f"\n🚨 Testing monitoring alerts...")
        monitor.start_monitoring()
        time.sleep(2)  # Let it run for a bit
        monitor.stop_monitoring()
        
        print("\n✅ Health monitoring demo completed!")
    
    return demo_health_monitoring


if __name__ == "__main__":
    # Run health monitoring demo
    demo = create_health_monitoring_demo()
    demo()

