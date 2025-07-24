#!/usr/bin/env python3
"""
Performance Monitoring and Metrics System for Instant-DB

This module provides comprehensive performance monitoring capabilities:
- Real-time performance metrics collection
- Performance dashboard generation
- Bottleneck identification and analysis
- Automated performance alerts
- Historical performance tracking

Usage:
    from performance_monitoring import PerformanceMonitor, MetricsDashboard
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Monitor operations
    with monitor.track_operation("document_processing"):
        # Your operation here
        pass
    
    # Generate dashboard
    dashboard = MetricsDashboard(monitor)
    dashboard.generate_html_report("performance_report.html")
"""

import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, ContextManager
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from datetime import datetime, timedelta
import sqlite3
import statistics
from collections import defaultdict, deque

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta(self) -> float:
        """Memory change during operation."""
        return self.memory_end - self.memory_start

@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io: Dict[str, int] = field(default_factory=dict)
    
class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / 1024 / 1024 / 1024,
            disk_usage_percent=disk.percent
        )

class PerformanceDatabase:
    """SQLite database for storing performance metrics."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize performance metrics database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                memory_start REAL NOT NULL,
                memory_end REAL NOT NULL,
                memory_peak REAL NOT NULL,
                memory_delta REAL NOT NULL,
                cpu_percent REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                memory_available_gb REAL NOT NULL,
                disk_usage_percent REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_perf_operation 
            ON performance_metrics(operation)
        ''')
        
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_perf_start_time 
            ON performance_metrics(start_time)
        ''')
        
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_system_timestamp 
            ON system_metrics(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO performance_metrics 
            (operation, start_time, end_time, duration, memory_start, memory_end, 
             memory_peak, memory_delta, cpu_percent, success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.operation,
            metric.start_time,
            metric.end_time,
            metric.duration,
            metric.memory_start,
            metric.memory_end,
            metric.memory_peak,
            metric.memory_delta,
            metric.cpu_percent,
            metric.success,
            metric.error_message,
            json.dumps(metric.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def store_system_metric(self, metric: SystemMetrics):
        """Store a system metric."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO system_metrics 
            (timestamp, cpu_percent, memory_percent, memory_available_gb, disk_usage_percent)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            metric.timestamp,
            metric.cpu_percent,
            metric.memory_percent,
            metric.memory_available_gb,
            metric.disk_usage_percent
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_by_operation(self, operation: str, 
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics for a specific operation."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = '''
            SELECT * FROM performance_metrics 
            WHERE operation = ? 
            ORDER BY start_time DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor = conn.execute(query, (operation,))
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """Get summary statistics for an operation."""
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute('''
            SELECT 
                COUNT(*) as total_runs,
                AVG(duration) as avg_duration,
                MIN(duration) as min_duration,
                MAX(duration) as max_duration,
                AVG(memory_peak) as avg_memory_peak,
                MAX(memory_peak) as max_memory_peak,
                AVG(cpu_percent) as avg_cpu_percent,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
            FROM performance_metrics 
            WHERE operation = ?
        ''', (operation,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "operation": operation,
                "total_runs": result[0],
                "avg_duration": result[1] or 0,
                "min_duration": result[2] or 0,
                "max_duration": result[3] or 0,
                "avg_memory_peak": result[4] or 0,
                "max_memory_peak": result[5] or 0,
                "avg_cpu_percent": result[6] or 0,
                "success_count": result[7] or 0,
                "error_count": result[8] or 0,
                "success_rate": (result[7] or 0) / max(result[0], 1) * 100
            }
        
        return {}

class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, db_path: str = "./performance_metrics.db", 
                 system_monitoring_interval: float = 5.0):
        self.db_path = Path(db_path)
        self.collector = MetricsCollector()
        self.database = PerformanceDatabase(self.db_path)
        
        # In-memory metrics for real-time monitoring
        self.current_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 system metrics
        
        # System monitoring
        self.system_monitoring_interval = system_monitoring_interval
        self.system_monitoring_active = False
        self.system_monitoring_thread = None
        
        # Performance alerts
        self.alert_thresholds = {
            "memory_mb": 1000,  # Alert if memory usage > 1GB
            "duration_seconds": 30,  # Alert if operation > 30s
            "cpu_percent": 80,  # Alert if CPU > 80%
            "error_rate_percent": 10  # Alert if error rate > 10%
        }
        
        self.alerts: List[Dict[str, Any]] = []
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self.system_monitoring_active:
            return
        
        self.system_monitoring_active = True
        self.system_monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True
        )
        self.system_monitoring_thread.start()
        print("📊 System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self.system_monitoring_active = False
        if self.system_monitoring_thread:
            self.system_monitoring_thread.join(timeout=1)
        print("📊 System monitoring stopped")
    
    def _system_monitoring_loop(self):
        """Background loop for system monitoring."""
        while self.system_monitoring_active:
            try:
                metric = self.collector.get_system_metrics()
                self.system_metrics.append(metric)
                self.database.store_system_metric(metric)
                
                # Check for system alerts
                self._check_system_alerts(metric)
                
                time.sleep(self.system_monitoring_interval)
            except Exception as e:
                print(f"❌ System monitoring error: {e}")
                time.sleep(self.system_monitoring_interval)
    
    @contextmanager
    def track_operation(self, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> ContextManager[PerformanceMetric]:
        """Context manager to track operation performance."""
        metadata = metadata or {}
        
        # Start tracking
        start_time = time.time()
        memory_start = self.collector.get_memory_usage_mb()
        cpu_start = self.collector.get_cpu_percent()
        
        metric = PerformanceMetric(
            operation=operation_name,
            start_time=start_time,
            end_time=0,
            duration=0,
            memory_start=memory_start,
            memory_end=0,
            memory_peak=memory_start,
            cpu_percent=0,
            success=True,
            metadata=metadata
        )
        
        try:
            yield metric
            
        except Exception as e:
            metric.success = False
            metric.error_message = str(e)
            raise
            
        finally:
            # End tracking
            end_time = time.time()
            memory_end = self.collector.get_memory_usage_mb()
            cpu_end = self.collector.get_cpu_percent()
            
            metric.end_time = end_time
            metric.duration = end_time - start_time
            metric.memory_end = memory_end
            metric.memory_peak = max(metric.memory_peak, memory_end)
            metric.cpu_percent = max(cpu_end - cpu_start, 0)
            
            # Store metric
            self.current_metrics[operation_name].append(metric)
            self.database.store_metric(metric)
            
            # Check for performance alerts
            self._check_performance_alerts(metric)
            
            print(f"📊 {operation_name}: {metric.duration:.3f}s, "
                  f"{metric.memory_peak:.2f}MB peak, "
                  f"{'✅' if metric.success else '❌'}")
    
    def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check for performance alerts."""
        alerts = []
        
        if metric.memory_peak > self.alert_thresholds["memory_mb"]:
            alerts.append({
                "type": "high_memory",
                "operation": metric.operation,
                "value": metric.memory_peak,
                "threshold": self.alert_thresholds["memory_mb"],
                "timestamp": metric.end_time
            })
        
        if metric.duration > self.alert_thresholds["duration_seconds"]:
            alerts.append({
                "type": "slow_operation",
                "operation": metric.operation,
                "value": metric.duration,
                "threshold": self.alert_thresholds["duration_seconds"],
                "timestamp": metric.end_time
            })
        
        if metric.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "operation": metric.operation,
                "value": metric.cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"],
                "timestamp": metric.end_time
            })
        
        self.alerts.extend(alerts)
        
        # Print alerts
        for alert in alerts:
            print(f"⚠️  ALERT: {alert['type']} - {alert['operation']} "
                  f"({alert['value']:.2f} > {alert['threshold']})")
    
    def _check_system_alerts(self, metric: SystemMetrics):
        """Check for system-level alerts."""
        if metric.memory_percent > 90:
            self.alerts.append({
                "type": "system_memory_high",
                "value": metric.memory_percent,
                "threshold": 90,
                "timestamp": metric.timestamp
            })
            print(f"⚠️  SYSTEM ALERT: High memory usage ({metric.memory_percent:.1f}%)")
        
        if metric.cpu_percent > 95:
            self.alerts.append({
                "type": "system_cpu_high",
                "value": metric.cpu_percent,
                "threshold": 95,
                "timestamp": metric.timestamp
            })
            print(f"⚠️  SYSTEM ALERT: High CPU usage ({metric.cpu_percent:.1f}%)")
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        return self.database.get_operation_summary(operation)
    
    def get_recent_metrics(self, operation: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics for an operation."""
        return self.database.get_metrics_by_operation(operation, limit)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all operations
        cursor = conn.execute('''
            SELECT DISTINCT operation FROM performance_metrics
        ''')
        operations = [row[0] for row in cursor.fetchall()]
        
        # Get summary for each operation
        operation_summaries = {}
        for operation in operations:
            operation_summaries[operation] = self.get_operation_stats(operation)
        
        # Get overall stats
        cursor = conn.execute('''
            SELECT 
                COUNT(*) as total_operations,
                AVG(duration) as avg_duration,
                AVG(memory_peak) as avg_memory_peak,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as total_success,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as total_errors
            FROM performance_metrics
        ''')
        
        overall = cursor.fetchone()
        conn.close()
        
        return {
            "overall": {
                "total_operations": overall[0] or 0,
                "avg_duration": overall[1] or 0,
                "avg_memory_peak": overall[2] or 0,
                "total_success": overall[3] or 0,
                "total_errors": overall[4] or 0,
                "success_rate": (overall[3] or 0) / max(overall[0], 1) * 100
            },
            "by_operation": operation_summaries,
            "alerts_count": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else []
        }

class MetricsDashboard:
    """Generate performance dashboards and reports."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def generate_html_report(self, output_path: str):
        """Generate HTML performance report."""
        summary = self.monitor.get_performance_summary()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Instant-DB Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
        .alert {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Instant-DB Performance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>📈 Overall Performance</h2>
    <div class="metric">
        <strong>Total Operations:</strong> {summary['overall']['total_operations']}<br>
        <strong>Average Duration:</strong> {summary['overall']['avg_duration']:.3f}s<br>
        <strong>Average Memory Peak:</strong> {summary['overall']['avg_memory_peak']:.2f}MB<br>
        <strong>Success Rate:</strong> <span class="success">{summary['overall']['success_rate']:.1f}%</span><br>
        <strong>Total Errors:</strong> <span class="error">{summary['overall']['total_errors']}</span>
    </div>
    
    <h2>🔍 Performance by Operation</h2>
    <table>
        <tr>
            <th>Operation</th>
            <th>Total Runs</th>
            <th>Avg Duration (s)</th>
            <th>Avg Memory (MB)</th>
            <th>Success Rate (%)</th>
            <th>Errors</th>
        </tr>
"""
        
        for operation, stats in summary['by_operation'].items():
            html_content += f"""
        <tr>
            <td>{operation}</td>
            <td>{stats['total_runs']}</td>
            <td>{stats['avg_duration']:.3f}</td>
            <td>{stats['avg_memory_peak']:.2f}</td>
            <td class="{'success' if stats['success_rate'] > 95 else 'error'}">{stats['success_rate']:.1f}</td>
            <td class="{'success' if stats['error_count'] == 0 else 'error'}">{stats['error_count']}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>⚠️ Recent Alerts</h2>
"""
        
        if summary['recent_alerts']:
            for alert in summary['recent_alerts']:
                html_content += f"""
    <div class="metric alert">
        <strong>{alert['type']}:</strong> {alert.get('operation', 'System')} - 
        Value: {alert['value']:.2f}, Threshold: {alert['threshold']}<br>
        <small>Time: {datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
"""
        else:
            html_content += "<p>No recent alerts 🎉</p>"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Performance report generated: {output_path}")
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.monitor.get_performance_summary()
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Operations: {summary['overall']['total_operations']}")
        print(f"Average Duration: {summary['overall']['avg_duration']:.3f}s")
        print(f"Average Memory Peak: {summary['overall']['avg_memory_peak']:.2f}MB")
        print(f"Success Rate: {summary['overall']['success_rate']:.1f}%")
        print(f"Total Errors: {summary['overall']['total_errors']}")
        
        print(f"\n🔍 PERFORMANCE BY OPERATION")
        print("-" * 50)
        for operation, stats in summary['by_operation'].items():
            print(f"{operation}:")
            print(f"  Runs: {stats['total_runs']}, "
                  f"Avg: {stats['avg_duration']:.3f}s, "
                  f"Memory: {stats['avg_memory_peak']:.2f}MB, "
                  f"Success: {stats['success_rate']:.1f}%")
        
        if summary['recent_alerts']:
            print(f"\n⚠️  RECENT ALERTS ({len(summary['recent_alerts'])})")
            print("-" * 50)
            for alert in summary['recent_alerts'][-5:]:  # Show last 5
                print(f"  {alert['type']}: {alert.get('operation', 'System')} "
                      f"({alert['value']:.2f} > {alert['threshold']})")


def create_performance_test():
    """Create a test to demonstrate performance monitoring."""
    
    def test_performance_monitoring():
        """Test the performance monitoring system."""
        print("🧪 TESTING PERFORMANCE MONITORING")
        print("=" * 50)
        
        # Initialize monitor
        monitor = PerformanceMonitor("./test_performance.db")
        monitor.start_system_monitoring()
        
        try:
            # Test various operations
            with monitor.track_operation("test_fast_operation"):
                time.sleep(0.1)  # Fast operation
            
            with monitor.track_operation("test_slow_operation"):
                time.sleep(2.0)  # Slow operation
            
            with monitor.track_operation("test_memory_operation"):
                # Simulate memory usage
                data = [i for i in range(100000)]  # Use some memory
                time.sleep(0.5)
                del data
            
            # Test error handling
            try:
                with monitor.track_operation("test_error_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected error
            
            # Wait for system metrics
            time.sleep(6)
            
            # Generate report
            dashboard = MetricsDashboard(monitor)
            dashboard.print_summary()
            dashboard.generate_html_report("./test_performance_report.html")
            
        finally:
            monitor.stop_system_monitoring()
        
        print("\n✅ Performance monitoring test completed!")
    
    return test_performance_monitoring


if __name__ == "__main__":
    # Run performance monitoring test
    test = create_performance_test()
    test()

