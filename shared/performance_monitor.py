"""
Comprehensive Performance Monitoring for Nutrition Agent

This module provides real-time performance monitoring including:
- System resource metrics (CPU, memory, disk)
- Response time tracking and analysis
- Throughput monitoring
- Error rate tracking
- Performance alerts and thresholds
- Historical metric storage
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from collections import deque
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a single performance metric."""
    timestamp: datetime
    value: float
    metric_type: str
    metadata: Dict[str, Any]

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, agent_type: str = "nutrition", max_history: int = 1000):
        self.agent_type = agent_type
        self.max_history = max_history
        
        # Metric storage
        self.response_times: deque = deque(maxlen=max_history)
        self.throughput_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.error_metrics: deque = deque(maxlen=max_history)
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 5000.0,
            "error_rate": 0.05
        }
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        
        # Alert history
        self.alerts: List[Dict[str, Any]] = []
        self.max_alerts = 100
        
        # Performance statistics
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "avg_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0,
            "requests_per_second": 0.0,
            "error_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the performance monitor."""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"PerformanceMonitor initialized for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceMonitor: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"PerformanceMonitor cleaned up for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def record_response_time(self, endpoint: str, duration_ms: float, user_id: Optional[str] = None):
        """Record response time for an endpoint."""
        try:
            timestamp = datetime.utcnow()
            metric = PerformanceMetric(
                timestamp=timestamp,
                value=duration_ms,
                metric_type="response_time",
                metadata={
                    "endpoint": endpoint,
                    "user_id": user_id
                }
            )
            
            self.response_times.append(metric)
            
            # Update statistics
            self.request_count += 1
            self.stats["total_requests"] += 1
            
            # Update response time stats
            if duration_ms < self.stats["min_response_time"]:
                self.stats["min_response_time"] = duration_ms
            if duration_ms > self.stats["max_response_time"]:
                self.stats["max_response_time"] = duration_ms
            
            # Calculate moving average
            current_avg = self.stats["avg_response_time"]
            if self.stats["total_requests"] == 1:
                self.stats["avg_response_time"] = duration_ms
            else:
                alpha = 0.1  # Smoothing factor
                self.stats["avg_response_time"] = alpha * duration_ms + (1 - alpha) * current_avg
            
            # Calculate percentiles
            self._update_percentiles()
            
            # Check for alerts
            self._check_alerts("response_time", duration_ms, endpoint)
            
        except Exception as e:
            logger.error(f"Failed to record response time: {e}")
    
    def record_error(self, endpoint: str, error_type: str, error_message: str, user_id: Optional[str] = None):
        """Record an error occurrence."""
        try:
            timestamp = datetime.utcnow()
            metric = PerformanceMetric(
                timestamp=timestamp,
                value=1.0,  # Count as 1 error
                metric_type="error",
                metadata={
                    "endpoint": endpoint,
                    "error_type": error_type,
                    "error_message": error_message,
                    "user_id": user_id
                }
            )
            
            self.error_metrics.append(metric)
            
            # Update error statistics
            self.error_count += 1
            self.stats["total_errors"] += 1
            self.stats["error_rate"] = self.error_count / max(self.request_count, 1)
            
            # Check for alerts
            self._check_alerts("error_rate", self.stats["error_rate"], endpoint)
            
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def record_throughput(self, requests_per_second: float):
        """Record current throughput."""
        try:
            timestamp = datetime.utcnow()
            metric = PerformanceMetric(
                timestamp=timestamp,
                value=requests_per_second,
                metric_type="throughput",
                metadata={}
            )
            
            self.throughput_metrics.append(metric)
            
            # Update throughput stats
            self.stats["requests_per_second"] = requests_per_second
            
        except Exception as e:
            logger.error(f"Failed to record throughput: {e}")
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Network usage
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
            # Store metrics
            self.system_metrics.append(metrics)
            
            # Check for alerts
            self._check_alerts("cpu_percent", cpu_percent, "system")
            self._check_alerts("memory_percent", memory_percent, "system")
            self._check_alerts("disk_usage_percent", disk_usage_percent, "system")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_used_gb=0.0,
                disk_free_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Get current system metrics
            current_system = await self.get_system_metrics()
            
            # Calculate uptime
            uptime = datetime.utcnow() - self.start_time
            
            summary = {
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "performance_stats": self.stats.copy(),
                "system_metrics": {
                    "cpu_percent": current_system.cpu_percent,
                    "memory_percent": current_system.memory_percent,
                    "disk_usage_percent": current_system.disk_usage_percent,
                    "memory_used_mb": current_system.memory_used_mb,
                    "memory_available_mb": current_system.memory_available_mb,
                    "disk_used_gb": current_system.disk_used_gb,
                    "disk_free_gb": current_system.disk_free_gb
                },
                "recent_alerts": self.alerts[-10:] if self.alerts else [],
                "metric_counts": {
                    "response_times": len(self.response_times),
                    "throughput_metrics": len(self.throughput_metrics),
                    "system_metrics": len(self.system_metrics),
                    "error_metrics": len(self.error_metrics)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a specific metric."""
        if metric_name in self.alert_thresholds:
            self.alert_thresholds[metric_name] = threshold
            logger.info(f"Set alert threshold for {metric_name}: {threshold}")
        else:
            logger.warning(f"Unknown metric for alert threshold: {metric_name}")
    
    def get_metric_history(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics for a specific type."""
        try:
            if metric_type == "response_time":
                metrics = list(self.response_times)[-limit:]
            elif metric_type == "throughput":
                metrics = list(self.throughput_metrics)[-limit:]
            elif metric_type == "system":
                metrics = list(self.system_metrics)[-limit:]
            elif metric_type == "error":
                metrics = list(self.error_metrics)[-limit:]
            else:
                return []
            
            return [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "metadata": metric.metadata
                }
                for metric in metrics
            ]
            
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []
    
    def _update_percentiles(self):
        """Update percentile calculations for response times."""
        try:
            if len(self.response_times) < 2:
                return
            
            # Extract response time values
            values = [metric.value for metric in self.response_times]
            values.sort()
            
            # Calculate percentiles
            n = len(values)
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)
            
            self.stats["p95_response_time"] = values[p95_index] if p95_index < n else values[-1]
            self.stats["p99_response_time"] = values[p99_index] if p99_index < n else values[-1]
            
        except Exception as e:
            logger.error(f"Failed to update percentiles: {e}")
    
    def _check_alerts(self, metric_name: str, value: float, source: str):
        """Check if metric exceeds alert threshold."""
        try:
            threshold = self.alert_thresholds.get(metric_name)
            if threshold and value > threshold:
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metric_name": metric_name,
                    "value": value,
                    "threshold": threshold,
                    "source": source,
                    "severity": "warning" if value < threshold * 1.5 else "critical"
                }
                
                self.alerts.append(alert)
                
                # Keep only recent alerts
                if len(self.alerts) > self.max_alerts:
                    self.alerts = self.alerts[-self.max_alerts:]
                
                logger.warning(f"Performance alert: {metric_name} = {value} (threshold: {threshold})")
                
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Performance monitoring loop started")
        
        while True:
            try:
                # Update system metrics every 30 seconds
                await self.get_system_metrics()
                
                # Calculate current throughput
                if self.request_count > 0:
                    uptime = (datetime.utcnow() - self.start_time).total_seconds()
                    current_throughput = self.request_count / uptime
                    self.record_throughput(current_throughput)
                
                # Log performance summary every 5 minutes
                if self.request_count % 1000 == 0 and self.request_count > 0:
                    summary = await self.get_performance_summary()
                    logger.info(f"Performance Summary: {summary}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Performance monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _calculate_statistics(self, metrics: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of metrics."""
        try:
            if not metrics:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0
                }
            
            count = len(metrics)
            min_val = min(metrics)
            max_val = max(metrics)
            mean_val = sum(metrics) / count
            
            # Calculate standard deviation
            variance = sum((x - mean_val) ** 2 for x in metrics) / count
            std_val = variance ** 0.5
            
            return {
                "count": count,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0
            }
    
    async def generate_performance_report(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Filter metrics within time range
            response_times = [
                metric for metric in self.response_times
                if start_time <= metric.timestamp <= end_time
            ]
            
            error_metrics = [
                metric for metric in self.error_metrics
                if start_time <= metric.timestamp <= end_time
            ]
            
            system_metrics = [
                metric for metric in self.system_metrics
                if start_time <= metric.timestamp <= end_time
            ]
            
            # Calculate statistics
            response_time_values = [metric.value for metric in response_times]
            response_time_stats = self._calculate_statistics(response_time_values)
            
            # Calculate error rate
            total_requests = len(response_times)
            total_errors = len(error_metrics)
            error_rate = total_errors / max(total_requests, 1)
            
            # Calculate system averages
            if system_metrics:
                avg_cpu = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
                avg_memory = sum(m.memory_percent for m in system_metrics) / len(system_metrics)
                avg_disk = sum(m.disk_usage_percent for m in system_metrics) / len(system_metrics)
            else:
                avg_cpu = avg_memory = avg_disk = 0.0
            
            report = {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": duration_hours
                },
                "request_metrics": {
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "error_rate": error_rate,
                    "response_time_stats": response_time_stats
                },
                "system_metrics": {
                    "average_cpu_percent": avg_cpu,
                    "average_memory_percent": avg_memory,
                    "average_disk_usage_percent": avg_disk
                },
                "alerts": [
                    alert for alert in self.alerts
                    if start_time <= datetime.fromisoformat(alert["timestamp"]) <= end_time
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            } 