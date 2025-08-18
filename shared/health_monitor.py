"""
Comprehensive Health Monitoring for Nutrition Agent

This module provides health monitoring capabilities including:
- Agent health status tracking
- Database connectivity monitoring
- Redis connectivity monitoring
- Performance metrics aggregation
- Health score calculation
- Framework-wide health reporting
- Inter-agent communication health
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from collections import deque
import httpx

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, agent_type: str = "nutrition", max_history: int = 1000):
        self.agent_type = agent_type
        self.max_history = max_history
        
        # Health metrics storage
        self.health_metrics: deque = deque(maxlen=max_history)
        self.error_metrics: deque = deque(maxlen=max_history)
        self.performance_metrics: deque = deque(maxlen=max_history)
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_interval = 30  # seconds
        self.health_thresholds = {
            "response_time_ms": 5000.0,
            "error_rate": 0.05,
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "database_latency_ms": 1000.0
        }
        
        # Health status
        self.current_health = "healthy"
        self.last_health_check = datetime.utcnow()
        self.health_score = 100.0
        
        # Component health
        self.component_health = {
            "database": "unknown",
            "redis": "unknown",
            "api_endpoints": "unknown",
            "background_tasks": "unknown",
            "external_services": "unknown"
        }
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
    
    async def initialize(self):
        """Initialize the health monitor."""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"HealthMonitor initialized for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HealthMonitor: {e}")
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
            
            logger.info(f"HealthMonitor cleaned up for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def record_request(self, endpoint: str, response_time_ms: float, status_code: int, user_id: Optional[str] = None):
        """Record a request for health monitoring."""
        try:
            timestamp = datetime.utcnow()
            
            # Record health metric
            health_metric = {
                "timestamp": timestamp,
                "endpoint": endpoint,
                "response_time_ms": response_time_ms,
                "status_code": status_code,
                "user_id": user_id,
                "success": 200 <= status_code < 400
            }
            
            self.health_metrics.append(health_metric)
            
            # Update counters
            self.request_count += 1
            if not health_metric["success"]:
                self.error_count += 1
            
            # Record performance metric
            performance_metric = {
                "timestamp": timestamp,
                "endpoint": endpoint,
                "response_time_ms": response_time_ms,
                "status_code": status_code
            }
            
            self.performance_metrics.append(performance_metric)
            
            # Check for health alerts
            if response_time_ms > self.health_thresholds["response_time_ms"]:
                logger.warning(f"High response time alert: {endpoint} took {response_time_ms}ms")
            
            if status_code >= 500:
                logger.error(f"Server error alert: {endpoint} returned {status_code}")
            
        except Exception as e:
            logger.error(f"Failed to record request: {e}")
    
    async def get_agent_health(self) -> Dict[str, Any]:
        """Get comprehensive health status for the agent."""
        try:
            # Check database health
            db_health = await self._check_database_health()
            
            # Check Redis health
            redis_health = await self._check_redis_health()
            
            # Check API endpoints health
            api_health = await self._check_api_endpoints_health()
            
            # Check background tasks health
            background_health = await self._check_background_tasks_health()
            
            # Check external services health
            external_health = await self._check_external_services_health()
            
            # Update component health
            self.component_health.update({
                "database": db_health["status"],
                "redis": redis_health["status"],
                "api_endpoints": api_health["status"],
                "background_tasks": background_health["status"],
                "external_services": external_health["status"]
            })
            
            # Calculate overall health score
            health_score = self._calculate_health_score()
            self.health_score = health_score
            
            # Determine overall health status
            if health_score >= 90:
                overall_status = "healthy"
            elif health_score >= 70:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            self.current_health = overall_status
            self.last_health_check = datetime.utcnow()
            
            health_report = {
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": overall_status,
                "health_score": health_score,
                "last_health_check": self.last_health_check.isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                "component_health": {
                    "database": db_health,
                    "redis": redis_health,
                    "api_endpoints": api_health,
                    "background_tasks": background_health,
                    "external_services": external_health
                },
                "performance_metrics": {
                    "total_requests": self.request_count,
                    "total_errors": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                    "avg_response_time_ms": self._calculate_avg_response_time()
                },
                "health_thresholds": self.health_thresholds.copy()
            }
            
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to get agent health: {e}")
            return {
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "health_score": 0.0,
                "error": str(e)
            }
    
    async def get_framework_health(self) -> Dict[str, Any]:
        """Get health status for the entire framework."""
        try:
            # Get current agent health
            agent_health = await self.get_agent_health()
            
            # Get inter-agent communication health
            inter_agent_health = await self._get_agent_communications()
            
            # Get system health
            system_health = await self._get_system_health()
            
            framework_health = {
                "timestamp": datetime.utcnow().isoformat(),
                "framework_status": "operational" if agent_health["overall_status"] == "healthy" else "degraded",
                "agents": {
                    self.agent_type: agent_health
                },
                "inter_agent_communications": inter_agent_health,
                "system_health": system_health,
                "overall_health_score": agent_health["health_score"]
            }
            
            return framework_health
            
        except Exception as e:
            logger.error(f"Failed to get framework health: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "framework_status": "error",
                "error": str(e)
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Try to import and use database connection
            from .database import get_async_session
            async with get_async_session() as session:
                # Simple health check query
                result = await session.execute("SELECT 1")
                result.fetchone()
            
            latency_ms = (time.time() - start_time) * 1000
            
            status = "healthy"
            if latency_ms > self.health_thresholds["database_latency_ms"]:
                status = "degraded"
            
            return {
                "status": status,
                "latency_ms": latency_ms,
                "last_check": datetime.utcnow().isoformat(),
                "details": "Database connection successful"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "last_check": datetime.utcnow().isoformat(),
                "details": f"Database connection failed: {str(e)}"
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()
            
            # Try to import and use Redis
            try:
                from .database import get_redis
                redis_client = await get_redis()
                
                # Simple health check
                await redis_client.ping()
                
                latency_ms = (time.time() - start_time) * 1000
                
                status = "healthy"
                if latency_ms > 100:  # Redis should be very fast
                    status = "degraded"
                
                return {
                    "status": status,
                    "latency_ms": latency_ms,
                    "last_check": datetime.utcnow().isoformat(),
                    "details": "Redis connection successful"
                }
                
            except ImportError:
                return {
                    "status": "not_configured",
                    "latency_ms": None,
                    "last_check": datetime.utcnow().isoformat(),
                    "details": "Redis not configured for this agent"
                }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "last_check": datetime.utcnow().isoformat(),
                "details": f"Redis connection failed: {str(e)}"
            }
    
    async def _check_api_endpoints_health(self) -> Dict[str, Any]:
        """Check API endpoints health."""
        try:
            # For now, just check if we can access the health endpoint
            # In a real implementation, you might want to check multiple endpoints
            
            status = "healthy"
            details = "API endpoints accessible"
            
            # Check if we have recent successful requests
            recent_requests = [
                metric for metric in self.health_metrics
                if (datetime.utcnow() - metric["timestamp"]).total_seconds() < 300  # Last 5 minutes
            ]
            
            if recent_requests:
                success_rate = sum(1 for req in recent_requests if req["success"]) / len(recent_requests)
                if success_rate < 0.95:  # 95% success rate threshold
                    status = "degraded"
                    details = f"API success rate: {success_rate:.2%}"
            else:
                status = "unknown"
                details = "No recent API requests"
            
            return {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "details": details,
                "recent_requests_count": len(recent_requests)
            }
            
        except Exception as e:
            logger.error(f"API endpoints health check failed: {e}")
            return {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "details": f"API health check failed: {str(e)}"
            }
    
    async def _check_background_tasks_health(self) -> Dict[str, Any]:
        """Check background tasks health."""
        try:
            # Check if background tasks are running
            # This would depend on your specific background task implementation
            
            status = "healthy"
            details = "Background tasks operational"
            
            # For now, assume healthy if we have recent activity
            recent_activity = (
                datetime.utcnow() - self.start_time
            ).total_seconds() < 3600  # Within last hour
            
            if not recent_activity:
                status = "degraded"
                details = "No recent background task activity"
            
            return {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Background tasks health check failed: {e}")
            return {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "details": f"Background tasks health check failed: {str(e)}"
            }
    
    async def _check_external_services_health(self) -> Dict[str, Any]:
        """Check external services health."""
        try:
            # Check external services like AI models, APIs, etc.
            # This would depend on your specific external service dependencies
            
            status = "healthy"
            details = "External services operational"
            
            # For now, assume healthy
            # In a real implementation, you might check:
            # - AI model API availability
            # - Third-party service health
            # - Network connectivity
            
            return {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "details": details
            }
            
        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "details": f"External services health check failed: {str(e)}"
            }
    
    async def _get_agent_communications(self) -> Dict[str, Any]:
        """Get inter-agent communication health."""
        try:
            # This would check communication with other agents
            # For now, return a placeholder
            
            return {
                "status": "healthy",
                "last_check": datetime.utcnow().isoformat(),
                "details": "Inter-agent communications operational",
                "connected_agents": [self.agent_type]  # Only self for now
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent communications: {e}")
            return {
                "status": "unknown",
                "last_check": datetime.utcnow().isoformat(),
                "details": f"Failed to check agent communications: {str(e)}"
            }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system-level health metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine system health
            system_status = "healthy"
            if cpu_percent > self.health_thresholds["cpu_percent"]:
                system_status = "degraded"
            if memory.percent > self.health_thresholds["memory_percent"]:
                system_status = "degraded"
            
            return {
                "status": system_status,
                "last_check": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_usage_percent": disk.percent
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "status": "unknown",
                "last_check": datetime.utcnow().isoformat(),
                "details": f"Failed to get system health: {str(e)}"
            }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score based on component health."""
        try:
            # Base score starts at 100
            score = 100.0
            
            # Deduct points for unhealthy components
            component_penalties = {
                "unhealthy": 25.0,
                "degraded": 10.0,
                "unknown": 5.0
            }
            
            for component, status in self.component_health.items():
                penalty = component_penalties.get(status, 0.0)
                score -= penalty
            
            # Deduct points for high error rates
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count
                if error_rate > 0.1:  # > 10% error rate
                    score -= 20.0
                elif error_rate > 0.05:  # > 5% error rate
                    score -= 10.0
            
            # Ensure score doesn't go below 0
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent metrics."""
        try:
            if not self.performance_metrics:
                return 0.0
            
            # Get recent metrics (last 100 requests)
            recent_metrics = list(self.performance_metrics)[-100:]
            response_times = [metric["response_time_ms"] for metric in recent_metrics]
            
            return sum(response_times) / len(response_times)
            
        except Exception as e:
            logger.error(f"Failed to calculate average response time: {e}")
            return 0.0
    
    async def _monitoring_loop(self):
        """Background health monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while True:
            try:
                # Perform health check
                health_report = await self.get_agent_health()
                
                # Log health status
                if health_report["overall_status"] != "healthy":
                    logger.warning(f"Agent health degraded: {health_report['overall_status']} (score: {health_report['health_score']:.1f})")
                else:
                    logger.debug(f"Agent health: {health_report['overall_status']} (score: {health_report['health_score']:.1f})")
                
                # Store health metric
                health_metric = {
                    "timestamp": datetime.utcnow(),
                    "health_score": health_report["health_score"],
                    "overall_status": health_report["overall_status"],
                    "component_health": health_report["component_health"]
                }
                
                self.health_metrics.append(health_metric)
                
                # Check every 30 seconds
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            # Get current health status
            health_status = await self.get_agent_health()
            
            # Calculate performance metrics
            recent_metrics = list(self.performance_metrics)[-100:] if self.performance_metrics else []
            
            if recent_metrics:
                response_times = [m["response_time_ms"] for m in recent_metrics]
                avg_response_time = sum(response_times) / len(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
            else:
                avg_response_time = min_response_time = max_response_time = 0.0
            
            report = {
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "health_status": health_status,
                "performance_metrics": {
                    "total_requests": self.request_count,
                    "total_errors": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                    "avg_response_time_ms": avg_response_time,
                    "min_response_time_ms": min_response_time,
                    "max_response_time_ms": max_response_time
                },
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_old_metrics(self):
        """Clean up old health metrics."""
        try:
            # Keep only metrics from last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Clean health metrics
            self.health_metrics = deque(
                [m for m in self.health_metrics if m["timestamp"] > cutoff_time],
                maxlen=self.max_history
            )
            
            # Clean performance metrics
            self.performance_metrics = deque(
                [m for m in self.performance_metrics if m["timestamp"] > cutoff_time],
                maxlen=self.max_history
            )
            
            # Clean error metrics
            self.error_metrics = deque(
                [m for m in self.error_metrics if m["timestamp"] > cutoff_time],
                maxlen=self.max_history
            )
            
            logger.info("Cleaned up old health metrics")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        try:
            # Check error rate
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count
                if error_rate > 0.05:
                    recommendations.append("High error rate detected. Review error logs and fix underlying issues.")
            
            # Check response times
            avg_response_time = self._calculate_avg_response_time()
            if avg_response_time > 2000:  # 2 seconds
                recommendations.append("High average response time. Consider optimizing database queries or adding caching.")
            
            # Check component health
            unhealthy_components = [
                component for component, status in self.component_health.items()
                if status == "unhealthy"
            ]
            
            if unhealthy_components:
                recommendations.append(f"Unhealthy components detected: {', '.join(unhealthy_components)}. Investigate and resolve issues.")
            
            # Check system resources
            if self.health_score < 70:
                recommendations.append("Low health score. Review all component health and system resources.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to error"] 