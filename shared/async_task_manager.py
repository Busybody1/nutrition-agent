"""
Advanced Async Task Manager for Nutrition Agent

This module provides sophisticated async task processing capabilities including:
- Priority queues for different request types
- Request batching and processing
- Background task monitoring
- Task timeout handling
- Task progress tracking
- High-load scenario optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import heapq

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class AsyncTask:
    """Represents an async task with metadata."""
    id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    timeout_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

class PriorityQueue:
    """Priority queue implementation for tasks."""
    
    def __init__(self):
        self._queue = []
        self._counter = 0  # For stable sorting when priorities are equal
    
    def put(self, task: AsyncTask):
        """Add task to priority queue."""
        # Use negative priority for max-heap behavior (highest priority first)
        # Add counter for stable sorting when priorities are equal
        heapq.heappush(self._queue, (-task.priority.value, self._counter, task))
        self._counter += 1
    
    def get(self) -> Optional[AsyncTask]:
        """Get highest priority task from queue."""
        if not self._queue:
            return None
        return heapq.heappop(self._queue)[2]
    
    def qsize(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

class AsyncTaskManager:
    """Advanced async task manager with priority queues and monitoring."""
    
    def __init__(self, agent_type: str = "nutrition", max_workers: int = 10):
        self.agent_type = agent_type
        self.max_workers = max_workers
        self.priority_queue = PriorityQueue()
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: Dict[str, AsyncTask] = {}
        self.failed_tasks: Dict[str, AsyncTask] = {}
        self.workers: List[asyncio.Task] = []
        self.task_handlers: Dict[str, Callable] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "timeout_tasks": 0,
            "avg_processing_time": 0.0,
            "queue_size": 0
        }
        
    async def initialize(self):
        """Initialize the task manager."""
        try:
            # Start worker tasks
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self.workers.append(worker)
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"AsyncTaskManager initialized for {self.agent_type} with {self.max_workers} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize AsyncTaskManager: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
            
            # Cancel monitoring and cleanup tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"AsyncTaskManager cleaned up for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered task handler for {task_type}")
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task for processing."""
        try:
            task_id = str(uuid4())
            timeout_at = None
            if timeout:
                timeout_at = datetime.utcnow() + timedelta(seconds=timeout)
            
            task = AsyncTask(
                id=task_id,
                task_type=task_type,
                priority=priority,
                payload=payload,
                created_at=datetime.utcnow(),
                timeout_at=timeout_at,
                metadata=metadata or {}
            )
            
            # Add to priority queue
            self.priority_queue.put(task)
            self.stats["total_tasks"] += 1
            self.stats["queue_size"] = self.priority_queue.qsize()
            
            logger.info(f"Submitted task {task_id} of type {task_type} with priority {priority.name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            return self._task_to_dict(task)
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task."""
        try:
            # Check if task is in queue (would need to implement queue removal)
            # For now, just mark active tasks as cancelled
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                logger.info(f"Cancelled task {task_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and system status."""
        return {
            "queue_size": self.priority_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_workers": len(self.workers),
            "stats": self.stats.copy(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _worker(self, worker_name: str):
        """Worker task that processes tasks from the priority queue."""
        logger.info(f"Worker {worker_name} started")
        
        while True:
            try:
                # Get next task from priority queue
                task = self.priority_queue.get()
                if not task:
                    await asyncio.sleep(0.1)  # Wait for tasks
                    continue
                
                # Check if task has timed out
                if task.timeout_at and datetime.utcnow() > task.timeout_at:
                    task.status = TaskStatus.TIMEOUT
                    task.completed_at = datetime.utcnow()
                    self.failed_tasks[task.id] = task
                    self.stats["timeout_tasks"] += 1
                    logger.warning(f"Task {task.id} timed out")
                    continue
                
                # Process task
                await self._process_task(task, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_task(self, task: AsyncTask, worker_name: str):
        """Process a single task."""
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()
            self.active_tasks[task.id] = task
            
            logger.info(f"Worker {worker_name} processing task {task.id}")
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")
            
            # Execute task with progress tracking
            result = await self._execute_with_progress(task, handler)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 100.0
            task.completed_at = datetime.utcnow()
            
            # Move to completed tasks
            self.completed_tasks[task.id] = task
            del self.active_tasks[task.id]
            
            # Update stats
            self.stats["completed_tasks"] += 1
            if task.started_at:
                processing_time = (task.completed_at - task.started_at).total_seconds()
                self._update_avg_processing_time(processing_time)
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            await self._handle_task_failure(task, e)
    
    async def _execute_with_progress(self, task: AsyncTask, handler: Callable) -> Any:
        """Execute task with progress tracking."""
        try:
            # For now, simple execution - in future could add progress callbacks
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload, task)
            else:
                result = handler(task.payload, task)
            
            # Update progress
            task.progress = 100.0
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _handle_task_failure(self, task: AsyncTask, error: Exception):
        """Handle task failure with retry logic."""
        try:
            task.error = str(error)
            task.completed_at = datetime.utcnow()
            
            if task.retry_count < task.max_retries:
                # Retry task
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = None
                task.completed_at = None
                task.started_at = None
                task.progress = 0.0
                
                # Add back to queue with lower priority
                if task.priority.value > TaskPriority.LOW.value:
                    task.priority = TaskPriority(task.priority.value - 1)
                
                self.priority_queue.put(task)
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
                
            else:
                # Mark as failed
                task.status = TaskStatus.FAILED
                self.failed_tasks[task.id] = task
                self.stats["failed_tasks"] += 1
                
                # Remove from active tasks
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                logger.error(f"Task {task.id} failed after {task.max_retries} retries")
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for system health."""
        logger.info("Task monitoring loop started")
        
        while True:
            try:
                # Update queue size in stats
                self.stats["queue_size"] = self.priority_queue.qsize()
                
                # Log system status periodically
                if self.stats["total_tasks"] % 100 == 0 and self.stats["total_tasks"] > 0:
                    logger.info(f"Task Manager Status: {self.stats}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                logger.info("Task monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _cleanup_loop(self):
        """Background cleanup loop for old completed tasks."""
        logger.info("Task cleanup loop started")
        
        while True:
            try:
                # Clean up old completed tasks (keep last 1000)
                if len(self.completed_tasks) > 1000:
                    # Remove oldest tasks
                    sorted_tasks = sorted(
                        self.completed_tasks.values(),
                        key=lambda t: t.completed_at or t.created_at
                    )
                    
                    tasks_to_remove = sorted_tasks[:-1000]
                    for task in tasks_to_remove:
                        del self.completed_tasks[task.id]
                    
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
                
                # Clean up old failed tasks (keep last 500)
                if len(self.failed_tasks) > 500:
                    sorted_failed = sorted(
                        self.failed_tasks.values(),
                        key=lambda t: t.completed_at or t.created_at
                    )
                    
                    failed_to_remove = sorted_failed[:-500]
                    for task in failed_to_remove:
                        del self.failed_tasks[task.id]
                    
                    logger.info(f"Cleaned up {len(failed_to_remove)} old failed tasks")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except asyncio.CancelledError:
                logger.info("Task cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    def _task_to_dict(self, task: AsyncTask) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "id": task.id,
            "task_type": task.task_type,
            "priority": task.priority.name,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "timeout_at": task.timeout_at.isoformat() if task.timeout_at else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "error": task.error,
            "metadata": task.metadata
        }
    
    def _update_avg_processing_time(self, new_time: float):
        """Update average processing time statistics."""
        current_avg = self.stats["avg_processing_time"]
        completed_count = self.stats["completed_tasks"]
        
        if completed_count == 1:
            self.stats["avg_processing_time"] = new_time
        else:
            # Exponential moving average
            alpha = 0.1  # Smoothing factor
            self.stats["avg_processing_time"] = alpha * new_time + (1 - alpha) * current_avg 