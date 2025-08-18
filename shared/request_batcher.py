"""
Request Batching Manager for Multi-User AI Agent Framework

This module provides intelligent request batching capabilities including:
- Request grouping by type and similarity
- Batch size optimization
- Time-based batching
- Priority-aware batching
- Batch processing with aggregation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine
from dataclasses import dataclass, field
from uuid import uuid4
import heapq

logger = logging.getLogger(__name__)

@dataclass
class BatchedRequest:
    """Represents a batch of similar requests."""
    id: str
    request_type: str
    requests: List[Dict[str, Any]]
    created_at: datetime
    batch_size: int
    priority: int
    timeout_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class RequestBatchingManager:
    """Manages intelligent request batching for improved performance."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.batch_configs: Dict[str, Dict[str, Any]] = {}
        self.pending_batches: Dict[str, BatchedRequest] = {}
        self.batch_processors: Dict[str, Callable] = {}
        self.batch_timer: Optional[asyncio.Task] = None
        self.stats = {
            "total_batches": 0,
            "total_requests_batched": 0,
            "avg_batch_size": 0.0,
            "avg_processing_time": 0.0,
            "batches_processed": 0
        }
        
        # Default batch configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default batch configurations for common request types."""
        self.batch_configs = {
            "user_preferences": {
                "max_batch_size": 50,
                "max_wait_time": 5,  # seconds
                "priority": 2
            },
            "health_check": {
                "max_batch_size": 100,
                "max_wait_time": 2,
                "priority": 1
            },
            "data_sync": {
                "max_batch_size": 25,
                "max_wait_time": 10,
                "priority": 3
            },
            "analytics": {
                "max_batch_size": 200,
                "max_wait_time": 15,
                "priority": 1
            }
        }
    
    async def initialize(self):
        """Initialize the batching manager."""
        try:
            # Start batch timer for time-based processing
            self.batch_timer = asyncio.create_task(self._batch_timer_loop())
            logger.info(f"RequestBatchingManager initialized for {self.agent_type}")
        except Exception as e:
            logger.error(f"Failed to initialize RequestBatchingManager: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.batch_timer:
                self.batch_timer.cancel()
                try:
                    await self.batch_timer
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"RequestBatchingManager cleaned up for {self.agent_type}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def add_request_to_batch(
        self,
        request_type: str,
        request_data: Dict[str, Any],
        priority: int = 1,
        user_id: str = None
    ) -> str:
        """Add a request to a batch."""
        try:
            # Get or create batch for this request type
            batch_id = await self._get_or_create_batch(request_type, priority)
            
            # Add request to batch
            request_item = {
                "id": str(uuid4()),
                "data": request_data,
                "priority": priority,
                "user_id": user_id,
                "added_at": datetime.utcnow()
            }
            
            self.pending_batches[batch_id].requests.append(request_item)
            self.pending_batches[batch_id].batch_size = len(self.pending_batches[batch_id].requests)
            
            # Check if batch is ready for processing
            config = self.batch_configs.get(request_type, {})
            max_size = config.get("max_batch_size", 50)
            
            if len(self.pending_batches[batch_id].requests) >= max_size:
                await self.process_batch(batch_id)
            
            logger.debug(f"Added request to batch {batch_id} for {request_type}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to add request to batch: {e}")
            raise
    
    async def process_batch(self, batch_id: str) -> Optional[Any]:
        """Process a specific batch."""
        try:
            if batch_id not in self.pending_batches:
                logger.warning(f"Batch {batch_id} not found")
                return None
            
            batch = self.pending_batches[batch_id]
            request_type = batch.request_type
            
            # Check if processor exists
            if request_type not in self.batch_processors:
                logger.warning(f"No processor registered for request type: {request_type}")
                return None
            
            # Process batch
            start_time = datetime.utcnow()
            processor = self.batch_processors[request_type]
            
            if asyncio.iscoroutinefunction(processor):
                result = await processor(batch.requests)
            else:
                result = processor(batch.requests)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update batch status
            batch.status = "completed"
            batch.result = result
            batch.metadata["processing_time"] = processing_time
            batch.metadata["completed_at"] = datetime.utcnow()
            
            # Update statistics
            self._update_stats(batch, processing_time)
            
            # Remove from pending batches
            del self.pending_batches[batch_id]
            
            logger.info(f"Processed batch {batch_id} with {batch.batch_size} requests in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_id}: {e}")
            if batch_id in self.pending_batches:
                self.pending_batches[batch_id].status = "failed"
                self.pending_batches[batch_id].metadata["error"] = str(e)
            return None
    
    def register_batch_processor(
        self,
        request_type: str,
        processor: Callable[[List[Dict[str, Any]]], Any]
    ):
        """Register a processor for a specific request type."""
        try:
            self.batch_processors[request_type] = processor
            logger.info(f"Registered processor for request type: {request_type}")
        except Exception as e:
            logger.error(f"Failed to register processor for {request_type}: {e}")
    
    async def _batch_timer_loop(self):
        """Background loop for time-based batch processing."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                await self._process_pending_batches()
            except asyncio.CancelledError:
                logger.info("Batch timer loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch timer loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_pending_batches(self):
        """Process batches that have exceeded their wait time."""
        try:
            current_time = datetime.utcnow()
            batches_to_process = []
            
            for batch_id, batch in self.pending_batches.items():
                if batch.status != "pending":
                    continue
                
                config = self.batch_configs.get(batch.request_type, {})
                max_wait_time = config.get("max_wait_time", 10)
                
                if (current_time - batch.created_at).total_seconds() >= max_wait_time:
                    batches_to_process.append(batch_id)
            
            # Process batches in priority order
            for batch_id in sorted(
                batches_to_process,
                key=lambda x: self.pending_batches[x].priority
            ):
                await self.process_batch(batch_id)
                
        except Exception as e:
            logger.error(f"Error processing pending batches: {e}")
    
    async def _get_or_create_batch(self, request_type: str, priority: int) -> str:
        """Get existing batch or create new one for request type."""
        try:
            # Look for existing batch with same type and priority
            for batch_id, batch in self.pending_batches.items():
                if (batch.request_type == request_type and 
                    batch.priority == priority and 
                    batch.status == "pending"):
                    return batch_id
            
            # Create new batch
            batch_id = str(uuid4())
            config = self.batch_configs.get(request_type, {})
            max_wait_time = config.get("max_wait_time", 10)
            
            new_batch = BatchedRequest(
                id=batch_id,
                request_type=request_type,
                requests=[],
                created_at=datetime.utcnow(),
                batch_size=0,
                priority=priority,
                timeout_at=datetime.utcnow() + timedelta(seconds=max_wait_time),
                status="pending"
            )
            
            self.pending_batches[batch_id] = new_batch
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to get or create batch: {e}")
            raise
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific batch."""
        try:
            if batch_id not in self.pending_batches:
                return None
            
            batch = self.pending_batches[batch_id]
            return {
                "id": batch.id,
                "request_type": batch.request_type,
                "batch_size": batch.batch_size,
                "priority": batch.priority,
                "status": batch.status,
                "created_at": batch.created_at.isoformat(),
                "timeout_at": batch.timeout_at.isoformat() if batch.timeout_at else None,
                "result": batch.result,
                "metadata": batch.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            return None
    
    def get_batching_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        try:
            pending_count = len([b for b in self.pending_batches.values() if b.status == "pending"])
            
            return {
                **self.stats,
                "pending_batches": pending_count,
                "registered_processors": list(self.batch_processors.keys()),
                "batch_configs": list(self.batch_configs.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get batching stats: {e}")
            return {}
    
    def _update_stats(self, batch: BatchedRequest, processing_time: float):
        """Update batching statistics."""
        try:
            self.stats["total_batches"] += 1
            self.stats["total_requests_batched"] += batch.batch_size
            self.stats["batches_processed"] += 1
            
            # Update average batch size
            current_avg = self.stats["avg_batch_size"]
            new_avg = (current_avg * (self.stats["batches_processed"] - 1) + batch.batch_size) / self.stats["batches_processed"]
            self.stats["avg_batch_size"] = new_avg
            
            # Update average processing time
            current_avg_time = self.stats["avg_processing_time"]
            new_avg_time = (current_avg_time * (self.stats["batches_processed"] - 1) + processing_time) / self.stats["batches_processed"]
            self.stats["avg_processing_time"] = new_avg_time
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
    
    async def get_all_batches(self) -> List[Dict[str, Any]]:
        """Get all pending and completed batches."""
        try:
            all_batches = []
            
            # Get pending batches
            for batch_id, batch in self.pending_batches.items():
                all_batches.append(self.get_batch_status(batch_id))
            
            return [b for b in all_batches if b is not None]
            
        except Exception as e:
            logger.error(f"Failed to get all batches: {e}")
            return []
    
    async def clear_completed_batches(self) -> int:
        """Clear completed batches and return count of cleared batches."""
        try:
            cleared_count = 0
            
            for batch_id, batch in list(self.pending_batches.items()):
                if batch.status in ["completed", "failed"]:
                    del self.pending_batches[batch_id]
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} completed batches")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear completed batches: {e}")
            return 0 