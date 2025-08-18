"""
Inter-Agent Communication Manager for Multi-User AI Agent Framework

This module provides communication infrastructure between different agents,
enabling cross-agent data sharing, request routing, and coordination.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from sqlalchemy import select, update, delete, and_

from .database import get_async_session
from .models import InterAgentCommunicationORM

logger = logging.getLogger(__name__)

class InterAgentCommunicationManager:
    """Manages communication between different AI agents."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.communication_queue = asyncio.Queue()
        self.active_communications = {}
        
    async def initialize(self):
        """Initialize the communication manager."""
        try:
            # Start background task for processing communications
            self._queue_task = asyncio.create_task(self._process_communication_queue())
            logger.info(f"InterAgentCommunicationManager initialized for {self.agent_type}")
        except Exception as e:
            logger.error(f"Failed to initialize communication manager: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, '_queue_task') and self._queue_task:
                self._queue_task.cancel()
                try:
                    await self._queue_task
                except asyncio.CancelledError:
                    pass
            logger.info(f"InterAgentCommunicationManager cleaned up for {self.agent_type}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def send_request_to_agent(
        self,
        target_agent: str,
        request_type: str,
        request_data: Dict[str, Any],
        user_id: str = None,
        priority: int = 1,
        timeout: int = 30
    ) -> str:
        """Send a request to another agent."""
        try:
            communication_id = str(uuid4())
            
            # Store in database
            async with get_async_session() as session:
                communication_orm = InterAgentCommunicationORM(
                    id=communication_id,
                    source_agent=self.agent_type,
                    target_agent=target_agent,
                    request_type=request_type,
                    request_data=request_data,
                    user_id=user_id,
                    priority=priority,
                    status="pending",
                    created_at=datetime.utcnow(),
                    timeout_at=datetime.utcnow() + timedelta(seconds=timeout)
                )
                
                session.add(communication_orm)
                await session.commit()
            
            # Add to processing queue
            await self.communication_queue.put({
                "communication_id": communication_id,
                "priority": priority,
                "data": {
                    "id": communication_id,
                    "source_agent": self.agent_type,
                    "target_agent": target_agent,
                    "request_type": request_type,
                    "request_data": request_data,
                    "user_id": user_id,
                    "priority": priority,
                    "status": "pending",
                    "created_at": datetime.utcnow(),
                    "timeout_at": datetime.utcnow() + timedelta(seconds=timeout)
                }
            })
            
            # Store in active communications
            self.active_communications[communication_id] = {
                "status": "pending",
                "created_at": datetime.utcnow(),
                "timeout_at": datetime.utcnow() + timedelta(seconds=timeout)
            }
            
            logger.info(f"Sent {request_type} request to {target_agent} agent")
            return communication_id
            
        except Exception as e:
            logger.error(f"Failed to send request to {target_agent}: {e}")
            raise
    
    async def get_agent_requests(
        self,
        request_type: str = None,
        status: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get requests for this agent."""
        try:
            async with get_async_session() as session:
                query = select(InterAgentCommunicationORM).where(
                    InterAgentCommunicationORM.target_agent == self.agent_type
                )
                
                if request_type:
                    query = query.where(InterAgentCommunicationORM.request_type == request_type)
                
                if status:
                    query = query.where(InterAgentCommunicationORM.status == status)
                
                query = query.order_by(InterAgentCommunicationORM.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                communications = result.scalars().all()
                
                return [self._communication_to_dict(comm) for comm in communications]
                
        except Exception as e:
            logger.error(f"Failed to get agent requests: {e}")
            return []
    
    async def respond_to_request(
        self,
        communication_id: str,
        response_data: Dict[str, Any],
        status: str = "completed"
    ) -> bool:
        """Respond to a request from another agent."""
        try:
            async with get_async_session() as session:
                # Update communication record
                await session.execute(
                    update(InterAgentCommunicationORM)
                    .where(InterAgentCommunicationORM.id == communication_id)
                    .where(InterAgentCommunicationORM.target_agent == self.agent_type)
                    .values(
                        response_data=response_data,
                        status=status,
                        completed_at=datetime.utcnow(),
                        processing_time_ms=int((datetime.utcnow() - datetime.utcnow()).total_seconds() * 1000)
                    )
                )
                await session.commit()
                
                # Update local tracking
                if communication_id in self.active_communications:
                    self.active_communications[communication_id]["status"] = status
                    self.active_communications[communication_id]["completed_at"] = datetime.utcnow()
                
                logger.info(f"Responded to request {communication_id} with status {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to respond to request {communication_id}: {e}")
            return False
    
    async def get_communication_status(
        self,
        communication_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a communication."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(InterAgentCommunicationORM)
                    .where(InterAgentCommunicationORM.id == communication_id)
                )
                
                communication = result.scalar_one_or_none()
                if communication:
                    return self._communication_to_dict(communication)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get communication status for {communication_id}: {e}")
            return None
    
    async def cleanup_expired_communications(self) -> int:
        """Clean up expired communications and return count of cleaned communications."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    delete(InterAgentCommunicationORM)
                    .where(InterAgentCommunicationORM.timeout_at < datetime.utcnow())
                    .where(InterAgentCommunicationORM.status.in_(["pending", "processing"]))
                )
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired communications")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired communications: {e}")
            return 0
    
    async def _process_communication_queue(self):
        """Process communications from the queue."""
        while True:
            try:
                # Get communication from queue
                communication_item = await self.communication_queue.get()
                communication_id = communication_item["communication_id"]
                priority = communication_item["priority"]
                data = communication_item["data"]
                
                # Process communication
                await self._process_communication(data)
                
                # Mark as done
                self.communication_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Communication queue processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing communication: {e}")
                continue
    
    async def _process_communication(self, communication_data: Dict[str, Any]):
        """Process a single communication."""
        try:
            communication_id = communication_data["id"]
            request_type = communication_data["request_type"]
            request_data = communication_data["request_data"]
            
            # Update status to processing
            await self._update_communication_status(communication_id, "processing")
            
            # Process based on request type
            if request_type == "user_data_request":
                response = await self._handle_user_data_request(request_data)
            elif request_type == "preference_sync":
                response = await self._handle_preference_sync(request_data)
            elif request_type == "health_check":
                response = await self._handle_health_check(request_data)
            else:
                response = {"error": f"Unknown request type: {request_type}"}
            
            # Send response
            await self.respond_to_request(communication_id, response, "completed")
            
        except Exception as e:
            logger.error(f"Failed to process communication: {e}")
            # Mark as failed
            await self._update_communication_status(communication_id, "failed")
    
    async def _handle_user_data_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user data request from another agent."""
        try:
            user_id = request_data.get("user_id")
            data_type = request_data.get("data_type")
            
            if data_type == "user_preferences":
                # Return user preferences for this agent
                from .cross_agent_data_manager import CrossAgentDataManager
                data_manager = CrossAgentDataManager(self.agent_type)
                preferences = await data_manager._get_local_preferences(user_id)
                return {"preferences": preferences}
            else:
                return {"error": f"Unknown data type: {data_type}"}
                
        except Exception as e:
            logger.error(f"Failed to handle user data request: {e}")
            return {"error": str(e)}
    
    async def _handle_preference_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle preference sync from another agent."""
        try:
            user_id = request_data.get("user_id")
            preferences = request_data.get("preferences")
            
            # Store preferences locally
            from .cross_agent_data_manager import CrossAgentDataManager
            data_manager = CrossAgentDataManager(self.agent_type)
            success = await data_manager._store_local_preferences(user_id, preferences)
            
            return {"success": success, "message": "Preferences synced"}
            
        except Exception as e:
            logger.error(f"Failed to handle preference sync: {e}")
            return {"error": str(e)}
    
    async def _handle_health_check(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request from another agent."""
        try:
            from .health_monitor import HealthMonitor
            health_monitor = HealthMonitor(self.agent_type)
            health_status = await health_monitor.get_agent_health()
            
            return {
                "status": "healthy",
                "agent_type": self.agent_type,
                "health_score": health_status.get("health_score", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to handle health check: {e}")
            return {"error": str(e)}
    
    async def _update_communication_status(self, communication_id: str, status: str):
        """Update communication status in database."""
        try:
            async with get_async_session() as session:
                await session.execute(
                    update(InterAgentCommunicationORM)
                    .where(InterAgentCommunicationORM.id == communication_id)
                    .values(status=status)
                )
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update communication status: {e}")
    
    def _communication_to_dict(self, communication: InterAgentCommunicationORM) -> Dict[str, Any]:
        """Convert communication ORM object to dictionary."""
        return {
            "id": str(communication.id),
            "user_id": str(communication.user_id) if communication.user_id else None,
            "source_agent": communication.source_agent,
            "target_agent": communication.target_agent,
            "request_type": communication.request_type,
            "request_data": communication.request_data,
            "response_data": communication.response_data,
            "status": communication.status,
            "created_at": communication.created_at.isoformat(),
            "completed_at": communication.completed_at.isoformat() if communication.completed_at else None,
            "processing_time_ms": communication.processing_time_ms
        } 