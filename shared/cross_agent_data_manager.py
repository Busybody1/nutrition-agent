"""
Cross-Agent Data Manager for Multi-User AI Agent Framework

This module provides data sharing and synchronization capabilities between
different agents, enabling seamless user experience across the framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .database import get_async_session
from .models import UserAgentPreferencesORM
from .inter_agent_communication import InterAgentCommunicationManager
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)

class CrossAgentDataManager:
    """Manages data sharing and synchronization between agents."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.communication_manager = InterAgentCommunicationManager(agent_type)
        self.sync_interval = 300  # 5 minutes
        self.sync_task = None
        
    async def initialize(self):
        """Initialize the cross-agent data manager."""
        try:
            await self.communication_manager.initialize()
            
            # Start background sync task
            self.sync_task = asyncio.create_task(self._background_sync_task())
            
            logger.info(f"CrossAgentDataManager initialized for {self.agent_type}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-agent data manager: {e}")
    
    async def share_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        target_agents: List[str] = None
    ) -> bool:
        """Share user preferences with other agents."""
        try:
            # Store preferences locally first
            await self._store_local_preferences(user_id, preferences)
            
            # If no specific agents specified, share with all known agents
            if not target_agents:
                target_agents = ["nutrition", "workout", "activity", "vision"]
            
            # Send preference sync requests to target agents
            sync_requests = []
            for target_agent in target_agents:
                if target_agent != self.agent_type:  # Don't send to self
                    request_data = {
                        "user_id": user_id,
                        "preferences": preferences,
                        "source_agent": self.agent_type,
                        "sync_type": "user_preferences"
                    }
                    
                    # Send async request
                    sync_requests.append(
                        self.communication_manager.send_request_to_agent(
                            target_agent=target_agent,
                            request_type="preference_sync",
                            request_data=request_data,
                            user_id=user_id,
                            priority=2
                        )
                    )
            
            # Wait for all requests to be sent
            if sync_requests:
                await asyncio.gather(*sync_requests, return_exceptions=True)
            
            logger.info(f"Shared preferences for user {user_id} with {len(target_agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to share preferences for user {user_id}: {e}")
            return False
    
    async def get_user_preferences_from_agents(
        self,
        user_id: str,
        source_agents: List[str] = None
    ) -> Dict[str, Any]:
        """Get user preferences from other agents."""
        try:
            # If no specific agents specified, get from all known agents
            if not source_agents:
                source_agents = ["nutrition", "workout", "activity", "vision"]
            
            # Get local preferences first
            local_preferences = await self._get_local_preferences(user_id)
            
            # Request preferences from other agents
            agent_preferences = {}
            for source_agent in source_agents:
                if source_agent != self.agent_type:  # Don't request from self
                    try:
                        response = await self.communication_manager.send_request_to_agent(
                            target_agent=source_agent,
                            request_type="user_data_request",
                            request_data={
                                "user_id": user_id,
                                "data_type": "user_preferences",
                                "source_agent": self.agent_type
                            },
                            user_id=user_id,
                            priority=1
                        )
                        
                        if response and response.get("status") == "completed":
                            agent_preferences[source_agent] = response.get("data", {})
                        else:
                            logger.warning(f"No response from {source_agent} for user {user_id}")
                            
                    except Exception as e:
                        logger.error(f"Failed to get preferences from {source_agent}: {e}")
                        agent_preferences[source_agent] = {}
            
            # Combine all preferences
            all_preferences = {
                "local": local_preferences,
                **agent_preferences
            }
            
            logger.info(f"Retrieved preferences for user {user_id} from {len(agent_preferences)} agents")
            return all_preferences
            
        except Exception as e:
            logger.error(f"Failed to get preferences for user {user_id}: {e}")
            return {"local": {}, "error": str(e)}
    
    async def sync_user_data(
        self,
        user_id: str,
        data_type: str,
        data: Dict[str, Any],
        target_agents: List[str] = None
    ) -> bool:
        """Sync user data with other agents."""
        try:
            # If no specific agents specified, sync with all known agents
            if not target_agents:
                target_agents = ["nutrition", "workout", "activity", "vision"]
            
            # Send sync requests to target agents
            sync_requests = []
            for target_agent in target_agents:
                if target_agent != self.agent_type:  # Don't sync with self
                    request_data = {
                        "user_id": user_id,
                        "data_type": data_type,
                        "data": data,
                        "source_agent": self.agent_type,
                        "sync_type": "data_sync"
                    }
                    
                    # Send async request
                    sync_requests.append(
                        self.communication_manager.send_request_to_agent(
                            target_agent=target_agent,
                            request_type="data_sync",
                            request_data=request_data,
                            user_id=user_id,
                            priority=2
                        )
                    )
            
            # Wait for all requests to be sent
            if sync_requests:
                await asyncio.gather(*sync_requests, return_exceptions=True)
            
            logger.info(f"Synced {data_type} data for user {user_id} with {len(target_agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync {data_type} data for user {user_id}: {e}")
            return False
    
    async def get_user_data_from_agents(
        self,
        user_id: str,
        data_type: str,
        source_agents: List[str] = None
    ) -> Dict[str, Any]:
        """Get user data from other agents."""
        try:
            # If no specific agents specified, get from all known agents
            if not source_agents:
                source_agents = ["nutrition", "workout", "activity", "vision"]
            
            # Request data from other agents
            agent_data = {}
            for source_agent in source_agents:
                if source_agent != self.agent_type:  # Don't request from self
                    try:
                        response = await self.communication_manager.send_request_to_agent(
                            target_agent=source_agent,
                            request_type="user_data_request",
                            request_data={
                                "user_id": user_id,
                                "data_type": data_type,
                                "source_agent": self.agent_type
                            },
                            user_id=user_id,
                            priority=1
                        )
                        
                        if response and response.get("status") == "completed":
                            agent_data[source_agent] = response.get("data", {})
                        else:
                            logger.warning(f"No response from {source_agent} for user {user_id}")
                            
                    except Exception as e:
                        logger.error(f"Failed to get {data_type} from {source_agent}: {e}")
                        agent_data[source_agent] = {}
            
            logger.info(f"Retrieved {data_type} data for user {user_id} from {len(agent_data)} agents")
            return agent_data
            
        except Exception as e:
            logger.error(f"Failed to get {data_type} data for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def _store_local_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Store user preferences locally."""
        try:
            async with get_async_session() as session:
                # Check if preferences already exist
                result = await session.execute(
                    select(UserAgentPreferencesORM)
                    .where(UserAgentPreferencesORM.user_id == user_id)
                    .where(UserAgentPreferencesORM.agent_type == self.agent_type)
                )
                
                existing_prefs = result.scalar_one_or_none()
                
                if existing_prefs:
                    # Update existing preferences
                    existing_prefs.preferences = preferences
                    existing_prefs.last_used = datetime.utcnow()
                    existing_prefs.updated_at = datetime.utcnow()
                    existing_prefs.usage_count += 1
                else:
                    # Create new preferences
                    new_prefs = UserAgentPreferencesORM(
                        user_id=user_id,
                        agent_type=self.agent_type,
                        preferences=preferences,
                        last_used=datetime.utcnow(),
                        usage_count=1,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(new_prefs)
                
                await session.commit()
                logger.debug(f"Stored local preferences for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store local preferences for user {user_id}: {e}")
            return False
    
    async def _get_local_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get local user preferences."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(UserAgentPreferencesORM)
                    .where(UserAgentPreferencesORM.user_id == user_id)
                    .where(UserAgentPreferencesORM.agent_type == self.agent_type)
                )
                
                prefs = result.scalar_one_or_none()
                if prefs:
                    return prefs.preferences
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get local preferences for user {user_id}: {e}")
            return {}
    
    async def _background_sync_task(self):
        """Background task for periodic data synchronization."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._perform_background_sync()
            except asyncio.CancelledError:
                logger.info("Background sync task cancelled")
                break
            except Exception as e:
                logger.error(f"Background sync task error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_background_sync(self):
        """Perform background synchronization tasks."""
        try:
            # Get all users with recent activity
            recent_users = await self._get_recent_active_users()
            
            for user_id in recent_users:
                try:
                    # Sync user preferences
                    local_prefs = await self._get_local_preferences(user_id)
                    if local_prefs:
                        await self.share_user_preferences(user_id, local_prefs)
                    
                    # Sync other data types as needed
                    # This can be extended based on specific requirements
                    
                except Exception as e:
                    logger.error(f"Failed to sync data for user {user_id}: {e}")
                    continue
            
            logger.info(f"Completed background sync for {len(recent_users)} users")
            
        except Exception as e:
            logger.error(f"Background sync failed: {e}")
    
    async def _get_recent_active_users(self) -> List[str]:
        """Get list of users with recent activity."""
        try:
            # This is a simplified implementation
            # In production, you might want to query actual user activity
            return []
        except Exception as e:
            logger.error(f"Failed to get recent active users: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.sync_task:
                self.sync_task.cancel()
                try:
                    await self.sync_task
                except asyncio.CancelledError:
                    pass
            
            await self.communication_manager.cleanup()
            logger.info(f"CrossAgentDataManager cleaned up for {self.agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup cross-agent data manager: {e}") 