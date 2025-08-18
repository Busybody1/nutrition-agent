"""
Unified User Context Management for Multi-User AI Agent Framework

This module provides unified user context management across all agents,
coordinating sessions, conversations, and user preferences.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .session_manager import FrameworkSessionManager
from .conversation_manager import ConversationStateManager
from .cross_agent_data_manager import CrossAgentDataManager
from .database import get_async_session
from .models import UserAgentPreferencesORM
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)

class UnifiedUserContextManager:
    """Unified user context management across all agents."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.session_manager = FrameworkSessionManager(agent_type)
        self.conversation_manager = ConversationStateManager(agent_type)
        self.cross_agent_manager = CrossAgentDataManager(agent_type)
    
    async def initialize(self):
        """Initialize all managers."""
        try:
            await self.session_manager.initialize()
            await self.cross_agent_manager.initialize()
            logger.info(f"UnifiedUserContextManager initialized successfully for {self.agent_type}")
        except Exception as e:
            logger.warning(f"Failed to initialize managers for {self.agent_type}: {e}")
            # Continue without managers - will use fallback mode
    
    async def cleanup(self):
        """Clean up all managers."""
        try:
            if hasattr(self, 'cross_agent_manager'):
                await self.cross_agent_manager.cleanup()
            logger.info(f"UnifiedUserContextManager cleaned up for {self.agent_type}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def create_user_context(
        self, 
        user_id: str, 
        agent_type: str = None,
        initial_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a complete user context with session and conversation."""
        try:
            # Create user session
            session_data = await self.session_manager.create_user_session(
                user_id, 
                agent_type or self.agent_type
            )
            
            # Create conversation session
            conversation_data = await self.conversation_manager.create_conversation_session(
                user_id=user_id,
                session_id=session_data["conversation_id"],
                conversation_type="general",
                initial_context=initial_preferences or {}
            )
            
            # Store user preferences if provided
            if initial_preferences:
                await self._store_user_preferences(user_id, initial_preferences)
            
            # Create unified context
            user_context = {
                "user_id": user_id,
                "session_token": session_data["session_token"],
                "conversation_id": conversation_data["id"],
                "agent_type": agent_type or self.agent_type,
                "session_data": session_data,
                "conversation_data": conversation_data,
                "preferences": initial_preferences or {},
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Created unified user context for {user_id} on {self.agent_type} agent")
            return user_context
            
        except Exception as e:
            logger.error(f"Failed to create user context for {user_id}: {e}")
            # Return a minimal context on failure to prevent crashes
            return {
                "user_id": user_id,
                "session_token": None,
                "conversation_id": None,
                "agent_type": agent_type or self.agent_type,
                "session_data": {},
                "conversation_data": {},
                "preferences": {},
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def get_user_context(
        self, 
        session_token: str
    ) -> Optional[Dict[str, Any]]:
        """Get complete user context by session token."""
        try:
            # Get session data
            session_data = await self.session_manager.get_session(session_token)
            if not session_data:
                return None
            
            # Get conversation data
            conversation_data = await self.conversation_manager.get_conversation_session(
                session_data["conversation_id"]
            )
            
            # Get user preferences
            preferences = await self._get_user_preferences(session_data["user_id"])
            
            # Create unified context
            user_context = {
                "user_id": session_data["user_id"],
                "session_token": session_token,
                "conversation_id": session_data["conversation_id"],
                "agent_type": session_data["agent_type"],
                "session_data": session_data,
                "conversation_data": conversation_data or {},
                "preferences": preferences,
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"]
            }
            
            return user_context
            
        except Exception as e:
            logger.error(f"Failed to get user context for session {session_token}: {e}")
            return None
    
    async def update_user_context(
        self, 
        session_token: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update user context with new information."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return False
            
            # Update session activity
            await self.session_manager.update_session_activity(session_token)
            
            # Update conversation state if provided
            if "conversation_state" in updates:
                await self.conversation_manager.update_conversation_state(
                    current_context["conversation_id"],
                    updates["conversation_state"]
                )
            
            # Update preferences if provided
            if "preferences" in updates:
                await self._store_user_preferences(
                    current_context["user_id"],
                    updates["preferences"]
                )
            
            logger.info(f"Updated user context for session {session_token}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user context for session {session_token}: {e}")
            return False
    
    async def add_user_message(
        self, 
        session_token: str, 
        message_content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a user message to the conversation."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return False
            
            # Add message to conversation
            success = await self.conversation_manager.add_message_to_conversation(
                current_context["conversation_id"],
                "user",
                message_content,
                metadata
            )
            
            if success:
                # Update session activity
                await self.session_manager.update_session_activity(session_token)
                logger.info(f"Added user message to conversation {current_context['conversation_id']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add user message for session {session_token}: {e}")
            return False
    
    async def add_assistant_message(
        self, 
        session_token: str, 
        message_content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add an assistant message to the conversation."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return False
            
            # Add message to conversation
            success = await self.conversation_manager.add_message_to_conversation(
                current_context["conversation_id"],
                "assistant",
                message_content,
                metadata
            )
            
            if success:
                # Update session activity
                await self.session_manager.update_session_activity(session_token)
                logger.info(f"Added assistant message to conversation {current_context['conversation_id']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add assistant message for session {session_token}: {e}")
            return False
    
    async def get_conversation_history(
        self, 
        session_token: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return []
            
            # Get conversation messages
            messages = await self.conversation_manager.get_conversation_messages(
                current_context["conversation_id"],
                limit
            )
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for session {session_token}: {e}")
            return []
    
    async def get_user_preferences(
        self, 
        session_token: str
    ) -> Dict[str, Any]:
        """Get user preferences for a session."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return {}
            
            return current_context.get("preferences", {})
            
        except Exception as e:
            logger.error(f"Failed to get user preferences for session {session_token}: {e}")
            return {}
    
    async def update_user_preferences(
        self, 
        session_token: str, 
        new_preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences for a session."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return False
            
            # Store new preferences
            success = await self._store_user_preferences(
                current_context["user_id"],
                new_preferences
            )
            
            if success:
                # Share preferences with other agents
                await self.cross_agent_manager.share_user_preferences(
                    current_context["user_id"],
                    new_preferences
                )
                
                logger.info(f"Updated preferences for user {current_context['user_id']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update user preferences for session {session_token}: {e}")
            return False
    
    async def _store_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """Store user preferences in database."""
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
                logger.debug(f"Stored preferences for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store preferences for user {user_id}: {e}")
            return False
    
    async def _get_user_preferences(
        self, 
        user_id: str
    ) -> Dict[str, Any]:
        """Get user preferences from database."""
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
            logger.error(f"Failed to get preferences for user {user_id}: {e}")
            return {}
    
    async def cleanup_user_context(
        self, 
        session_token: str
    ) -> bool:
        """Clean up user context and end session."""
        try:
            # Delete session
            success = await self.session_manager.delete_session(session_token)
            
            if success:
                logger.info(f"Cleaned up user context for session {session_token}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cleanup user context for session {session_token}: {e}")
            return False
    
    async def get_user_summary(
        self, 
        session_token: str
    ) -> Optional[Dict[str, Any]]:
        """Get a summary of user activity and context."""
        try:
            # Get current context
            current_context = await self.get_user_context(session_token)
            if not current_context:
                return None
            
            # Get conversation summary
            conversation_summary = await self.conversation_manager.get_conversation_summary(
                current_context["conversation_id"]
            )
            
            # Create user summary
            user_summary = {
                "user_id": current_context["user_id"],
                "agent_type": current_context["agent_type"],
                "session_created": current_context["created_at"],
                "last_activity": current_context["last_activity"],
                "conversation_summary": conversation_summary,
                "preferences_keys": list(current_context.get("preferences", {}).keys()),
                "context_status": "active"
            }
            
            return user_summary
            
        except Exception as e:
            logger.error(f"Failed to get user summary for session {session_token}: {e}")
            return None 