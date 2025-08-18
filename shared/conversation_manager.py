"""
Conversation State Management for Multi-User AI Agent Framework

This module provides conversation state persistence and management,
enabling user-specific conversation context and history across all agents.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from .database import get_async_session
from .models import (
    AgentConversationSessionORM,
    MessageHistoryORM,
    UserSessionORM
)

logger = logging.getLogger(__name__)

class ConversationStateManager:
    """Manages conversation state persistence and retrieval for multi-user support."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
    
    async def create_conversation_session(
        self, 
        user_id: str, 
        session_id: str,
        conversation_type: str = "general",
        initial_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new conversation session for a user."""
        try:
            from .database import get_async_session
            async with get_async_session() as session:
                conversation_id = str(uuid4())
                
                conversation_data = {
                    "id": conversation_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_type": self.agent_type,
                    "conversation_type": conversation_type,
                    "current_state": initial_context or {},
                    "conversation_history": [],
                    "agent_responses": [],
                    "context_data": {},
                    "agent_specific_data": {},
                    "metadata": {
                        "created_by": self.agent_type,
                        "conversation_type": conversation_type,
                        "initial_context": initial_context
                    }
                }
                
                conversation_orm = AgentConversationSessionORM(
                    id=conversation_id,
                    user_id=user_id,
                    session_id=session_id,
                    agent_type=self.agent_type,
                    conversation_type=conversation_type,
                    current_state=initial_context or {},
                    conversation_history=[],
                    agent_responses=[],
                    context_data={},
                    agent_specific_data={},
                    metadata=conversation_data["metadata"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    last_message_at=datetime.utcnow()
                )
                
                session.add(conversation_orm)
                await session.commit()
                
                logger.info(f"Created conversation session {conversation_id} for user {user_id}")
                return conversation_data
                
        except Exception as e:
            logger.error(f"Failed to create conversation session for user {user_id}: {e}")
            raise
    
    async def get_conversation_session(
        self, 
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get conversation session by ID."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.id == conversation_id)
                    .options(selectinload(AgentConversationSessionORM.messages))
                )
                
                conversation = result.scalar_one_or_none()
                if conversation:
                    return self._conversation_to_dict(conversation)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get conversation session {conversation_id}: {e}")
            return None
    
    async def get_user_conversations(
        self, 
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent conversations for a user."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.user_id == user_id)
                    .where(AgentConversationSessionORM.agent_type == self.agent_type)
                    .order_by(AgentConversationSessionORM.last_message_at.desc())
                    .limit(limit)
                )
                
                conversations = result.scalars().all()
                return [self._conversation_to_dict(conv) for conv in conversations]
                
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []
    
    async def update_conversation_state(
        self, 
        conversation_id: str, 
        new_state: Dict[str, Any]
    ) -> bool:
        """Update conversation state."""
        try:
            async with get_async_session() as session:
                await session.execute(
                    update(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.id == conversation_id)
                    .values(
                        current_state=new_state,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                logger.info(f"Updated conversation state for {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update conversation state for {conversation_id}: {e}")
            return False
    
    async def add_message_to_conversation(
        self, 
        conversation_id: str, 
        message_type: str, 
        content: str, 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a message to conversation history."""
        try:
            async with get_async_session() as session:
                # Add message to history
                message_orm = MessageHistoryORM(
                    id=str(uuid4()),
                    user_id=await self._get_user_id_from_conversation(conversation_id),
                    conversation_id=conversation_id,
                    message_type=message_type,
                    content=content,
                    metadata=metadata or {},
                    timestamp=datetime.utcnow()
                )
                
                session.add(message_orm)
                
                # Update conversation last message time
                await session.execute(
                    update(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.id == conversation_id)
                    .values(
                        last_message_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.commit()
                
                logger.info(f"Added {message_type} message to conversation {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            return False
    
    async def get_conversation_messages(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(MessageHistoryORM)
                    .where(MessageHistoryORM.conversation_id == conversation_id)
                    .order_by(MessageHistoryORM.timestamp.desc())
                    .limit(limit)
                )
                
                messages = result.scalars().all()
                return [self._message_to_dict(msg) for msg in messages]
                
        except Exception as e:
            logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
            return []
    
    async def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up old conversations and return count of cleaned conversations."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            async with get_async_session() as session:
                result = await session.execute(
                    delete(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.last_message_at < cutoff_date)
                    .where(AgentConversationSessionORM.agent_type == self.agent_type)
                )
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} old conversations")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
            return 0
    
    async def get_conversation_summary(
        self, 
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a summary of conversation activity."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(AgentConversationSessionORM)
                    .where(AgentConversationSessionORM.id == conversation_id)
                )
                
                conversation = result.scalar_one_or_none()
                if not conversation:
                    return None
                
                # Get message count
                message_count = await self._get_message_count(conversation_id)
                
                return {
                    "conversation_id": conversation_id,
                    "user_id": str(conversation.user_id),
                    "agent_type": conversation.agent_type,
                    "conversation_type": conversation.conversation_type,
                    "message_count": message_count,
                    "created_at": conversation.created_at.isoformat(),
                    "last_message_at": conversation.last_message_at.isoformat(),
                    "current_state_keys": list(conversation.current_state.keys()) if conversation.current_state else []
                }
                
        except Exception as e:
            logger.error(f"Failed to get conversation summary for {conversation_id}: {e}")
            return None
    
    async def _get_user_id_from_conversation(self, conversation_id: str) -> str:
        """Get user ID from conversation ID."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(AgentConversationSessionORM.user_id)
                    .where(AgentConversationSessionORM.id == conversation_id)
                )
                
                user_id = result.scalar_one_or_none()
                return str(user_id) if user_id else None
                
        except Exception as e:
            logger.error(f"Failed to get user ID from conversation {conversation_id}: {e}")
            return None
    
    async def _get_message_count(self, conversation_id: str) -> int:
        """Get message count for a conversation."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(MessageHistoryORM)
                    .where(MessageHistoryORM.conversation_id == conversation_id)
                )
                
                return len(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Failed to get message count for conversation {conversation_id}: {e}")
            return 0
    
    def _conversation_to_dict(self, conversation: AgentConversationSessionORM) -> Dict[str, Any]:
        """Convert conversation ORM object to dictionary."""
        return {
            "id": str(conversation.id),
            "user_id": str(conversation.user_id),
            "session_id": str(conversation.session_id),
            "agent_type": conversation.agent_type,
            "conversation_type": conversation.conversation_type,
            "current_state": conversation.current_state,
            "conversation_history": conversation.conversation_history,
            "agent_responses": conversation.agent_responses,
            "context_data": conversation.context_data,
            "agent_specific_data": conversation.agent_specific_data,
            "metadata": conversation.metadata,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "last_message_at": conversation.last_message_at.isoformat()
        }
    
    def _message_to_dict(self, message: MessageHistoryORM) -> Dict[str, Any]:
        """Convert message ORM object to dictionary."""
        return {
            "id": str(message.id),
            "user_id": str(message.user_id),
            "conversation_id": str(message.conversation_id),
            "message_type": message.message_type,
            "content": message.content,
            "metadata": message.metadata,
            "timestamp": message.timestamp.isoformat()
        } 