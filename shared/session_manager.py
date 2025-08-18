"""
Unified Session Management for Nutrition Agent

This module provides session management for the nutrition agent,
enabling scalability from single-user to 100+ concurrent users.
"""

import asyncio
import json
import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List
from uuid import UUID, uuid4

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
from sqlalchemy import select, update, delete

from .database import get_async_session
from .models import UserSessionORM, AgentConversationSessionORM, MessageHistoryORM

logger = logging.getLogger(__name__)

class FrameworkSessionManager:
    """Unified session management for the nutrition agent."""
    
    def __init__(self, agent_type: str = "nutrition"):
        self.agent_type = agent_type
        self.redis_client: Optional[Any] = None
        self.session_cache_ttl = 600  # 10 minutes
    
    async def initialize(self):
        """Initialize Redis client for caching."""
        try:
            from .database import get_redis
            self.redis_client = await get_redis()
            logger.info(f"Redis client initialized for {self.agent_type} agent")
        except Exception as e:
            logger.warning(f"Redis not available for {self.agent_type} agent: {e}")
            self.redis_client = None
            # Continue without Redis - database-only mode
            logger.info(f"Continuing in database-only mode for {self.agent_type} agent")
    
    async def create_user_session(self, user_id: Union[str, UUID], agent_type: str = None) -> Dict[str, Any]:
        """Create session for nutrition agent."""
        if agent_type is None:
            agent_type = self.agent_type
            
        user_id = str(user_id)
        session_token = self._generate_session_token(user_id)
        conversation_id = str(uuid4())
        
        session_data = {
            "user_id": user_id,
            "session_token": session_token,
            "agent_type": agent_type,
            "conversation_id": conversation_id,
            "agent_context": {},
            "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await self._store_session_in_db(session_data)
        
        if self.redis_client and REDIS_AVAILABLE:
            await self._cache_session(session_data)
        
        logger.info(f"Created session for user {user_id} on {agent_type} agent")
        return session_data
    
    async def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session by token from cache or database."""
        # Try Redis cache first
        if self.redis_client and REDIS_AVAILABLE:
            cached_session = await self._get_cached_session(session_token)
            if cached_session:
                return cached_session
        
        # Fallback to database
        return await self._get_session_from_db(session_token)
    
    async def validate_session(self, session_token: str) -> bool:
        """Validate if session exists and is active."""
        session = await self.get_session(session_token)
        if not session:
            return False
        
        # Check if session is expired
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.utcnow() > expires_at:
            await self.delete_session(session_token)
            return False
        
        # Check if session is active
        if not session.get("is_active", True):
            return False
        
        # Update last activity
        await self.update_session_activity(session_token)
        return True
    
    async def update_session_activity(self, session_token: str):
        """Update session last activity timestamp."""
        try:
            async with get_async_session() as session:
                await session.execute(
                    update(UserSessionORM)
                    .where(UserSessionORM.session_token == session_token)
                    .values(
                        last_activity=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                # Update cache if available
                if self.redis_client and REDIS_AVAILABLE:
                    await self._update_cached_session(session_token, {"last_activity": datetime.utcnow().isoformat()})
                    
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
    
    async def delete_session(self, session_token: str) -> bool:
        """Delete a session."""
        try:
            async with get_async_session() as session:
                await session.execute(
                    delete(UserSessionORM)
                    .where(UserSessionORM.session_token == session_token)
                )
                await session.commit()
                
                # Remove from cache if available
                if self.redis_client and REDIS_AVAILABLE:
                    await self._remove_cached_session(session_token)
                
                logger.info(f"Deleted session {session_token}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    async def get_user_sessions(self, user_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(UserSessionORM)
                    .where(UserSessionORM.user_id == str(user_id))
                    .where(UserSessionORM.is_active == True)
                    .where(UserSessionORM.expires_at > datetime.utcnow())
                )
                
                sessions = result.scalars().all()
                return [self._session_to_dict(s) for s in sessions]
                
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def extend_session(self, session_token: str, minutes: int = 30) -> bool:
        """Extend session expiration time."""
        try:
            new_expires_at = datetime.utcnow() + timedelta(minutes=minutes)
            
            async with get_async_session() as session:
                await session.execute(
                    update(UserSessionORM)
                    .where(UserSessionORM.session_token == session_token)
                    .values(
                        expires_at=new_expires_at,
                        updated_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
                # Update cache if available
                if self.redis_client and REDIS_AVAILABLE:
                    await self._update_cached_session(session_token, {"expires_at": new_expires_at.isoformat()})
                
                logger.info(f"Extended session {session_token} by {minutes} minutes")
                return True
                
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    delete(UserSessionORM)
                    .where(UserSessionORM.expires_at < datetime.utcnow())
                )
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate cryptographically secure session token."""
        random_bytes = secrets.token_bytes(32)
        user_hash = hashlib.sha256(str(user_id).encode()).hexdigest()
        return f"{user_hash}:{random_bytes.hex()}"
    
    async def _store_session_in_db(self, session_data: Dict[str, Any]):
        """Store session data in database."""
        try:
            async with get_async_session() as session:
                db_session = UserSessionORM(
                    id=uuid4(),
                    user_id=session_data["user_id"],
                    session_token=session_data["session_token"],
                    agent_type=session_data["agent_type"],
                    conversation_id=session_data["conversation_id"],
                    agent_context=session_data["agent_context"],
                    expires_at=datetime.fromisoformat(session_data["expires_at"]),
                    is_active=session_data["is_active"],
                    created_at=datetime.fromisoformat(session_data["created_at"]),
                    updated_at=datetime.fromisoformat(session_data["updated_at"])
                )
                
                session.add(db_session)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store session in database: {e}")
            raise
    
    async def _cache_session(self, session_data: Dict[str, Any]):
        """Cache session data in Redis."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return
        
        try:
            cache_key = f"session:{session_data['session_token']}"
            await self.redis_client.setex(
                cache_key,
                self.session_cache_ttl,
                json.dumps(session_data)
            )
        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
    
    async def _get_cached_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session from Redis cache."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return None
        
        try:
            cache_key = f"session:{session_token}"
            cached = await self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to get cached session: {e}")
        
        return None
    
    async def _get_session_from_db(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session from database."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(UserSessionORM)
                    .where(UserSessionORM.session_token == session_token)
                )
                
                db_session = result.scalar_one_or_none()
                if db_session:
                    return self._session_to_dict(db_session)
                    
        except Exception as e:
            logger.error(f"Failed to get session from database: {e}")
        
        return None
    
    async def _update_cached_session(self, session_token: str, updates: Dict[str, Any]):
        """Update cached session data."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return
        
        try:
            cache_key = f"session:{session_token}"
            cached = await self.redis_client.get(cache_key)
            if cached:
                session_data = json.loads(cached)
                session_data.update(updates)
                await self.redis_client.setex(
                    cache_key,
                    self.session_cache_ttl,
                    json.dumps(session_data)
                )
        except Exception as e:
            logger.warning(f"Failed to update cached session: {e}")
    
    async def _remove_cached_session(self, session_token: str):
        """Remove session from cache."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return
        
        try:
            cache_key = f"session:{session_token}"
            await self.redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to remove cached session: {e}")
    
    def _session_to_dict(self, session: UserSessionORM) -> Dict[str, Any]:
        """Convert session ORM object to dictionary."""
        return {
            "id": str(session.id),
            "user_id": str(session.user_id),
            "session_token": session.session_token,
            "agent_type": session.agent_type,
            "conversation_id": str(session.conversation_id),
            "agent_context": session.agent_context,
            "last_activity": session.last_activity.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "is_active": session.is_active,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        } 