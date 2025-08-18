"""
Session Middleware for Nutrition Agent

This module provides FastAPI middleware for validating user sessions
and extracting user information from requests.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from .session_manager import FrameworkSessionManager

logger = logging.getLogger(__name__)

class SessionMiddleware:
    """Middleware for session validation and user extraction."""
    
    def __init__(self, agent_type: str = "nutrition"):
        self.agent_type = agent_type
        self.session_manager = FrameworkSessionManager(agent_type)
    
    async def validate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate request and return session data if valid."""
        try:
            session_token = self._extract_session_token(request)
            if not session_token:
                return None
            
            # Validate session
            if not await self.session_manager.validate_session(session_token):
                return None
            
            # Get session data
            session_data = await self.session_manager.get_session(session_token)
            return session_data
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None
    
    def _extract_session_token(self, request: Request) -> Optional[str]:
        """Extract session token from request headers or query parameters."""
        # Check Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Check X-Session-Token header
        session_header = request.headers.get("X-Session-Token")
        if session_header:
            return session_header
        
        # Check query parameter
        session_param = request.query_params.get("session_token")
        if session_param:
            return session_param
        
        # Check cookies
        session_cookie = request.cookies.get("session_token")
        if session_cookie:
            return session_cookie
        
        return None

def require_valid_session(func):
    """Decorator to require valid session for endpoint."""
    async def wrapper(*args, **kwargs):
        # Extract request from args or kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            for key, value in kwargs.items():
                if isinstance(value, Request):
                    request = value
                    break
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request object not found"
            )
        
        # Create session manager
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        # Validate session
        session_token = session_manager._extract_session_token(request)
        if not session_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session token required"
            )
        
        if not await session_manager.validate_session(session_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        # Add session data to request state
        session_data = await session_manager.get_session(session_token)
        request.state.session = session_data
        request.state.user_id = session_data["user_id"]
        
        return await func(*args, **kwargs)
    
    return wrapper

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from request session."""
    if not hasattr(request.state, 'session'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid session required"
        )
    
    return request.state.session

async def get_current_user_id(request: Request) -> str:
    """Get current user ID from request session."""
    if not hasattr(request.state, 'user_id'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid session required"
        )
    
    return request.state.user_id

class SessionValidationMiddleware:
    """FastAPI middleware for automatic session validation."""
    
    def __init__(self, app, agent_type: str = "nutrition"):
        self.app = app
        self.agent_type = agent_type
        self.session_manager = FrameworkSessionManager(agent_type)
    
    def _extract_session_token(self, request: Request) -> Optional[str]:
        """Extract session token from request headers or query parameters."""
        # Check Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Check X-Session-Token header
        session_header = request.headers.get("X-Session-Token")
        if session_header:
            return session_header
        
        # Check query parameter
        session_param = request.query_params.get("session_token")
        if session_param:
            return session_param
        
        # Check cookies
        session_cookie = request.cookies.get("session_token")
        if session_cookie:
            return session_cookie
        
        return None
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Initialize session manager
            await self.session_manager.initialize()
            
            # Create request object for validation
            request = Request(scope, receive)
            
            # Extract session token
            session_token = self._extract_session_token(request)
            
            if session_token:
                # Validate session
                try:
                    if await self.session_manager.validate_session(session_token):
                        # Add session data to scope
                        session_data = await self.session_manager.get_session(session_token)
                        scope["session"] = session_data
                        scope["user_id"] = session_data["user_id"]
                        logger.debug(f"Valid session for user {session_data['user_id']}")
                    else:
                        logger.warning(f"Invalid session token: {session_token}")
                except Exception as e:
                    logger.error(f"Session validation error: {e}")
        
        await self.app(scope, receive, send)

async def cleanup_expired_sessions():
    """Background task to cleanup expired sessions."""
    try:
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        cleaned_count = await session_manager.cleanup_expired_sessions()
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            
    except Exception as e:
        logger.error(f"Failed to cleanup expired sessions: {e}")

# Utility functions for session management
async def create_user_session(user_id: str) -> Dict[str, Any]:
    """Create a new user session."""
    try:
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        session_data = await session_manager.create_user_session(user_id)
        logger.info(f"Created session for user {user_id}")
        return session_data
        
    except Exception as e:
        logger.error(f"Failed to create session for user {user_id}: {e}")
        raise

async def validate_session_token(session_token: str) -> bool:
    """Validate a session token."""
    try:
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        is_valid = await session_manager.validate_session(session_token)
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to validate session token: {e}")
        return False

async def get_session_data(session_token: str) -> Optional[Dict[str, Any]]:
    """Get session data by token."""
    try:
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        session_data = await session_manager.get_session(session_token)
        return session_data
        
    except Exception as e:
        logger.error(f"Failed to get session data: {e}")
        return None

async def delete_user_session(session_token: str) -> bool:
    """Delete a user session."""
    try:
        session_manager = FrameworkSessionManager("nutrition")
        await session_manager.initialize()
        
        success = await session_manager.delete_session(session_token)
        if success:
            logger.info(f"Deleted session {session_token}")
        return success
        
    except Exception as e:
        logger.error(f"Failed to delete session {session_token}: {e}")
        return False 