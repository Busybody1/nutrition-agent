"""
Database utilities for the Nutrition Agent - Multi-Database Support.

This module provides database connection management for different databases:
- USER_DATABASE_URI: User management and authentication
- DATABASE_URL: Main agent database for sessions and conversations
- NUTRITION_DB_URI: Read-only food and nutrition reference data
- WORKOUT_DB_URI: Read-only workout and exercise reference data

FIXES:
- Automatically converts postgres:// to postgresql:// for compatibility
- Handles connection errors gracefully
- Provides connection status monitoring
"""

import os
import logging
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse

# Create Base class for SQLAlchemy models
Base = declarative_base()
metadata = MetaData()

# Set up logging
logger = logging.getLogger(__name__)

def fix_database_url(url: str) -> str:
    """
    Fix database URL protocol from postgres:// to postgresql://
    This is a common issue with Heroku and other providers that use postgres://
    but SQLAlchemy requires postgresql://
    """
    if not url:
        return url
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if it's a postgres URL that needs fixing
        if parsed.scheme == 'postgres':
            # Create new URL with postgresql scheme
            fixed_url = urlunparse(('postgresql',) + parsed[1:])
            logger.info(f"Fixed database URL protocol: {parsed.scheme}:// -> postgresql://")
            return fixed_url
        elif parsed.scheme == 'postgresql':
            # Already correct
            return url
        else:
            # Other database types, return as-is
            return url
            
    except Exception as e:
        logger.warning(f"Could not parse database URL for protocol fix: {e}")
        # If parsing fails, try simple string replacement
        if url.startswith('postgres://'):
            fixed_url = url.replace('postgres://', 'postgresql://', 1)
            logger.info("Fixed database URL protocol using string replacement")
            return fixed_url
        return url

def get_user_db() -> Session:
    """Get database session for user database (USER_DATABASE_URI)."""
    user_db_uri = os.getenv("USER_DATABASE_URI")
    
    if not user_db_uri:
        raise ValueError("USER_DATABASE_URI environment variable not set")
    
    # Fix the database URL protocol if needed
    fixed_uri = fix_database_url(user_db_uri)
    
    try:
        engine = create_engine(
            fixed_uri,
            pool_size=5,  # Smaller pool for user database
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10}
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create user database connection: {e}")
        raise

def get_main_db() -> Session:
    """Get database session for main agent database (DATABASE_URL)."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Fix the database URL protocol if needed
    fixed_url = fix_database_url(database_url)
    
    try:
        engine = create_engine(
            fixed_url,
            pool_size=10,  # Reasonable pool size for 20-30 users
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10}
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create main database connection: {e}")
        raise

def get_nutrition_db() -> Optional[Session]:
    """Get database session for nutrition database (NUTRITION_DB_URI) - Optional."""
    nutrition_db_uri = os.getenv("NUTRITION_DB_URI")
    
    if not nutrition_db_uri:
        logger.warning("NUTRITION_DB_URI not configured - nutrition features will be limited")
        return None
    
    # Fix the database URL protocol if needed
    fixed_uri = fix_database_url(nutrition_db_uri)
    
    try:
        engine = create_engine(
            fixed_uri,
            pool_size=5,  # Read-only, smaller pool
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10}
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create nutrition database connection: {e}")
        return None

def get_workout_db() -> Optional[Session]:
    """Get database session for workout database (WORKOUT_DB_URI) - Optional."""
    workout_db_uri = os.getenv("WORKOUT_DB_URI")
    
    if not workout_db_uri:
        logger.warning("WORKOUT_DB_URI not configured - workout features will be limited")
        return None
    
    # Fix the database URL protocol if needed
    fixed_uri = fix_database_url(workout_db_uri)
    
    try:
        engine = create_engine(
            fixed_uri,
            pool_size=5,  # Read-only, smaller pool
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10}
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create workout database connection: {e}")
        return None

def test_database_connection(db_type: str) -> Tuple[bool, str]:
    """
    Test database connection and return status.
    
    Args:
        db_type: Type of database to test ('main', 'user', 'nutrition', 'workout')
    
    Returns:
        Tuple[bool, str]: (is_connected, message)
    """
    try:
        if db_type == "main":
            db = get_main_db()
            db.execute(text("SELECT 1"))
            db.close()
            return True, "Main database connection successful"
            
        elif db_type == "user":
            db = get_user_db()
            db.execute(text("SELECT 1"))
            db.close()
            return True, "User database connection successful"
            
        elif db_type == "nutrition":
            db = get_nutrition_db()
            if db is None:
                return False, "Nutrition database not configured"
            db.execute(text("SELECT 1"))
            db.close()
            return True, "Nutrition database connection successful"
            
        elif db_type == "workout":
            db = get_workout_db()
            if db is None:
                return False, "Workout database not configured"
            db.execute(text("SELECT 1"))
            db.close()
            return True, "Workout database connection successful"
            
        else:
            return False, f"Unknown database type: {db_type}"
            
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def get_database_status() -> dict:
    """
    Get real-time status of all database connections.
    
    Returns:
        dict: Status of all databases with connection details
    """
    status = {}
    
    # Test each database
    for db_type in ["main", "user", "nutrition", "workout"]:
        is_connected, message = test_database_connection(db_type)
        status[db_type] = {
            "connected": is_connected,
            "message": message,
            "timestamp": str(os.getenv("DATABASE_URL" if db_type == "main" else f"{db_type.upper()}_DATABASE_URI", "Not configured"))[:20] + "..."
        }
    
    return status
