"""
Database utilities for the AI Agent Framework - Multi-Database Support.

This module provides database connection management for different databases:
- USER_DATABASE_URI: User management and authentication
- DATABASE_URL: Main agent database for sessions and conversations
- NUTRITION_DB_URI: Read-only food and nutrition reference data
- WORKOUT_DB_URI: Read-only workout and exercise reference data
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional

def get_user_db() -> Session:
    """Get database session for user database (USER_DATABASE_URI)."""
    user_db_uri = os.getenv("USER_DATABASE_URI")
    
    if not user_db_uri:
        raise ValueError("USER_DATABASE_URI environment variable not set")
    
    engine = create_engine(
        user_db_uri,
        pool_size=5,  # Smaller pool for user database
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_main_db() -> Session:
    """Get database session for main agent database (DATABASE_URL)."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_size=10,  # Reasonable pool size for 20-30 users
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_nutrition_db() -> Optional[Session]:
    """Get database session for nutrition database (NUTRITION_DB_URI) - Optional."""
    nutrition_db_uri = os.getenv("NUTRITION_DB_URI")
    
    if not nutrition_db_uri:
        return None
    
    try:
        engine = create_engine(
            nutrition_db_uri,
            pool_size=3,  # Small pool for read-only reference data
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception:
        return None

def get_workout_db() -> Optional[Session]:
    """Get database session for workout database (WORKOUT_DB_URI) - Optional."""
    workout_db_uri = os.getenv("WORKOUT_DB_URI")
    
    if not workout_db_uri:
        return None
    
    try:
        engine = create_engine(
            workout_db_uri,
            pool_size=3,  # Small pool for read-only reference data
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    except Exception:
        return None

# Legacy function for backward compatibility
def get_db() -> Session:
    """Get main database session - backward compatibility."""
    return get_main_db()

def test_database_connection(database_type: str = "main") -> tuple[bool, str]:
    """Test database connection with a simple query."""
    try:
        if database_type == "user":
            db = get_user_db()
        elif database_type == "nutrition":
            db = get_nutrition_db()
            if not db:
                return False, "NUTRITION_DB_URI not configured"
        elif database_type == "workout":
            db = get_workout_db()
            if not db:
                return False, "WORKOUT_DB_URI not configured"
        else:  # main
            db = get_main_db()
        
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        return True, str(result[0]) if result else "No result"
    except Exception as e:
        return False, str(e)
