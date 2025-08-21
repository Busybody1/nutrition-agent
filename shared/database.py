"""
Database utilities for the AI Agent Framework - Simplified.

This module provides simple database connection management for PostgreSQL.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

def get_db() -> Session:
    """Get database session - simple and reliable."""
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

def test_database_connection():
    """Test database connection with a simple query."""
    try:
        db = get_db()
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        return True, result[0]
    except Exception as e:
        return False, str(e)
