#!/usr/bin/env python3
"""
Database Migration Script for Fitness AI Agent Framework
This script creates all necessary database tables.
"""

import os
import sys
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from datetime import datetime, timezone

def utc_now():
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)

def create_users_table(engine):
    """Create users table."""
    metadata = MetaData()
    
    users_table = Table(
        'users', metadata,
        Column('id', PGUUID(as_uuid=True), primary_key=True),
        Column('email', String(255), unique=True, nullable=False),
        Column('username', String(100), unique=True, nullable=False),
        Column('hashed_password', String(255), nullable=False),
        Column('first_name', String(100)),
        Column('last_name', String(100)),
        Column('date_of_birth', DateTime),
        Column('gender', String(20)),
        Column('height_cm', Float),
        Column('weight_kg', Float),
        Column('timezone', String(50)),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime, default=utc_now),
        Column('updated_at', DateTime, default=utc_now, onupdate=utc_now)
    )
    
    metadata.create_all(engine)
    print("Users table created")

def create_user_sessions_table(engine):
    """Create user sessions table."""
    metadata = MetaData()
    
    sessions_table = Table(
        'user_sessions', metadata,
        Column('id', PGUUID(as_uuid=True), primary_key=True),
        Column('user_id', PGUUID(as_uuid=True), nullable=False),
        Column('session_token', String(255), unique=True, nullable=False),
        Column('agent_type', String(50), nullable=False),
        Column('conversation_id', PGUUID(as_uuid=True), nullable=False),
        Column('agent_context', JSONB, default={}),
        Column('last_activity', DateTime, nullable=False, default=utc_now),
        Column('expires_at', DateTime, nullable=False),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime, nullable=False, default=utc_now),
        Column('updated_at', DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    )
    
    metadata.create_all(engine)
    print("User sessions table created")

def main():
    """Main migration function."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("Database connection successful")
        
        # Create tables
        create_users_table(engine)
        create_user_sessions_table(engine)
        
        print("Database migration completed successfully!")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
