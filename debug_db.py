#!/usr/bin/env python3
"""
Debug script to check environment variables and database connections
"""

import os
import logging
from sqlalchemy import text, create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_environment():
    """Debug environment variables and database connections."""
    
    logger.info("=== ENVIRONMENT VARIABLES ===")
    
    # Check all relevant environment variables
    env_vars = [
        'DATABASE_URL',
        'NUTRITION_DB_URI',
        'WORKOUT_DB_URI',
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Show first 50 chars of URL for security
            display_value = value[:50] + "..." if len(value) > 50 else value
            logger.info(f"{var}: {display_value}")
        else:
            logger.warning(f"{var}: NOT SET")
    
    logger.info("=== DATABASE CONNECTION TESTS ===")
    
    # Test DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        logger.info("Testing DATABASE_URL connection...")
        try:
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            engine = create_engine(database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).scalar()
                logger.info(f"✓ DATABASE_URL connection successful: {result}")
                
                # Test if tables exist
                tables = ['users', 'food_logs', 'workout_logs']
                for table in tables:
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                        logger.info(f"✓ Table '{table}' exists with {count} rows")
                    except Exception as e:
                        logger.error(f"✗ Table '{table}' does not exist: {e}")
                        
        except Exception as e:
            logger.error(f"✗ DATABASE_URL connection failed: {e}")
    else:
        logger.warning("DATABASE_URL not set")
    
    # Test NUTRITION_DB_URI
    nutrition_db_uri = os.getenv('NUTRITION_DB_URI')
    if nutrition_db_uri:
        logger.info("Testing NUTRITION_DB_URI connection...")
        try:
            if nutrition_db_uri.startswith("postgres://"):
                nutrition_db_uri = nutrition_db_uri.replace("postgres://", "postgresql://", 1)
            
            engine = create_engine(nutrition_db_uri)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).scalar()
                logger.info(f"✓ NUTRITION_DB_URI connection successful: {result}")
                
                # Test if foods table exists
                try:
                    count = conn.execute(text("SELECT COUNT(*) FROM foods")).scalar()
                    logger.info(f"✓ Table 'foods' exists with {count} rows")
                    
                    # Show sample foods
                    foods = conn.execute(text("SELECT id, name, calories FROM foods LIMIT 3")).fetchall()
                    logger.info(f"✓ Sample foods:")
                    for food in foods:
                        logger.info(f"  - {food[1]} ({food[2]} calories)")
                        
                except Exception as e:
                    logger.error(f"✗ Table 'foods' does not exist: {e}")
                    
        except Exception as e:
            logger.error(f"✗ NUTRITION_DB_URI connection failed: {e}")
    else:
        logger.warning("NUTRITION_DB_URI not set")
    
    logger.info("=== CONFIGURATION TEST ===")
    
    # Test the configuration system
    try:
        from shared.config import get_settings
        settings = get_settings()
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"Database URL: {settings.database.url[:50]}...")
        logger.info(f"Multi DB settings: {settings.multi_db}")
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
    
    logger.info("=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_environment() 