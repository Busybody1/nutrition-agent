#!/usr/bin/env python3
"""
Test script to verify Heroku database tables are working
"""

import os
import logging
from sqlalchemy import text, create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_heroku_database():
    """Test the Heroku database connection and tables."""
    
    # Get database URLs from environment
    database_url = os.getenv('DATABASE_URL')
    nutrition_db_uri = os.getenv('NUTRITION_DB_URI')
    
    logger.info(f"Testing with DATABASE_URL: {database_url[:50]}..." if database_url else "DATABASE_URL not set")
    logger.info(f"Testing with NUTRITION_DB_URI: {nutrition_db_uri[:50]}..." if nutrition_db_uri else "NUTRITION_DB_URI not set")
    
    # Test shared database (DATABASE_URL)
    if database_url:
        try:
            # Fix postgres:// to postgresql://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            engine = create_engine(database_url)
            with engine.connect() as conn:
                # Test if tables exist
                tables = ['users', 'food_logs', 'workout_logs']
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    logger.info(f"✓ Shared DB - Table '{table}' exists with {result} rows")
                
                # Test if we can insert a test user
                test_user_id = "550e8400-e29b-41d4-a716-446655440000"
                conn.execute(text("""
                    INSERT INTO users (id, email, username, first_name, last_name, created_at, updated_at)
                    VALUES (:id, :email, :username, :first_name, :last_name, NOW(), NOW())
                    ON CONFLICT (id) DO NOTHING
                """), {
                    "id": test_user_id,
                    "email": "test@example.com",
                    "username": "testuser",
                    "first_name": "Test",
                    "last_name": "User"
                })
                conn.commit()
                logger.info("✓ Successfully inserted test user in shared database")
                
        except Exception as e:
            logger.error(f"✗ Shared database test failed: {e}")
            return False
    else:
        logger.warning("DATABASE_URL not set - skipping shared database test")
    
    # Test nutrition database (NUTRITION_DB_URI)
    if nutrition_db_uri:
        try:
            # Fix postgres:// to postgresql://
            if nutrition_db_uri.startswith("postgres://"):
                nutrition_db_uri = nutrition_db_uri.replace("postgres://", "postgresql://", 1)
            
            engine = create_engine(nutrition_db_uri)
            with engine.connect() as conn:
                # Test if foods table exists
                result = conn.execute(text("SELECT COUNT(*) FROM foods")).scalar()
                logger.info(f"✓ Nutrition DB - Table 'foods' exists with {result} rows")
                
                # Test if we can query food data
                foods = conn.execute(text("SELECT id, name, calories FROM foods LIMIT 5")).fetchall()
                logger.info(f"✓ Found {len(foods)} food items in nutrition database")
                for food in foods:
                    logger.info(f"  - {food[1]} ({food[2]} calories)")
                    
        except Exception as e:
            logger.error(f"✗ Nutrition database test failed: {e}")
            return False
    else:
        logger.warning("NUTRITION_DB_URI not set - skipping nutrition database test")
    
    logger.info("✓ All database tests passed!")
    return True

if __name__ == "__main__":
    success = test_heroku_database()
    if success:
        print("Heroku database tables are working correctly!")
    else:
        print("Heroku database tests failed!")
        exit(1) 