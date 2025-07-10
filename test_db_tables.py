#!/usr/bin/env python3
"""
Test script to verify database tables are created correctly
"""

import asyncio
import logging
from sqlalchemy import text
from shared.database import get_fitness_db_engine, get_nutrition_db_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_tables():
    """Test that all required tables exist and are accessible."""
    
    # Test fitness database (shared database)
    logger.info("Testing fitness database tables...")
    try:
        fitness_engine = get_fitness_db_engine()
        with fitness_engine.connect() as conn:
            # Test if tables exist
            tables = ['users', 'food_logs', 'workout_logs']
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"✓ Table '{table}' exists with {result} rows")
            
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
            logger.info("✓ Successfully inserted test user")
            
    except Exception as e:
        logger.error(f"✗ Fitness database test failed: {e}")
        return False
    
    # Test nutrition database
    logger.info("Testing nutrition database tables...")
    try:
        nutrition_engine = get_nutrition_db_engine()
        with nutrition_engine.connect() as conn:
            # Test if foods table exists
            result = conn.execute(text("SELECT COUNT(*) FROM foods")).scalar()
            logger.info(f"✓ Table 'foods' exists with {result} rows")
            
            # Test if we can query food data
            foods = conn.execute(text("SELECT id, name, calories FROM foods LIMIT 5")).fetchall()
            logger.info(f"✓ Found {len(foods)} food items")
            for food in foods:
                logger.info(f"  - {food[1]} ({food[2]} calories)")
                
    except Exception as e:
        logger.error(f"✗ Nutrition database test failed: {e}")
        return False
    
    logger.info("✓ All database tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_database_tables())
    if success:
        print("Database tables are working correctly!")
    else:
        print("Database tests failed!")
        exit(1) 