#!/usr/bin/env python3
"""
Database initialization script for Nutrition Agent
Creates all required tables and indexes with proper structure
"""

import asyncio
import logging
import os
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database_sync():
    """Initialize database with tables and indexes synchronously."""
    try:
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        nutrition_db_uri = os.getenv('NUTRITION_DB_URI')
        
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            return False
        
        # Fix postgres:// to postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        logger.info("Creating database engine...")
        engine = create_engine(database_url)
        
        # Create tables
        logger.info("Creating tables...")
        
        # Create users table
        engine.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(50) UNIQUE NOT NULL,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                age INTEGER,
                height_cm FLOAT,
                weight_kg FLOAT,
                primary_goal VARCHAR(50),
                daily_calorie_target INTEGER,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE
            )
        """))
        logger.info("✓ Created users table")
        
        # Create foods table
        engine.execute(text("""
            CREATE TABLE IF NOT EXISTS foods (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                brand_id VARCHAR(255),
                category_id VARCHAR(100) NOT NULL,
                serving_size FLOAT NOT NULL,
                serving_unit VARCHAR(50) NOT NULL,
                serving VARCHAR(100),
                calories FLOAT NOT NULL,
                protein_g FLOAT NOT NULL,
                carbs_g FLOAT NOT NULL,
                fat_g FLOAT NOT NULL,
                fiber_g FLOAT DEFAULT 0,
                sugar_g FLOAT DEFAULT 0,
                source VARCHAR(50) DEFAULT 'user_input',
                verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        logger.info("✓ Created foods table")
        
        # Create food_logs table
        engine.execute(text("""
            CREATE TABLE IF NOT EXISTS food_logs (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                food_item_id UUID NOT NULL,
                quantity_g FLOAT NOT NULL,
                meal_type VARCHAR(20) NOT NULL,
                consumed_at TIMESTAMP NOT NULL,
                calories FLOAT NOT NULL,
                protein_g FLOAT NOT NULL,
                carbs_g FLOAT NOT NULL,
                fat_g FLOAT NOT NULL,
                serving_size FLOAT,
                serving_unit VARCHAR(50),
                serving VARCHAR(100),
                notes TEXT,
                created_at TIMESTAMP NOT NULL
            )
        """))
        logger.info("✓ Created food_logs table")
        
        # Create workout_logs table
        engine.execute(text("""
            CREATE TABLE IF NOT EXISTS workout_logs (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                name VARCHAR(255) NOT NULL,
                duration_minutes INTEGER NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                calories_burned INTEGER,
                notes TEXT,
                created_at TIMESTAMP NOT NULL
            )
        """))
        logger.info("✓ Created workout_logs table")
        
        # Create indexes
        logger.info("Creating indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_foods_name ON foods(name)",
            "CREATE INDEX IF NOT EXISTS idx_foods_category_id ON foods(category_id)",
            "CREATE INDEX IF NOT EXISTS idx_food_logs_user_id ON food_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_food_logs_consumed_at ON food_logs(consumed_at)",
            "CREATE INDEX IF NOT EXISTS idx_workout_logs_user_id ON workout_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_workout_logs_started_at ON workout_logs(started_at)"
        ]
        
        for index_sql in indexes:
            engine.execute(text(index_sql))
        
        logger.info("✓ Created all indexes")
        
        # Insert sample data
        logger.info("Inserting sample data...")
        
        # Insert sample foods
        sample_foods = [
            ("550e8400-e29b-41d4-a716-446655440001", "Chicken Breast", "brand_001", "protein", 100.0, "g", "1 serving", 165.0, 31.0, 0.0, 3.6, 0.0, 0.0, "user_input", False),
            ("550e8400-e29b-41d4-a716-446655440002", "Brown Rice", "brand_002", "grains", 100.0, "g", "1/2 cup", 111.0, 2.6, 23.0, 0.9, 1.8, 0.4, "user_input", False),
            ("550e8400-e29b-41d4-a716-446655440003", "Broccoli", "brand_003", "vegetables", 100.0, "g", "1 cup", 34.0, 2.8, 7.0, 0.4, 2.6, 1.5, "user_input", False),
            ("550e8400-e29b-41d4-a716-446655440004", "Salmon", "brand_004", "protein", 100.0, "g", "1 fillet", 208.0, 25.0, 0.0, 12.0, 0.0, 0.0, "user_input", False),
            ("550e8400-e29b-41d4-a716-446655440005", "Sweet Potato", "brand_005", "vegetables", 100.0, "g", "1 medium", 86.0, 1.6, 20.0, 0.1, 3.0, 4.2, "user_input", False)
        ]
        
        for food in sample_foods:
            engine.execute(text("""
                INSERT INTO foods (id, name, brand_id, category_id, serving_size, serving_unit, serving, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, source, verified, created_at, updated_at)
                VALUES (:id, :name, :brand_id, :category_id, :serving_size, :serving_unit, :serving, :calories, :protein_g, :carbs_g, :fat_g, :fiber_g, :sugar_g, :source, :verified, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": food[0], "name": food[1], "brand_id": food[2], "category_id": food[3],
                "serving_size": food[4], "serving_unit": food[5], "serving": food[6],
                "calories": food[7], "protein_g": food[8], "carbs_g": food[9], "fat_g": food[10],
                "fiber_g": food[11], "sugar_g": food[12], "source": food[13], "verified": food[14]
            })
        
        logger.info("✓ Inserted sample food data")
        
        # Insert sample user
        engine.execute(text("""
            INSERT INTO users (id, email, username, first_name, last_name, created_at, updated_at)
            VALUES (:id, :email, :username, :first_name, :last_name, NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """), {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "test@example.com",
            "username": "testuser",
            "first_name": "Test",
            "last_name": "User"
        })
        
        logger.info("✓ Inserted sample user")
        
        engine.dispose()
        logger.info("✓ Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Initialize the database."""
    success = init_database_sync()
    if success:
        logger.info("Database setup completed successfully!")
    else:
        logger.error("Database setup failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 