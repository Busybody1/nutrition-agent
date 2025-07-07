#!/usr/bin/env python3
"""Test script to debug database URL configuration."""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_db_urls():
    """Test database URL configuration."""
    try:
        print("Testing database URL configuration...")
        
        # Test settings
        from shared.config import get_settings
        settings = get_settings()
        
        print(f"Environment variables:")
        print(f"  DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
        print(f"  NUTRITION_DB_URI: {os.getenv('NUTRITION_DB_URI', 'Not set')}")
        print(f"  WORKOUT_DB_URI: {os.getenv('WORKOUT_DB_URI', 'Not set')}")
        
        print(f"\nSettings:")
        print(f"  settings.database.url: {settings.database.url}")
        print(f"  settings.multi_db.nutrition_db_uri: {settings.multi_db.nutrition_db_uri}")
        print(f"  settings.multi_db.workout_db_uri: {settings.multi_db.workout_db_uri}")
        
        # Test database engines
        from shared.database import get_nutrition_db_engine, get_fitness_db_engine
        
        nutrition_engine = get_nutrition_db_engine()
        fitness_engine = get_fitness_db_engine()
        
        print(f"\nDatabase engines:")
        print(f"  Nutrition engine URL: {nutrition_engine.url}")
        print(f"  Fitness engine URL: {fitness_engine.url}")
        
        print("🎉 Database URL configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database URL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_db_urls()
    sys.exit(0 if success else 1) 