#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports."""
    try:
        print("Testing imports...")
        
        # Test shared imports
        from shared import FoodLogEntry, DataValidator
        print("✅ FoodLogEntry and DataValidator imported successfully")
        
        # Test model creation
        from shared.models import NutritionInfo, MealType
        from uuid import uuid4
        import datetime
        
        # Create a test FoodLogEntry
        nutrition_info = NutritionInfo(
            calories=100.0,
            protein_g=10.0,
            carbs_g=20.0,
            fat_g=5.0
        )
        
        food_log_entry = FoodLogEntry(
            user_id=uuid4(),
            food_item_id=uuid4(),
            quantity_g=100.0,
            meal_type=MealType.BREAKFAST,
            consumed_at=datetime.datetime.utcnow(),
            actual_nutrition=nutrition_info
        )
        
        print("✅ FoodLogEntry created successfully")
        
        # Test DataValidator
        result = DataValidator.validate_nutrition_data(nutrition_info.model_dump())
        print(f"✅ DataValidator.validate_nutrition_data returned: {result}")
        
        print("🎉 All imports and model creation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 