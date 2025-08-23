"""
SQLAlchemy models for Nutrition Agent based on user-fitness-app schema.
"""

from sqlalchemy import Column, String, Text, Integer, Numeric, DateTime, Boolean, ForeignKey, Date, Time
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from utils.database import Base
import uuid


class MealType(Base):
    __tablename__ = "meal_types"
    
    meal_type_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    display_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MealPlan(Base):
    __tablename__ = "meal_plans"
    
    meal_plan_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True)
    is_template = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MealPlanDay(Base):
    __tablename__ = "meal_plan_days"
    
    day_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meal_plan_id = Column(UUID(as_uuid=True), ForeignKey("meal_plans.meal_plan_id"), nullable=False)
    day_number = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MealPlanMeal(Base):
    __tablename__ = "meal_plan_meals"
    
    meal_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    day_id = Column(UUID(as_uuid=True), ForeignKey("meal_plan_days.day_id"), nullable=False)
    meal_type_id = Column(UUID(as_uuid=True), ForeignKey("meal_types.meal_type_id"), nullable=False)
    time_of_day = Column(Time)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MealPlanItem(Base):
    __tablename__ = "meal_plan_items"
    
    item_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meal_id = Column(UUID(as_uuid=True), ForeignKey("meal_plan_meals.meal_id"), nullable=False)
    food_id = Column(String)  # External food database ID
    recipe_id = Column(UUID(as_uuid=True))  # Simplified for now
    quantity = Column(Numeric)
    servings = Column(Integer, default=1)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserFoodLog(Base):
    __tablename__ = "user_food_logs"
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    meal_type_id = Column(UUID(as_uuid=True), ForeignKey("meal_types.meal_type_id"), nullable=False)
    time_of_day = Column(Time)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserFoodLogItem(Base):
    __tablename__ = "user_food_log_items"
    
    item_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_id = Column(UUID(as_uuid=True), ForeignKey("user_food_logs.log_id"), nullable=False)
    food_id = Column(String)  # External food database ID
    recipe_id = Column(UUID(as_uuid=True))  # Simplified for now
    quantity = Column(Numeric, nullable=False)
    serving = Column(Integer, default=1)
    calories = Column(Integer)
    protein_g = Column(Numeric)
    carbs_g = Column(Numeric)
    fat_g = Column(Numeric)
    photo_url = Column(Text)
    ai_verified = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserNutritionTarget(Base):
    __tablename__ = "user_nutrition_targets"
    
    target_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    calories = Column(Integer)
    protein_g = Column(Numeric)
    carbs_g = Column(Numeric)
    fat_g = Column(Numeric)
    fiber_g = Column(Numeric)
    sugar_g = Column(Numeric)
    sodium_mg = Column(Integer)
    water_ml = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserNutritionSummary(Base):
    __tablename__ = "user_nutrition_summaries"
    
    summary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    total_calories = Column(Integer, default=0)
    total_protein_g = Column(Numeric, default=0)
    total_carbs_g = Column(Numeric, default=0)
    total_fat_g = Column(Numeric, default=0)
    total_fiber_g = Column(Numeric, default=0)
    total_sugar_g = Column(Numeric, default=0)
    total_sodium_mg = Column(Integer, default=0)
    total_water_ml = Column(Integer, default=0)
    target_calories = Column(Integer)
    target_protein_g = Column(Numeric)
    target_carbs_g = Column(Numeric)
    target_fat_g = Column(Numeric)
    calories_remaining = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
