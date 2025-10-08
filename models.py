"""
SQLAlchemy models for Nutrition Agent based on user-fitness-app schema.
"""

from sqlalchemy import Column, String, Text, Integer, Numeric, DateTime, Boolean, ForeignKey, Date, Time
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from utils.database import Base
import uuid


class User(Base):
    """User model - mirrors user-fitness-app-backend users table."""
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    date_of_birth = Column(Date)
    gender = Column(String)
    height_cm = Column(Numeric)
    profile_picture_url = Column(Text)
    timezone = Column(String)
    notification_preferences = Column(JSON)
    account_status = Column(String, default="active")
    last_login = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User(user_id={self.user_id}, email={self.email})>"

    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return "Unknown User"

    @property
    def is_active(self) -> bool:
        """Check if user account is active"""
        return self.account_status == "active"


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
    # Enhanced micronutrient targets
    vitamin_a_mcg = Column(Integer)
    vitamin_c_mg = Column(Integer)
    vitamin_d_mcg = Column(Integer)
    vitamin_e_mg = Column(Integer)
    vitamin_k_mcg = Column(Integer)
    thiamin_mg = Column(Numeric)
    riboflavin_mg = Column(Numeric)
    niacin_mg = Column(Numeric)
    vitamin_b6_mg = Column(Numeric)
    folate_mcg = Column(Integer)
    vitamin_b12_mcg = Column(Numeric)
    pantothenic_acid_mg = Column(Numeric)
    biotin_mcg = Column(Numeric)
    choline_mg = Column(Numeric)
    calcium_mg = Column(Integer)
    iron_mg = Column(Numeric)
    magnesium_mg = Column(Integer)
    phosphorus_mg = Column(Integer)
    potassium_mg = Column(Integer)
    zinc_mg = Column(Numeric)
    selenium_mcg = Column(Integer)
    copper_mg = Column(Numeric)
    manganese_mg = Column(Numeric)
    chromium_mcg = Column(Numeric)
    molybdenum_mcg = Column(Numeric)
    iodine_mcg = Column(Integer)
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
    # Enhanced micronutrient totals
    total_vitamin_a_mcg = Column(Integer, default=0)
    total_vitamin_c_mg = Column(Integer, default=0)
    total_vitamin_d_mcg = Column(Integer, default=0)
    total_vitamin_e_mg = Column(Integer, default=0)
    total_vitamin_k_mcg = Column(Integer, default=0)
    total_thiamin_mg = Column(Numeric, default=0)
    total_riboflavin_mg = Column(Numeric, default=0)
    total_niacin_mg = Column(Numeric, default=0)
    total_vitamin_b6_mg = Column(Numeric, default=0)
    total_folate_mcg = Column(Integer, default=0)
    total_vitamin_b12_mcg = Column(Numeric, default=0)
    total_pantothenic_acid_mg = Column(Numeric, default=0)
    total_biotin_mcg = Column(Numeric, default=0)
    total_choline_mg = Column(Numeric, default=0)
    total_calcium_mg = Column(Integer, default=0)
    total_iron_mg = Column(Numeric, default=0)
    total_magnesium_mg = Column(Integer, default=0)
    total_phosphorus_mg = Column(Integer, default=0)
    total_potassium_mg = Column(Integer, default=0)
    total_zinc_mg = Column(Numeric, default=0)
    total_selenium_mcg = Column(Integer, default=0)
    total_copper_mg = Column(Numeric, default=0)
    total_manganese_mg = Column(Numeric, default=0)
    total_chromium_mcg = Column(Numeric, default=0)
    total_molybdenum_mcg = Column(Numeric, default=0)
    total_iodine_mcg = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
