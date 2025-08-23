"""
Pydantic schemas for Nutrition Agent API based on user-fitness-app schema.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time
from uuid import UUID


# Meal Type Schemas
class MealTypeBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    display_order: int = 0


class MealTypeCreate(MealTypeBase):
    pass


class MealTypeResponse(MealTypeBase):
    meal_type_id: UUID
    created_at: datetime
    
    model_config = {"from_attributes": True}


# Meal Plan Schemas
class MealPlanItemBase(BaseModel):
    food_id: Optional[str] = None
    recipe_id: Optional[UUID] = None
    quantity: Optional[float] = Field(None, gt=0)
    servings: int = 1
    notes: Optional[str] = None


class MealPlanItemCreate(MealPlanItemBase):
    pass


class MealPlanItemResponse(MealPlanItemBase):
    item_id: UUID
    meal_id: UUID
    created_at: datetime
    
    model_config = {"from_attributes": True}


class MealPlanMealBase(BaseModel):
    meal_type_id: UUID
    time_of_day: Optional[time] = None
    notes: Optional[str] = None


class MealPlanMealCreate(MealPlanMealBase):
    items: Optional[List[MealPlanItemCreate]] = []


class MealPlanMealResponse(MealPlanMealBase):
    meal_id: UUID
    day_id: UUID
    created_at: datetime
    items: List[MealPlanItemResponse] = []
    
    model_config = {"from_attributes": True}


class MealPlanDayBase(BaseModel):
    day_number: int = Field(..., ge=1)
    date: date
    notes: Optional[str] = None


class MealPlanDayCreate(MealPlanDayBase):
    meals: Optional[List[MealPlanMealCreate]] = []


class MealPlanDayResponse(MealPlanDayBase):
    day_id: UUID
    meal_plan_id: UUID
    created_at: datetime
    meals: List[MealPlanMealResponse] = []
    
    model_config = {"from_attributes": True}


class MealPlanBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    start_date: date
    end_date: date
    is_active: bool = True
    is_template: bool = False


class MealPlanCreate(MealPlanBase):
    days: Optional[List[MealPlanDayCreate]] = []


class MealPlanUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_active: Optional[bool] = None
    is_template: Optional[bool] = None


class MealPlanResponse(MealPlanBase):
    meal_plan_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    days: List[MealPlanDayResponse] = []
    
    model_config = {"from_attributes": True}


# Food Log Schemas
class UserFoodLogItemBase(BaseModel):
    food_id: Optional[str] = None
    recipe_id: Optional[UUID] = None
    quantity: float = Field(..., gt=0)
    serving: int = 1
    calories: Optional[int] = Field(None, ge=0)
    protein_g: Optional[float] = Field(None, ge=0)
    carbs_g: Optional[float] = Field(None, ge=0)
    fat_g: Optional[float] = Field(None, ge=0)
    photo_url: Optional[str] = None
    notes: Optional[str] = None


class UserFoodLogItemCreate(UserFoodLogItemBase):
    pass


class UserFoodLogItemResponse(UserFoodLogItemBase):
    item_id: UUID
    log_id: UUID
    ai_verified: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}


class UserFoodLogBase(BaseModel):
    date: date
    meal_type_id: UUID
    time_of_day: Optional[time] = None


class UserFoodLogCreate(UserFoodLogBase):
    items: Optional[List[UserFoodLogItemCreate]] = []


class UserFoodLogResponse(UserFoodLogBase):
    log_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    items: List[UserFoodLogItemResponse] = []
    
    model_config = {"from_attributes": True}


# Nutrition Target Schemas
class UserNutritionTargetBase(BaseModel):
    date: date
    calories: Optional[int] = Field(None, ge=0)
    protein_g: Optional[float] = Field(None, ge=0)
    carbs_g: Optional[float] = Field(None, ge=0)
    fat_g: Optional[float] = Field(None, ge=0)
    fiber_g: Optional[float] = Field(None, ge=0)
    sugar_g: Optional[float] = Field(None, ge=0)
    sodium_mg: Optional[int] = Field(None, ge=0)
    water_ml: Optional[int] = Field(None, ge=0)


class UserNutritionTargetCreate(UserNutritionTargetBase):
    pass


class UserNutritionTargetUpdate(BaseModel):
    calories: Optional[int] = Field(None, ge=0)
    protein_g: Optional[float] = Field(None, ge=0)
    carbs_g: Optional[float] = Field(None, ge=0)
    fat_g: Optional[float] = Field(None, ge=0)
    fiber_g: Optional[float] = Field(None, ge=0)
    sugar_g: Optional[float] = Field(None, ge=0)
    sodium_mg: Optional[int] = Field(None, ge=0)
    water_ml: Optional[int] = Field(None, ge=0)


class UserNutritionTargetResponse(UserNutritionTargetBase):
    target_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


# Nutrition Summary Schemas
class UserNutritionSummaryResponse(BaseModel):
    summary_id: UUID
    user_id: UUID
    date: date
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sugar_g: float
    total_sodium_mg: int
    total_water_ml: int
    target_calories: Optional[int] = None
    target_protein_g: Optional[float] = None
    target_carbs_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    calories_remaining: int
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


# Nutrition Stats Schemas
class NutritionStats(BaseModel):
    date: date
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    calories_remaining: int
    target_met_percentage: float
    
    model_config = {"from_attributes": True}


class WeeklyNutritionSummary(BaseModel):
    week_start: date
    week_end: date
    daily_stats: List[NutritionStats]
    weekly_averages: Dict[str, float]
    weekly_totals: Dict[str, float]


# AI Response Schemas
class AIAnalysisRequest(BaseModel):
    description: Optional[str] = None
    food_items: Optional[str] = None
    meal_type: Optional[str] = None
    portion_size: Optional[str] = None
    eating_time: Optional[str] = None
    location: Optional[str] = None
    mood_before: Optional[str] = None
    mood_after: Optional[str] = None
    hunger_level: Optional[str] = None
    satisfaction_level: Optional[str] = None
    estimated_calories: Optional[int] = None
    notes: Optional[str] = None


class MealPlanRequest(BaseModel):
    description: Optional[str] = None
    plan_type: Optional[str] = "single_meal"
    meal_type: Optional[str] = "dinner"
    dietary_restrictions: Optional[List[str]] = []
    calorie_target: Optional[int] = 0
    cuisine_preference: Optional[str] = "any"
    cooking_time: Optional[str] = "medium"
    skill_level: Optional[str] = "intermediate"
    budget: Optional[str] = "medium"
