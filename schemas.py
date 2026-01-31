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
    # Enhanced micronutrient targets
    vitamin_a_mcg: Optional[int] = Field(None, ge=0)
    vitamin_c_mg: Optional[int] = Field(None, ge=0)
    vitamin_d_mcg: Optional[int] = Field(None, ge=0)
    vitamin_e_mg: Optional[int] = Field(None, ge=0)
    vitamin_k_mcg: Optional[int] = Field(None, ge=0)
    thiamin_mg: Optional[float] = Field(None, ge=0)
    riboflavin_mg: Optional[float] = Field(None, ge=0)
    niacin_mg: Optional[float] = Field(None, ge=0)
    vitamin_b6_mg: Optional[float] = Field(None, ge=0)
    folate_mcg: Optional[int] = Field(None, ge=0)
    vitamin_b12_mcg: Optional[float] = Field(None, ge=0)
    pantothenic_acid_mg: Optional[float] = Field(None, ge=0)
    biotin_mcg: Optional[float] = Field(None, ge=0)
    choline_mg: Optional[float] = Field(None, ge=0)
    calcium_mg: Optional[int] = Field(None, ge=0)
    iron_mg: Optional[float] = Field(None, ge=0)
    magnesium_mg: Optional[int] = Field(None, ge=0)
    phosphorus_mg: Optional[int] = Field(None, ge=0)
    potassium_mg: Optional[int] = Field(None, ge=0)
    zinc_mg: Optional[float] = Field(None, ge=0)
    selenium_mcg: Optional[int] = Field(None, ge=0)
    copper_mg: Optional[float] = Field(None, ge=0)
    manganese_mg: Optional[float] = Field(None, ge=0)
    chromium_mcg: Optional[float] = Field(None, ge=0)
    molybdenum_mcg: Optional[float] = Field(None, ge=0)
    iodine_mcg: Optional[int] = Field(None, ge=0)


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
    # Enhanced micronutrient targets
    vitamin_a_mcg: Optional[int] = Field(None, ge=0)
    vitamin_c_mg: Optional[int] = Field(None, ge=0)
    vitamin_d_mcg: Optional[int] = Field(None, ge=0)
    vitamin_e_mg: Optional[int] = Field(None, ge=0)
    vitamin_k_mcg: Optional[int] = Field(None, ge=0)
    thiamin_mg: Optional[float] = Field(None, ge=0)
    riboflavin_mg: Optional[float] = Field(None, ge=0)
    niacin_mg: Optional[float] = Field(None, ge=0)
    vitamin_b6_mg: Optional[float] = Field(None, ge=0)
    folate_mcg: Optional[int] = Field(None, ge=0)
    vitamin_b12_mcg: Optional[float] = Field(None, ge=0)
    pantothenic_acid_mg: Optional[float] = Field(None, ge=0)
    biotin_mcg: Optional[float] = Field(None, ge=0)
    choline_mg: Optional[float] = Field(None, ge=0)
    calcium_mg: Optional[int] = Field(None, ge=0)
    iron_mg: Optional[float] = Field(None, ge=0)
    magnesium_mg: Optional[int] = Field(None, ge=0)
    phosphorus_mg: Optional[int] = Field(None, ge=0)
    potassium_mg: Optional[int] = Field(None, ge=0)
    zinc_mg: Optional[float] = Field(None, ge=0)
    selenium_mcg: Optional[int] = Field(None, ge=0)
    copper_mg: Optional[float] = Field(None, ge=0)
    manganese_mg: Optional[float] = Field(None, ge=0)
    chromium_mcg: Optional[float] = Field(None, ge=0)
    molybdenum_mcg: Optional[float] = Field(None, ge=0)
    iodine_mcg: Optional[int] = Field(None, ge=0)


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
    # Enhanced micronutrient totals
    total_vitamin_a_mcg: int = 0
    total_vitamin_c_mg: int = 0
    total_vitamin_d_mcg: int = 0
    total_vitamin_e_mg: int = 0
    total_vitamin_k_mcg: int = 0
    total_thiamin_mg: float = 0
    total_riboflavin_mg: float = 0
    total_niacin_mg: float = 0
    total_vitamin_b6_mg: float = 0
    total_folate_mcg: int = 0
    total_vitamin_b12_mcg: float = 0
    total_pantothenic_acid_mg: float = 0
    total_biotin_mcg: float = 0
    total_choline_mg: float = 0
    total_calcium_mg: int = 0
    total_iron_mg: float = 0
    total_magnesium_mg: int = 0
    total_phosphorus_mg: int = 0
    total_potassium_mg: int = 0
    total_zinc_mg: float = 0
    total_selenium_mcg: int = 0
    total_copper_mg: float = 0
    total_manganese_mg: float = 0
    total_chromium_mcg: float = 0
    total_molybdenum_mcg: float = 0
    total_iodine_mcg: int = 0
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


# New Enhanced AI Response Schemas
class ServingInfo(BaseModel):
    """Serving information for meals and recipes."""
    serving_size: str
    quantity: str
    portion_description: str


class NutrientDetail(BaseModel):
    """Detailed nutrient information with daily value and importance."""
    nutrient: str
    amount: Optional[float] = None
    unit: str
    daily_value_percent: Optional[str] = None
    importance: str
    daily_target: Optional[str] = None
    food_sources: Optional[str] = None
    recommendation: Optional[str] = None


class MacrosInfo(BaseModel):
    """Macronutrient information."""
    protein: float
    carbs: float
    fat: float


class EnhancedNutritionInfo(BaseModel):
    """Enhanced nutrition information with expanded nutrients."""
    calories: int
    macros: MacrosInfo
    nutrients_summary: List[NutrientDetail]


class MealAnalysis(BaseModel):
    """AI-generated meal analysis with enhanced nutrition."""
    meal_name: str
    serving_info: ServingInfo
    estimated_nutrition: EnhancedNutritionInfo
    key_nutrients: List[str]
    health_assessment: Dict[str, Any]
    satisfaction_analysis: Dict[str, Any]
    timing_insights: str
    mood_connection: str
    recommendations: List[str]
    balance_suggestions: str


class MealPlanMealEnhanced(BaseModel):
    """Enhanced meal plan meal with serving info and nutrients."""
    name: str
    serving_info: ServingInfo
    calories: int
    macros: MacrosInfo
    nutrients_summary: List[NutrientDetail]
    ingredients: List[str]


class MealPlanDayEnhanced(BaseModel):
    """Enhanced meal plan day with improved meal structure."""
    day: int
    meals: Dict[str, MealPlanMealEnhanced]
    total_calories: int


class EnhancedMealPlan(BaseModel):
    """Enhanced meal plan with improved structure."""
    days: List[MealPlanDayEnhanced]


class RecipeEnhanced(BaseModel):
    """Enhanced recipe with serving info and nutrients."""
    name: str
    cuisine: str
    prep_time: str
    cook_time: str
    total_time: str
    servings: int
    serving_info: ServingInfo
    difficulty: str
    nutrition_per_serving: EnhancedNutritionInfo
    ingredients: List[Dict[str, str]]
    instructions: List[str]
    tips: List[str]
    variations: List[str]
    storage: str
    dietary_notes: str


class NutritionSummary(BaseModel):
    """Enhanced nutrition summary with serving guidelines."""
    period_days: int
    user_goals: str
    summary_analysis: str
    serving_guidelines: Dict[str, str]
    nutrient_analysis: Dict[str, Any]
    goal_assessment: str
    personalized_recommendations: List[str]
    meal_planning_suggestions: str
    next_steps: str
    generated_at: str


class NutritionResponse(BaseModel):
    """Enhanced nutrition response with structured guidance."""
    user_question: str
    expert_analysis: str
    practical_advice: str
    serving_guidelines: Dict[str, str]
    nutrient_focus: Dict[str, Any]
    motivation: str
    next_steps: List[str]
    additional_resources: str
    generated_at: str


class MealPlanRequest(BaseModel):
    description: Optional[str] = None
    plan_type: Optional[str] = "weekly"
    days_per_week: Optional[int] = Field(5, ge=1, le=7)
    meals_per_day: Optional[int] = Field(3, ge=1, le=5)
    dietary_restrictions: Optional[List[str]] = []
    calorie_target: Optional[int] = 0
    cuisine_preference: Optional[str] = "any"
    cooking_time: Optional[str] = "medium"
    skill_level: Optional[str] = "intermediate"
    budget: Optional[str] = "medium"


class CreateMealRequest(BaseModel):
    """Request model for creating a single meal."""
    description: Optional[str] = None
    meal_type: Optional[str] = "dinner"
    dietary_restrictions: Optional[List[str]] = []
    calorie_target: Optional[int] = 0
    cuisine_preference: Optional[str] = "any"
    cooking_time: Optional[str] = "medium"
    skill_level: Optional[str] = "intermediate"
    budget: Optional[str] = "medium"
