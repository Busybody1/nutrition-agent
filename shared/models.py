"""
Core Pydantic models for the Nutrition Agent.

This module defines the data models used by the nutrition agent service
for consistent data validation and serialization.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# Helper function for UTC timestamps
def utc_now():
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)

# SQLAlchemy ORM models for database schema
from sqlalchemy import Boolean, Column, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as SAUUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Index

Base = declarative_base()

# =============================================================================
# MULTI-USER SCALABILITY TABLES
# =============================================================================

class UserSessionORM(Base):
    """User session management for multi-user support across all agents."""
    __tablename__ = "user_sessions"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    agent_type = Column(String(50), nullable=False)  # 'supervisor', 'nutrition', 'workout', 'activity', 'vision'
    conversation_id = Column(SAUUID(as_uuid=True), nullable=False)
    agent_context = Column(JSONB, default={})
    last_activity = Column(DateTime, nullable=False, default=utc_now)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    
    # Relationships
    user = relationship("UserORM", back_populates="sessions")
    conversations = relationship("AgentConversationSessionORM", back_populates="session")

class AgentConversationSessionORM(Base):
    """Agent-specific conversation sessions for multi-user support."""
    __tablename__ = "agent_conversation_sessions"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(SAUUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=False)
    agent_type = Column(String(50), nullable=False)  # 'supervisor', 'nutrition', 'workout', 'activity', 'vision'
    conversation_type = Column(String(50), nullable=False)  # specific to each agent
    current_state = Column(JSONB, nullable=False, default={})
    conversation_history = Column(JSONB, default=[])
    agent_responses = Column(JSONB, default=[])
    context_data = Column(JSONB, default={})
    agent_specific_data = Column(JSONB, default={})  # nutrition goals, workout plans, activity tracking, vision analysis
    meta_data = Column(JSONB, default={})
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    last_message_at = Column(DateTime, nullable=False, default=utc_now)
    
    # Relationships
    user = relationship("UserORM", back_populates="agent_conversations")
    session = relationship("UserSessionORM", back_populates="conversations")
    messages = relationship("MessageHistoryORM", back_populates="conversation")

class InterAgentCommunicationORM(Base):
    """Cross-agent communication tracking for multi-user support."""
    __tablename__ = "inter_agent_communications"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    source_agent = Column(String(50), nullable=False)
    target_agent = Column(String(50), nullable=False)
    request_type = Column(String(100), nullable=False)
    request_data = Column(JSONB, default={})
    response_data = Column(JSONB, default={})
    status = Column(String(20), default="pending")  # 'pending', 'processing', 'completed', 'failed'
    created_at = Column(DateTime, nullable=False, default=utc_now)
    completed_at = Column(DateTime)
    processing_time_ms = Column(Integer)
    
    # Relationships
    user = relationship("UserORM", back_populates="inter_agent_communications")

class MessageHistoryORM(Base):
    """Message history persistence for multi-user support."""
    __tablename__ = "message_history"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    conversation_id = Column(SAUUID(as_uuid=True), ForeignKey("agent_conversation_sessions.id"), nullable=False)
    message_type = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    meta_data = Column(JSONB, default={})
    created_at = Column(DateTime, nullable=False, default=utc_now)
    
    # Relationships
    user = relationship("UserORM", back_populates="message_history")
    conversation = relationship("AgentConversationSessionORM", back_populates="messages")

class UserAgentPreferencesORM(Base):
    """User preferences for specific agents."""
    __tablename__ = "user_agent_preferences"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    preferences = Column(JSONB, default={})
    last_used = Column(DateTime, nullable=False, default=utc_now)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    
    # Relationships
    user = relationship("UserORM", back_populates="agent_preferences")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'agent_type', name='uq_user_agent_preferences'),
    )

# =============================================================================
# EXISTING NUTRITION-SPECIFIC MODELS
# =============================================================================

class UserORM(Base):
    """User model for the nutrition agent."""
    __tablename__ = "users"
    
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    
    # Relationships for multi-user support
    sessions = relationship("UserSessionORM", back_populates="user")
    agent_conversations = relationship("AgentConversationSessionORM", back_populates="user")
    inter_agent_communications = relationship("InterAgentCommunicationORM", back_populates="user")
    message_history = relationship("MessageHistoryORM", back_populates="user")
    agent_preferences = relationship("UserAgentPreferencesORM", back_populates="user")


class FoodItemORM(Base):
    __tablename__ = "food_items"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    brand = Column(String)
    barcode = Column(String)
    calories = Column(Float, nullable=False)
    protein_g = Column(Float, nullable=False)
    carbs_g = Column(Float, nullable=False)
    fat_g = Column(Float, nullable=False)
    fiber_g = Column(Float, default=0)
    sugar_g = Column(Float, default=0)
    sodium_mg = Column(Float, default=0)
    cholesterol_mg = Column(Float, default=0)
    vitamin_a_iu = Column(Float, default=0)
    vitamin_c_mg = Column(Float, default=0)
    vitamin_d_iu = Column(Float, default=0)
    calcium_mg = Column(Float, default=0)
    iron_mg = Column(Float, default=0)
    serving_size_g = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    subcategory = Column(String)
    tags = Column(String)
    source = Column(String, default="user_input")
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class FoodLogEntryORM(Base):
    __tablename__ = "food_log_entries"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    food_item_id = Column(
        SAUUID(as_uuid=True), ForeignKey("food_items.id"), nullable=False
    )
    quantity_g = Column(Float, nullable=False)
    meal_type = Column(String, nullable=False)
    consumed_at = Column(DateTime, nullable=False)
    calories = Column(Float, nullable=False)
    protein_g = Column(Float, nullable=False)
    carbs_g = Column(Float, nullable=False)
    fat_g = Column(Float, nullable=False)
    notes = Column(String)
    created_at = Column(DateTime, nullable=False)


class ExerciseORM(Base):
    __tablename__ = "exercises"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    type = Column(String, nullable=False)
    muscle_groups = Column(String)
    equipment = Column(String)
    difficulty = Column(String, nullable=False)
    instructions = Column(String)
    tips = Column(String)
    image_url = Column(String)
    video_url = Column(String)
    source = Column(String, default="database")
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class WorkoutLogORM(Base):
    __tablename__ = "workout_logs"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    calories_burned = Column(Integer)
    average_heart_rate = Column(Integer)
    max_heart_rate = Column(Integer)
    exercises = Column(String)
    notes = Column(String)
    rating = Column(Integer)
    created_at = Column(DateTime, nullable=False)


class ActivityLogORM(Base):
    __tablename__ = "activity_logs"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    steps = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)


class ActivityGoalORM(Base):
    __tablename__ = "activity_goals"
    id = Column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SAUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    goal = Column(Integer, nullable=False)  # daily step goal
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(
        DateTime, nullable=False, default=utc_now, onupdate=utc_now
    )


class UserRole(str, Enum):
    """User roles in the system."""

    USER = "user"
    ADMIN = "admin"
    COACH = "coach"


class Gender(str, Enum):
    """User gender options."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class ActivityLevel(str, Enum):
    """User activity level options."""

    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


class GoalType(str, Enum):
    """Fitness goal types."""

    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    MAINTENANCE = "maintenance"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"


class MealType(str, Enum):
    """Meal type classifications."""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class ExerciseType(str, Enum):
    """Exercise type classifications."""

    CARDIO = "cardio"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    BALANCE = "balance"
    SPORTS = "sports"


class DifficultyLevel(str, Enum):
    """Exercise difficulty levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class User(BaseModel):
    """User model for authentication and profile data."""

    id: UUID = Field(default_factory=uuid4, description="Unique user identifier")
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    first_name: str = Field(..., description="User's first name")
    last_name: str = Field(..., description="User's last name")
    role: UserRole = Field(default=UserRole.USER, description="User role")

    # Profile information
    age: Optional[int] = Field(None, ge=13, le=120, description="User age")
    gender: Optional[Gender] = Field(None, description="User gender")
    height_cm: Optional[float] = Field(None, gt=0, description="Height in centimeters")
    weight_kg: Optional[float] = Field(None, gt=0, description="Weight in kilograms")
    activity_level: Optional[ActivityLevel] = Field(None, description="Activity level")

    # Fitness goals
    primary_goal: Optional[GoalType] = Field(None, description="Primary fitness goal")
    target_weight_kg: Optional[float] = Field(None, gt=0, description="Target weight")
    daily_calorie_target: Optional[int] = Field(
        None, gt=0, description="Daily calorie target"
    )

    # Preferences
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions"
    )
    allergies: List[str] = Field(default_factory=list, description="Food allergies")
    preferred_exercises: List[str] = Field(
        default_factory=list, description="Preferred exercises"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=utc_now, description="Account creation time"
    )
    updated_at: datetime = Field(
        default_factory=utc_now, description="Last update time"
    )

    # Authentication
    is_active: bool = Field(default=True, description="Account active status")
    is_verified: bool = Field(default=False, description="Email verification status")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Validate email format."""
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be between 3 and 30 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v.lower()

    @field_serializer("created_at", "updated_at")
    def serialize_datetimes(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class NutritionInfo(BaseModel):
    """Nutritional information for food items."""

    calories: float = Field(..., ge=0, description="Calories per serving")
    protein_g: float = Field(..., ge=0, description="Protein in grams")
    carbs_g: float = Field(..., ge=0, description="Carbohydrates in grams")
    fat_g: float = Field(..., ge=0, description="Fat in grams")
    fiber_g: float = Field(default=0, ge=0, description="Fiber in grams")
    sugar_g: float = Field(default=0, ge=0, description="Sugar in grams")
    sodium_mg: float = Field(default=0, ge=0, description="Sodium in milligrams")
    cholesterol_mg: float = Field(
        default=0, ge=0, description="Cholesterol in milligrams"
    )

    # Vitamins and minerals
    vitamin_a_iu: float = Field(default=0, ge=0, description="Vitamin A in IU")
    vitamin_c_mg: float = Field(default=0, ge=0, description="Vitamin C in milligrams")
    vitamin_d_iu: float = Field(default=0, ge=0, description="Vitamin D in IU")
    calcium_mg: float = Field(default=0, ge=0, description="Calcium in milligrams")
    iron_mg: float = Field(default=0, ge=0, description="Iron in milligrams")


class FoodItem(BaseModel):
    """Food item model for nutrition tracking."""

    id: UUID = Field(default_factory=uuid4, description="Unique food item identifier")
    name: str = Field(..., description="Food item name")
    brand: Optional[str] = Field(None, description="Food brand")
    barcode: Optional[str] = Field(None, description="Product barcode")

    # Nutrition information
    nutrition_per_100g: NutritionInfo = Field(..., description="Nutrition per 100g")
    serving_size_g: float = Field(
        ..., gt=0, description="Standard serving size in grams"
    )

    # Categorization
    category: str = Field(..., description="Food category")
    subcategory: Optional[str] = Field(None, description="Food subcategory")
    tags: List[str] = Field(default_factory=list, description="Food tags")

    # Metadata
    source: str = Field(default="user_input", description="Data source")
    verified: bool = Field(default=False, description="Verified nutrition data")
    created_at: datetime = Field(
        default_factory=utc_now, description="Creation time"
    )
    updated_at: datetime = Field(
        default_factory=utc_now, description="Last update time"
    )

    @field_serializer("created_at", "updated_at")
    def serialize_datetimes(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class FoodLogEntry(BaseModel):
    """Food logging entry for user's daily intake."""

    id: UUID = Field(default_factory=uuid4, description="Unique log entry identifier")
    user_id: UUID = Field(..., description="User identifier")
    food_item_id: UUID = Field(..., description="Food item identifier")

    # Consumption details
    quantity_g: float = Field(..., gt=0, description="Quantity consumed in grams")
    meal_type: MealType = Field(..., description="Meal type")
    consumed_at: datetime = Field(..., description="When the food was consumed")

    # Calculated nutrition
    actual_nutrition: NutritionInfo = Field(
        ..., description="Actual nutrition consumed"
    )

    # Metadata
    notes: Optional[str] = Field(None, description="User notes")
    created_at: datetime = Field(
        default_factory=utc_now, description="Log creation time"
    )

    @field_serializer("created_at", "consumed_at")
    def serialize_datetimes(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class Exercise(BaseModel):
    """Exercise model for workout tracking."""

    id: UUID = Field(default_factory=uuid4, description="Unique exercise identifier")
    name: str = Field(..., description="Exercise name")
    description: Optional[str] = Field(None, description="Exercise description")

    # Categorization
    type: ExerciseType = Field(..., description="Exercise type")
    muscle_groups: List[str] = Field(
        default_factory=list, description="Target muscle groups"
    )
    equipment: List[str] = Field(default_factory=list, description="Required equipment")

    # Difficulty and instructions
    difficulty: DifficultyLevel = Field(..., description="Exercise difficulty level")
    instructions: List[str] = Field(
        default_factory=list, description="Exercise instructions"
    )
    tips: List[str] = Field(default_factory=list, description="Exercise tips")

    # Media
    image_url: Optional[str] = Field(None, description="Exercise image URL")
    video_url: Optional[str] = Field(None, description="Exercise video URL")

    # Metadata
    source: str = Field(default="database", description="Data source")
    verified: bool = Field(default=False, description="Verified exercise data")
    created_at: datetime = Field(
        default_factory=utc_now, description="Creation time"
    )
    updated_at: datetime = Field(
        default_factory=utc_now, description="Last update time"
    )

    @field_serializer("created_at", "updated_at")
    def serialize_datetimes(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class WorkoutLog(BaseModel):
    """Workout logging entry for user's exercise sessions."""

    id: UUID = Field(default_factory=uuid4, description="Unique workout log identifier")
    user_id: UUID = Field(..., description="User identifier")

    # Workout details
    name: str = Field(..., description="Workout name")
    type: ExerciseType = Field(..., description="Workout type")
    duration_minutes: int = Field(..., gt=0, description="Workout duration in minutes")

    # Timing
    started_at: datetime = Field(..., description="Workout start time")
    completed_at: Optional[datetime] = Field(
        None, description="Workout completion time"
    )

    # Performance metrics
    calories_burned: Optional[int] = Field(None, ge=0, description="Calories burned")
    average_heart_rate: Optional[int] = Field(
        None, ge=0, description="Average heart rate"
    )
    max_heart_rate: Optional[int] = Field(None, ge=0, description="Maximum heart rate")

    # Exercises performed
    exercises: List[Dict[str, Any]] = Field(
        default_factory=list, description="Exercises performed"
    )

    # Metadata
    notes: Optional[str] = Field(None, description="User notes")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Workout rating (1-5)")
    created_at: datetime = Field(
        default_factory=utc_now, description="Log creation time"
    )

    @field_serializer("created_at", "started_at", "completed_at")
    def serialize_datetimes(self, v):
        return v.isoformat() if v else None

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class AgentRequest(BaseModel):
    """Request model for inter-agent communication."""

    id: UUID = Field(default_factory=uuid4, description="Unique request identifier")
    user_id: UUID = Field(..., description="User identifier")
    agent_type: str = Field(..., description="Target agent type")

    # Request details
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )

    # Context
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")

    # Timing
    timestamp: datetime = Field(
        default_factory=utc_now, description="Request timestamp"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")

    @field_serializer("timestamp")
    def serialize_timestamp(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class AgentResponse(BaseModel):
    """Response model for inter-agent communication."""

    id: UUID = Field(default_factory=uuid4, description="Unique response identifier")
    request_id: UUID = Field(..., description="Original request identifier")
    agent_type: str = Field(..., description="Responding agent type")

    # Response details
    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Metadata
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=utc_now, description="Response timestamp"
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )


class ConversationMessage(BaseModel):
    """Message model for conversation history."""

    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: UUID = Field(..., description="User identifier")

    # Message content
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")

    # Metadata
    timestamp: datetime = Field(
        default_factory=utc_now, description="Message timestamp"
    )
    agent_responses: List[AgentResponse] = Field(
        default_factory=list, description="Agent responses"
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v):
        return v.isoformat()

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}
    )
