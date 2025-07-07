"""
Shared library for the AI Agent Framework.

This module provides common utilities, database connections, and configuration
that are shared across all agents in the framework.
"""

from .config import get_settings
from .database import (
    get_async_session,
    get_database,
    get_fitness_db_engine,
    get_nutrition_db_engine,
    get_redis,
    get_sync_session,
    get_workout_db_engine,
    health_check,
    init_database,
)
from .models import (
    ActivityLog,
    ConversationState,
    FoodItem,
    FoodLog,
    FoodLogEntry,
    User,
    WorkoutLog,
)
from .utils import (
    CacheManager,
    DataValidator,
    DateTimeUtils,
    ErrorHandler,
    PerformanceMonitor,
    ResponseBuilder,
    cache_result,
    setup_logging,
)

__all__ = [
    # Configuration
    "get_settings",
    
    # Database
    "get_database",
    "get_async_session",
    "get_sync_session",
    "get_redis",
    "get_nutrition_db_engine",
    "get_workout_db_engine",
    "get_fitness_db_engine",
    "health_check",
    "init_database",
    
    # Models
    "User",
    "FoodItem",
    "FoodLog",
    "FoodLogEntry",
    "WorkoutLog",
    "ActivityLog",
    "ConversationState",
    
    # Utilities
    "setup_logging",
    "CacheManager",
    "cache_result",
    "ErrorHandler",
    "PerformanceMonitor",
    "DataValidator",
    "DateTimeUtils",
    "ResponseBuilder",
]

__version__ = "0.1.0"
