"""
Utilities package for Nutrition Agent.

This package contains utilities for database connections and configuration management.
"""

# Import key functions for easy access
from .database.connections import (
    get_main_db, get_user_db, get_nutrition_db, get_workout_db,
    test_database_connection, get_database_status, Base
)
from .config.settings import (
    get_database_url, get_user_database_uri, get_nutrition_db_uri, get_workout_db_uri,
    get_redis_url, get_groq_api_key, get_environment, get_log_level,
    get_port, get_host, get_cors_origins, get_groq_model, get_groq_timeout,
    get_database_pool_size, get_database_max_overflow, get_database_connect_timeout,
    get_redis_ttl_short, get_redis_ttl_medium, get_redis_ttl_long
)

__all__ = [
    # Database functions
    "get_main_db", "get_user_db", "get_nutrition_db", "get_workout_db",
    "test_database_connection", "get_database_status", "Base",
    
    # Configuration functions
    "get_database_url", "get_user_database_uri", "get_nutrition_db_uri", "get_workout_db_uri",
    "get_redis_url", "get_groq_api_key", "get_environment", "get_log_level",
    "get_port", "get_host", "get_cors_origins", "get_groq_model", "get_groq_timeout",
    "get_database_pool_size", "get_database_max_overflow", "get_database_connect_timeout",
    "get_redis_ttl_short", "get_redis_ttl_medium", "get_redis_ttl_long"
]
