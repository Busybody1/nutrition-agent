"""
Configuration settings for the Nutrition Agent.

This module provides centralized access to environment variables and configuration.
"""

import os
from typing import Optional, List

def get_database_url() -> str:
    """Get main database URL (DATABASE_URL)."""
    return os.getenv("DATABASE_URL", "")

def get_user_database_uri() -> str:
    """Get user database URI (USER_DATABASE_URI)."""
    return os.getenv("USER_DATABASE_URI", "")

def get_nutrition_db_uri() -> Optional[str]:
    """Get nutrition database URI (NUTRITION_DB_URI) - Optional."""
    return os.getenv("NUTRITION_DB_URI")

def get_workout_db_uri() -> Optional[str]:
    """Get workout database URI (WORKOUT_DB_URI) - Optional."""
    return os.getenv("WORKOUT_DB_URI")

def get_redis_url() -> Optional[str]:
    """Get Redis URL (REDIS_URL) - Optional."""
    return os.getenv("REDIS_URL")

def get_groq_api_key() -> str:
    """Get Groq API key."""
    return os.getenv("GROQ_API_KEY", "")

def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return os.getenv("OPENAI_API_KEY", "")

def get_environment() -> str:
    """Get environment (development/production)."""
    return os.getenv("ENVIRONMENT", "development")

def get_log_level() -> str:
    """Get log level."""
    return os.getenv("LOG_LEVEL", "INFO")

def get_port() -> int:
    """Get server port."""
    return int(os.getenv("PORT", "8006"))

def get_host() -> str:
    """Get server host."""
    return os.getenv("HOST", "0.0.0.0")

def get_cors_origins() -> List[str]:
    """Get CORS allowed origins."""
    origins = os.getenv("CORS_ORIGINS", "*")
    if origins == "*":
        return ["*"]
    return [origin.strip() for origin in origins.split(",")]

# Groq AI Configuration
def get_groq_model() -> str:
    """Get Groq model to use."""
    return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def get_groq_timeout() -> int:
    """Get Groq API timeout in seconds."""
    return int(os.getenv("GROQ_TIMEOUT", "30"))

# OpenAI AI Configuration
def get_openai_model() -> str:
    """Get OpenAI model to use."""
    return os.getenv("OPENAI_MODEL", "gpt-4o")

def get_openai_timeout() -> int:
    """Get OpenAI API timeout in seconds."""
    return int(os.getenv("OPENAI_TIMEOUT", "60"))

def get_openai_max_tokens() -> int:
    """Get OpenAI max tokens for responses."""
    return int(os.getenv("OPENAI_MAX_TOKENS", "8192"))

# Database Configuration
def get_database_pool_size() -> int:
    """Get database connection pool size."""
    return int(os.getenv("DATABASE_POOL_SIZE", "10"))

def get_database_max_overflow() -> int:
    """Get database max overflow connections."""
    return int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))

def get_database_connect_timeout() -> int:
    """Get database connection timeout in seconds."""
    return int(os.getenv("DATABASE_CONNECT_TIMEOUT", "10"))

# Redis Configuration
def get_redis_ttl_short() -> int:
    """Get Redis TTL for short-term cache (seconds)."""
    return int(os.getenv("REDIS_TTL_SHORT", "300"))  # 5 minutes

def get_redis_ttl_medium() -> int:
    """Get Redis TTL for medium-term cache (seconds)."""
    return int(os.getenv("REDIS_TTL_MEDIUM", "3600"))  # 1 hour

def get_redis_ttl_long() -> int:
    """Get Redis TTL for long-term cache (seconds)."""
    return int(os.getenv("REDIS_TTL_LONG", "86400"))  # 24 hours
