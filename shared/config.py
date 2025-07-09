"""
Configuration management for the AI Agent Framework.

This module provides centralized configuration management using Pydantic settings
with environment variable support.
"""

import logging
import os
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Set up logger - reduced for faster startup
logger = logging.getLogger(__name__)


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    database: str = Field(default="fitness_ai_framework", alias="POSTGRES_DB")
    username: str = Field(default="fitness_user", alias="POSTGRES_USER")
    password: str = Field(default="fitness_password", alias="POSTGRES_PASSWORD")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Connection pooling
    pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")

    def __init__(self, **data):
        super().__init__(**data)
        # Reduced logging for faster startup
        # logger.info(f"DatabaseSettings initialized with database_url: {self.database_url}")
        # logger.info(f"DatabaseSettings host: {self.host}, port: {self.port}, database: {self.database}")

    @property
    def url(self) -> str:
        """Get database URL for SQLAlchemy."""
        # If DATABASE_URL is provided, use it directly
        if self.database_url:
            # Reduced logging for faster startup
            # logger.info(f"Using DATABASE_URL: {self.database_url}")
            # Handle both postgres:// and postgresql:// URLs
            if self.database_url.startswith("postgres://"):
                converted_url = self.database_url.replace("postgres://", "postgresql://", 1)
                # logger.info(f"Converted postgres:// to postgresql://: {converted_url}")
                return converted_url
            return self.database_url
        # Otherwise, construct from individual components
        constructed_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        # logger.info(f"Constructed database URL: {constructed_url}")
        return constructed_url

    model_config = ConfigDict(env_prefix="", extra="ignore", populate_by_name=True)


class RedisSettings(BaseModel):
    """Redis configuration settings."""

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")

    # Cache TTL settings
    ttl_short: int = Field(default=300, alias="CACHE_TTL_SHORT")  # 5 minutes
    ttl_medium: int = Field(default=3600, alias="CACHE_TTL_MEDIUM")  # 1 hour
    ttl_long: int = Field(default=86400, alias="CACHE_TTL_LONG")  # 24 hours

    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    model_config = ConfigDict(env_prefix="", extra="ignore", populate_by_name=True)


class LLMSettings(BaseModel):
    """LLM API configuration settings."""

    groq_api_key: Optional[str] = Field(default=None)
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_model: str = Field(default="llama3-70b-8192")

    # Alternative providers
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # Timeout settings
    timeout_seconds: int = Field(default=60)

    model_config = ConfigDict(env_prefix="", extra="ignore")


class ServiceSettings(BaseModel):
    """Service configuration settings."""

    # Service ports
    supervisor_port: int = Field(default=8000, alias="SUPERVISOR_AGENT_PORT")
    nutrition_port: int = Field(default=8001, alias="NUTRITION_AGENT_PORT")
    workout_port: int = Field(default=8002, alias="WORKOUT_AGENT_PORT")
    vision_port: int = Field(default=8003, alias="VISION_AGENT_PORT")
    activity_port: int = Field(default=8004, alias="ACTIVITY_AGENT_PORT")

    # Service URLs for inter-service communication
    supervisor_url: str = Field(
        default="http://localhost:8000", alias="SUPERVISOR_AGENT_URL"
    )
    nutrition_url: str = Field(
        default="http://localhost:8001", alias="NUTRITION_AGENT_URL"
    )
    workout_url: str = Field(default="http://localhost:8002", alias="WORKOUT_AGENT_URL")
    vision_url: str = Field(default="http://localhost:8003", alias="VISION_AGENT_URL")
    activity_url: str = Field(
        default="http://localhost:8004", alias="ACTIVITY_AGENT_URL"
    )

    # Request timeout
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = ConfigDict(
        env_prefix="SERVICE_", extra="ignore", populate_by_name=True
    )


class SecuritySettings(BaseModel):
    """Security configuration settings."""

    jwt_secret_key: str = Field(default="dev_secret_key_change_in_production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=100)
    rate_limit_requests_per_hour: int = Field(default=1000)

    model_config = ConfigDict(env_prefix="", extra="ignore")


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File logging
    log_file: Optional[str] = Field(default=None)
    max_file_size_mb: int = Field(default=10)
    backup_count: int = Field(default=5)

    model_config = ConfigDict(env_prefix="", extra="ignore")


class MonitoringSettings(BaseModel):
    """Monitoring configuration settings."""

    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)

    # Error tracking
    sentry_dsn: Optional[str] = Field(default=None)

    model_config = ConfigDict(env_prefix="MONITORING_", extra="ignore")


class FeatureFlags(BaseModel):
    """Feature flag configuration."""

    enable_vision_agent: bool = Field(default=True)
    enable_activity_agent: bool = Field(default=True)
    enable_push_notifications: bool = Field(default=False)
    enable_analytics: bool = Field(default=True)

    model_config = ConfigDict(env_prefix="", extra="ignore")


class MultiDatabaseSettings(BaseModel):
    nutrition_db_uri: Optional[str] = Field(default=None, alias="NUTRITION_DB_URI")
    workout_db_uri: Optional[str] = Field(default=None, alias="WORKOUT_DB_URI")
    model_config = ConfigDict(env_prefix="", extra="ignore", populate_by_name=True)


class Settings(BaseModel):
    """Main settings class that combines all configuration sections."""

    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # CORS settings
    cors_origins: List[str] = Field(default=["http://localhost:3000"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Reduced logging for faster startup
        # logger.info("Initializing Settings...")
        
        # Get mapped data for sub-settings
        llm_data = data.get("_llm_data", {})
        security_data = data.get("_security_data", {})
        multi_db_data = data.get("_multi_db_data", {})

        # Initialize sub-settings with mapped data
        # logger.info("Initializing DatabaseSettings...")
        import os
        self._database = DatabaseSettings.model_validate(os.environ)
        # logger.info("Initializing RedisSettings...")
        self._redis = RedisSettings()
        # logger.info("Initializing LLMSettings...")
        self._llm = LLMSettings.model_validate(llm_data)
        # logger.info("Initializing ServiceSettings...")
        self._service = ServiceSettings()
        # logger.info("Initializing SecuritySettings...")
        self._security = SecuritySettings.model_validate(security_data)
        # logger.info("Initializing LoggingSettings...")
        self._logging = LoggingSettings()
        # logger.info("Initializing MonitoringSettings...")
        self._monitoring = MonitoringSettings()
        # logger.info("Initializing FeatureFlags...")
        self._features = FeatureFlags()
        # logger.info("Initializing MultiDatabaseSettings...")
        self._multi_db = MultiDatabaseSettings.model_validate(os.environ)
        
        # logger.info("Settings initialization complete")

    @property
    def database(self) -> DatabaseSettings:
        """Get database settings."""
        return self._database

    @property
    def redis(self) -> RedisSettings:
        """Get Redis settings."""
        return self._redis

    @property
    def llm(self) -> LLMSettings:
        """Get LLM settings."""
        return self._llm

    @property
    def service(self) -> ServiceSettings:
        """Get service settings."""
        return self._service

    @property
    def security(self) -> SecuritySettings:
        """Get security settings."""
        return self._security

    @property
    def logging(self) -> LoggingSettings:
        """Get logging settings."""
        return self._logging

    @property
    def monitoring(self) -> MonitoringSettings:
        """Get monitoring settings."""
        return self._monitoring

    @property
    def features(self) -> FeatureFlags:
        """Get feature flags."""
        return self._features

    @property
    def multi_db(self) -> MultiDatabaseSettings:
        """Get multi-database settings."""
        return self._multi_db

    model_config = ConfigDict(env_prefix="", extra="allow")


# Global settings instance
_settings: Optional[Settings] = None


def _map_env_to_fields(env_vars: dict, field_mappings: dict) -> dict:
    """Map environment variables to field names based on env parameter."""
    result = {}
    for field_name, env_name in field_mappings.items():
        if env_name in env_vars:
            result[field_name] = env_vars[env_name]
    return result


def get_settings() -> Settings:
    """Get the global settings instance, always reloading from environment."""
    # Reduced logging for faster startup
    # logger.info("Loading application settings...")
    # logger.info(f"DATABASE_URL environment variable: {os.getenv('DATABASE_URL', 'Not set')}")
    # logger.info(f"REDIS_URL environment variable: {os.getenv('REDIS_URL', 'Not set')}")
    
    # Get all environment variables
    env_vars = dict(os.environ)

    # Define field mappings for each settings class
    llm_mappings = {
        "groq_api_key": "GROQ_API_KEY",
        "groq_base_url": "GROQ_BASE_URL",
        "groq_model": "GROQ_MODEL",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "timeout_seconds": "LLM_TIMEOUT_SECONDS",
    }

    security_mappings = {
        "jwt_secret_key": "JWT_SECRET_KEY",
        "jwt_algorithm": "JWT_ALGORITHM",
        "jwt_expiration_hours": "JWT_EXPIRATION_HOURS",
        "rate_limit_requests_per_minute": "RATE_LIMIT_REQUESTS_PER_MINUTE",
        "rate_limit_requests_per_hour": "RATE_LIMIT_REQUESTS_PER_HOUR",
    }

    # Create settings with mapped environment variables
    settings_data = {
        "environment": env_vars.get("ENVIRONMENT", "development"),
        "debug": env_vars.get("DEBUG", "true").lower() == "true",
        "cors_origins": env_vars.get("CORS_ORIGINS", "http://localhost:3000"),
        "_llm_data": _map_env_to_fields(env_vars, llm_mappings),
        "_security_data": _map_env_to_fields(env_vars, security_mappings),
    }

    settings = Settings.model_validate(settings_data)
    # Reduced logging for faster startup
    # logger.info(f"Final database URL: {settings.database.url}")
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    return get_settings()
