"""
Utility functions for the AI Agent Framework.

This module provides common utilities for logging, error handling,
caching, and other shared functionality.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import redis.asyncio as redis
from pydantic import BaseModel

from .config import get_settings

# Type variables for generic functions
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Global logger
logger = logging.getLogger("fitness_ai_framework")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Set up logging configuration."""
    settings = get_settings()

    # Use settings if not provided
    level = level or settings.logging.level
    log_file = log_file or settings.logging.log_file
    format_string = format_string or settings.logging.format

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)


class CacheManager:
    """Redis-based cache manager."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.settings = get_settings()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        try:
            if ttl is None:
                ttl = self.settings.redis.ttl_medium

            serialized = json.dumps(value, default=str)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False

    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments."""
        key_parts = [prefix]

        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(
                    hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[
                        :8
                    ]
                )
            else:
                key_parts.append(str(arg))

        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:8])

        return ":".join(key_parts)


def cache_result(
    prefix: str,
    ttl: Optional[int] = None,
    key_generator: Optional[Callable] = None,
):
    """Decorator to cache function results."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            from .database import get_redis

            redis_client = await get_redis()
            cache = CacheManager(redis_client)

            # Generate cache key
            if key_generator:
                cache_key = key_generator(prefix, *args, **kwargs)
            else:
                cache_key = cache.generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ErrorHandler:
    """Error handling utilities."""

    @staticmethod
    def handle_database_error(e: Exception, operation: str) -> Dict[str, Any]:
        """Handle database errors."""
        logger.error(f"Database error during {operation}: {e}")
        return {
            "success": False,
            "error": f"Database operation failed: {str(e)}",
            "operation": operation,
        }

    @staticmethod
    def handle_api_error(e: Exception, endpoint: str) -> Dict[str, Any]:
        """Handle API errors."""
        logger.error(f"API error calling {endpoint}: {e}")
        return {
            "success": False,
            "error": f"API call failed: {str(e)}",
            "endpoint": endpoint,
        }

    @staticmethod
    def handle_validation_error(e: Exception, data: Any) -> Dict[str, Any]:
        """Handle validation errors."""
        logger.error(f"Validation error: {e}")
        return {
            "success": False,
            "error": f"Validation failed: {str(e)}",
            "data": str(data),
        }


class PerformanceMonitor:
    """Performance monitoring utilities."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def time_function(self, func_name: str):
        """Decorator to time function execution."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_metric(func_name, execution_time)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_metric(func_name, execution_time)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

        # Keep only last 1000 measurements
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = self.metrics[name]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format."""
        if len(username) < 3 or len(username) > 30:
            return False
        return bool(username.replace("_", "").replace("-", "").isalnum())

    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input."""
        import html

        return html.escape(value.strip())

    @staticmethod
    def validate_nutrition_data(data: Dict[str, Any]) -> bool:
        """Validate nutrition data."""
        required_fields = ["calories", "protein_g", "carbs_g", "fat_g"]
        for field in required_fields:
            if field not in data:
                return False
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False
        return True


class DateTimeUtils:
    """Date and time utilities."""

    @staticmethod
    def now() -> datetime:
        """Get current UTC datetime."""
        return datetime.utcnow()

    @staticmethod
    def today() -> datetime:
        """Get start of today in UTC."""
        now = datetime.utcnow()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def start_of_week(date: datetime) -> datetime:
        """Get start of week for given date."""
        return date - timedelta(days=date.weekday())

    @staticmethod
    def start_of_month(date: datetime) -> datetime:
        """Get start of month for given date."""
        return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string."""
        return dt.strftime(format_str)

    @staticmethod
    def parse_datetime(
        date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S"
    ) -> datetime:
        """Parse string to datetime."""
        return datetime.strptime(date_str, format_str)


class ResponseBuilder:
    """Response building utilities."""

    @staticmethod
    def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """Build success response."""
        response = {
            "success": True,
            "message": message,
            "timestamp": DateTimeUtils.now().isoformat(),
        }
        if data is not None:
            response["data"] = data
        return response

    @staticmethod
    def error_response(
        error: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build error response."""
        response = {
            "success": False,
            "error": error,
            "timestamp": DateTimeUtils.now().isoformat(),
        }
        if error_code:
            response["error_code"] = error_code
        if details:
            response["details"] = details
        return response

    @staticmethod
    def paginated_response(
        data: List[Any],
        page: int,
        page_size: int,
        total: int,
    ) -> Dict[str, Any]:
        """Build paginated response."""
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size,
            },
            "timestamp": DateTimeUtils.now().isoformat(),
        }


# Initialize logging
# setup_logging()  # Removed to prevent module-level execution
