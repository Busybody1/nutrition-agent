"""
Database utilities for the AI Agent Framework.

This module provides database connection management for PostgreSQL and Redis,
including connection pooling and session management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from .config import get_settings

# Set up logger - reduced for faster startup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Global database instances
_async_engine: Optional[create_async_engine] = None
_sync_engine: Optional[create_engine] = None
_async_session_maker: Optional[async_sessionmaker] = None
_sync_session_maker: Optional[sessionmaker] = None
_redis_client: Optional[redis.Redis] = None


class DatabaseManager:
    """Database connection manager for PostgreSQL and Redis."""

    def __init__(self):
        self.settings = get_settings()
        self._async_engine = None
        self._sync_engine = None
        self._async_session_maker = None
        self._sync_session_maker = None
        self._redis_client = None

    async def initialize(self):
        """Initialize database connections."""
        await self._init_postgres()
        await self._init_redis()

    async def _init_postgres(self):
        """Initialize PostgreSQL connections."""
        # Create async engine
        self._async_engine = create_async_engine(
            self.settings.database.url.replace(
                "postgresql://", "postgresql+asyncpg://"
            ),
            echo=self.settings.debug,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        # Create sync engine for migrations and tests
        self._sync_engine = create_engine(
            self.settings.database.url,
            echo=self.settings.debug,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        # Create session makers
        self._async_session_maker = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._sync_session_maker = sessionmaker(
            self._sync_engine,
            expire_on_commit=False,
        )

    async def _init_redis(self):
        """Initialize Redis connection."""
        self._redis_client = redis.from_url(
            self.settings.redis.url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # Test connection
        await self._redis_client.ping()

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session."""
        if not self._async_session_maker:
            await self.initialize()

        async with self._async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self):
        """Get sync database session."""
        if not self._sync_session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        return self._sync_session_maker()

    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._redis_client:
            await self.initialize()

        return self._redis_client

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            # Check PostgreSQL
            async with self._async_session_maker() as session:
                await session.execute(text("SELECT 1"))

            # Check Redis
            await self._redis_client.ping()

            return True
        except Exception:
            return False

    async def close(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()

        if self._sync_engine:
            self._sync_engine.dispose()

        if self._redis_client:
            await self._redis_client.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        # Don't initialize immediately to avoid blocking startup
        # Initialization will happen when first needed
    return _db_manager


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session context manager."""
    db = await get_database()
    async for session in db.get_async_session():
        yield session


def get_sync_session():
    """Get sync database session."""
    db = asyncio.run(get_database())
    return db.get_sync_session()


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    db = await get_database()
    return await db.get_redis_client()


async def health_check() -> bool:
    """Check database health."""
    db = await get_database()
    return await db.health_check()


async def close_database():
    """Close database connections."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


# Database initialization utilities
async def init_database():
    """Initialize database with tables and indexes."""
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        Text,
    )
    from sqlalchemy.dialects.postgresql import UUID as PGUUID

    settings = get_settings()

    # Create sync engine for migrations
    db_url = settings.database.url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    metadata = MetaData()

    # Define tables
    users = Table(
        "users",
        metadata,
        Column("id", PGUUID, primary_key=True),
        Column("email", String(255), unique=True, nullable=False),
        Column("username", String(50), unique=True, nullable=False),
        Column("first_name", String(100), nullable=False),
        Column("last_name", String(100), nullable=False),
        Column("role", String(20), nullable=False, default="user"),
        Column("age", Integer),
        Column("height_cm", Float),
        Column("weight_kg", Float),
        Column("primary_goal", String(50)),
        Column("daily_calorie_target", Integer),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        Column("is_active", Boolean, default=True),
        Column("is_verified", Boolean, default=False),
    )

    # Create 'foods' table (not 'food_items') as expected by nutrition agent
    foods = Table(
        "foods",
        metadata,
        Column("id", PGUUID, primary_key=True),
        Column("name", String(255), nullable=False),
        Column("brand_id", String(255)),
        Column("category_id", String(100), nullable=False),
        Column("serving_size", Float, nullable=False),
        Column("serving_unit", String(50), nullable=False),
        Column("serving", String(100)),
        Column("calories", Float, nullable=False),
        Column("protein_g", Float, nullable=False),
        Column("carbs_g", Float, nullable=False),
        Column("fat_g", Float, nullable=False),
        Column("fiber_g", Float, default=0),
        Column("sugar_g", Float, default=0),
        Column("source", String(50), default="user_input"),
        Column("verified", Boolean, default=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )

    # Create 'food_logs' table with additional serving columns
    food_logs = Table(
        "food_logs",
        metadata,
        Column("id", PGUUID, primary_key=True),
        Column("user_id", PGUUID, nullable=False),
        Column("food_item_id", PGUUID, nullable=False),
        Column("quantity_g", Float, nullable=False),
        Column("meal_type", String(20), nullable=False),
        Column("consumed_at", DateTime, nullable=False),
        Column("calories", Float, nullable=False),
        Column("protein_g", Float, nullable=False),
        Column("carbs_g", Float, nullable=False),
        Column("fat_g", Float, nullable=False),
        Column("serving_size", Float),  # Additional serving columns
        Column("serving_unit", String(50)),
        Column("serving", String(100)),
        Column("notes", Text),
        Column("created_at", DateTime, nullable=False),
    )

    workout_logs = Table(
        "workout_logs",
        metadata,
        Column("id", PGUUID, primary_key=True),
        Column("user_id", PGUUID, nullable=False),
        Column("name", String(255), nullable=False),
        Column("duration_minutes", Integer, nullable=False),
        Column("started_at", DateTime, nullable=False),
        Column("completed_at", DateTime),
        Column("calories_burned", Integer),
        Column("notes", Text),
        Column("created_at", DateTime, nullable=False),
    )

    # Create tables
    metadata.create_all(engine)

    # Create indexes
    engine.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
    engine.execute(
        text("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    )
    engine.execute(
        text("CREATE INDEX IF NOT EXISTS idx_foods_name ON foods(name)")
    )
    engine.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_foods_category_id ON foods(category_id)"
        )
    )
    engine.execute(
        text("CREATE INDEX IF NOT EXISTS idx_food_logs_user_id ON food_logs(user_id)")
    )
    engine.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_food_logs_consumed_at ON food_logs(consumed_at)"
        )
    )
    engine.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_workout_logs_user_id ON workout_logs(user_id)"
        )
    )
    engine.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_workout_logs_started_at ON workout_logs(started_at)"
        )
    )

    engine.dispose()


def get_nutrition_db_engine():
    """Get nutrition database engine for reading nutrition data."""
    try:
        settings = get_settings()
        logger.info(f"Getting nutrition database settings...")
        logger.info(f"Multi DB settings: {settings.multi_db}")
        
        if settings.multi_db.nutrition_db_uri:
            db_url = settings.multi_db.nutrition_db_uri
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.info(f"Using nutrition database: {db_url[:50]}...")
            engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
            logger.info("Successfully created nutrition database engine")
            return engine
        else:
            # Fallback to main database if nutrition_db_uri is not provided
            db_url = settings.database.url
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.info(f"Using main database for nutrition: {db_url[:50]}...")
            engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
            logger.info("Successfully created main database engine for nutrition")
            return engine
    except Exception as e:
        import traceback
        error_msg = str(e) if e else "Unknown error"
        logger.error(f"Failed to create nutrition database engine: {error_msg}\n{traceback.format_exc()}")
        # Return a dummy engine for fallback
        return create_engine("sqlite:///:memory:")


def get_workout_db_engine():
    """Get workout database engine with lazy initialization."""
    try:
        settings = get_settings()
        if settings.multi_db.workout_db_uri:
            db_url = settings.multi_db.workout_db_uri
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            return create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
        else:
            # Fallback to main database if workout_db_uri is not provided
            db_url = settings.database.url
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            return create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
    except Exception as e:
        logger.error(f"Failed to create workout database engine: {e}")
        # Return a dummy engine for fallback
        return create_engine("sqlite:///:memory:")


def get_fitness_db_engine():
    """Get fitness database engine with lazy initialization."""
    try:
        from .config import get_settings
        settings = get_settings()
        logger.info(f"Getting fitness database settings...")
        logger.info(f"Database settings: {settings.database}")
        
        db_url = settings.database.url
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        logger.info(f"Using fitness database: {db_url[:50]}...")
        engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)  # Uses the main DB config (Fitness AI Agent DB)
        logger.info("Successfully created fitness database engine")
        return engine
    except Exception as e:
        import traceback
        error_msg = str(e) if e else "Unknown error"
        logger.error(f"Failed to create fitness database engine: {error_msg}\n{traceback.format_exc()}")
        # Return a dummy engine for fallback
        return create_engine("sqlite:///:memory:")
