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

        # Create sync engine for compatibility
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
# Table creation removed - tables are managed manually in PostgreSQL
async def init_database():
    """Tables are managed manually in PostgreSQL - this function does nothing."""
    pass


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


def get_user_db_engine():
    """Get user database engine with lazy initialization."""
    try:
        settings = get_settings()
        if settings.multi_db.user_database_uri:
            db_url = settings.multi_db.user_database_uri
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.info(f"Using user database: {db_url[:50]}...")
            engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
                echo=False,
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "nutrition-agent-user"
                }
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully created user database engine")
            return engine
        else:
            # Fallback to main database if user_database_uri is not provided
            db_url = settings.database.url
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.info(f"Using main database for user data: {db_url[:50]}...")
            engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
            logger.info("Successfully created main database engine for user data")
            return engine
    except Exception as e:
        logger.error(f"Failed to create user database engine: {e}")
        # Return a dummy engine for fallback
        return create_engine("sqlite:///:memory:")
