#!/usr/bin/env python3
"""
Database initialization script for Nutrition Agent
Creates all required tables and indexes
"""

import asyncio
import logging
from shared.database import init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Initialize the database with all required tables."""
    try:
        logger.info("Starting database initialization...")
        await init_database()
        logger.info("Database initialization completed successfully!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 