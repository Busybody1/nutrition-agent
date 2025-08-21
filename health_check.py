#!/usr/bin/env python3
"""
Comprehensive Health Check Script for Fitness AI Agent Framework
This script checks the health of all components.
"""

import os
import sys
import asyncio
import requests
from datetime import datetime

def check_database_connection(database_url):
    """Check database connection."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {e}"

def check_redis_connection(redis_url):
    """Check Redis connection."""
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        return True, "Redis connection successful"
    except Exception as e:
        return False, f"Redis connection failed: {e}"

def check_agent_endpoint(agent_url):
    """Check agent endpoint."""
    try:
        response = requests.get(f"{agent_url}/health", timeout=10)
        if response.status_code == 200:
            return True, f"Agent endpoint healthy: {response.status_code}"
        else:
            return False, f"Agent endpoint unhealthy: {response.status_code}"
    except Exception as e:
        return False, f"Agent endpoint check failed: {e}"

def main():
    """Main health check function."""
    print("Fitness AI Agent Framework Health Check")
    print("=" * 50)
    
    # Check environment variables
    database_url = os.getenv("DATABASE_URL")
    redis_url = os.getenv("REDIS_URL")
    
    if not database_url:
        print("DATABASE_URL not set")
        return False
    
    # Check database
    db_healthy, db_message = check_database_connection(database_url)
    print(f"{'OK' if db_healthy else 'FAILED'} Database: {db_message}")
    
    # Check Redis if available
    if redis_url:
        redis_healthy, redis_message = check_redis_connection(redis_url)
        print(f"{'OK' if redis_healthy else 'FAILED'} Redis: {redis_message}")
    else:
        print("WARNING: Redis: Not configured")
        redis_healthy = True  # Not critical
    
    # Check agent endpoints
    agents = {
        "Activity Agent": os.getenv("ACTIVITY_AGENT_URL", "http://localhost:8000"),
        "Nutrition Agent": os.getenv("NUTRITION_AGENT_URL", "http://localhost:8001"),
        "Supervisor Agent": os.getenv("SUPERVISOR_AGENT_URL", "http://localhost:8002"),
        "Workout Agent": os.getenv("WORKOUT_AGENT_URL", "http://localhost:8003"),
        "Vision Agent": os.getenv("VISION_AGENT_URL", "http://localhost:8004")
    }
    
    agent_health = {}
    for agent_name, agent_url in agents.items():
        healthy, message = check_agent_endpoint(agent_url)
        agent_health[agent_name] = healthy
        print(f"{'OK' if healthy else 'FAILED'} {agent_name}: {message}")
    
    # Summary
    all_healthy = db_healthy and redis_healthy and all(agent_health.values())
    
    print("\nHealth Check Summary")
    print("=" * 30)
    if all_healthy:
        print("All components are healthy!")
        return True
    else:
        print("Some components are unhealthy. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
