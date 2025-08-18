"""
User Data Access Module for Nutrition Agent

This module provides functions to access user data from the separate user database
using the USER_DATABASE_URI configuration.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy import text
from .database import get_user_db_engine

logger = logging.getLogger(__name__)

class UserDataAccess:
    """Handles user data access from the separate user database."""
    
    def __init__(self):
        self.user_engine = None
    
    def _get_user_engine(self):
        """Get user database engine with lazy initialization."""
        if self.user_engine is None:
            self.user_engine = get_user_db_engine()
        return self.user_engine
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from user database."""
        try:
            engine = self._get_user_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM users WHERE id = :user_id"),
                    {"user_id": user_id}
                ).fetchone()
                
                if result:
                    return dict(result._mapping)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def get_user_nutrition_data(self, user_id: str) -> Dict[str, Any]:
        """Get user nutrition data from user database."""
        try:
            engine = self._get_user_engine()
            with engine.connect() as conn:
                # Get nutrition targets
                targets_result = conn.execute(
                    text("SELECT * FROM user_nutrition_targets WHERE user_id = :user_id"),
                    {"user_id": user_id}
                ).fetchall()
                
                # Get recent food logs
                logs_result = conn.execute(
                    text("SELECT * FROM user_food_logs WHERE user_id = :user_id ORDER BY consumed_at DESC LIMIT 10"),
                    {"user_id": user_id}
                ).fetchall()
                
                # Get meal plans
                meal_plans_result = conn.execute(
                    text("SELECT * FROM meal_plans WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 5"),
                    {"user_id": user_id}
                ).fetchall()
                
                return {
                    "targets": [dict(target._mapping) for target in targets_result],
                    "recent_logs": [dict(log._mapping) for log in logs_result],
                    "meal_plans": [dict(plan._mapping) for plan in meal_plans_result],
                    "total_targets": len(targets_result),
                    "total_logs": len(logs_result),
                    "total_meal_plans": len(meal_plans_result)
                }
                
        except Exception as e:
            logger.error(f"Failed to get user nutrition data for {user_id}: {e}")
            return {"targets": [], "recent_logs": [], "meal_plans": [], "total_targets": 0, "total_logs": 0, "total_meal_plans": 0}
    
    async def get_user_workout_data(self, user_id: str) -> Dict[str, Any]:
        """Get user workout data from user database."""
        try:
            engine = self._get_user_engine()
            with engine.connect() as conn:
                # Get workout templates
                templates_result = conn.execute(
                    text("SELECT * FROM workout_templates WHERE user_id = :user_id"),
                    {"user_id": user_id}
                ).fetchall()
                
                # Get recent workouts
                workouts_result = conn.execute(
                    text("SELECT * FROM user_workouts WHERE user_id = :user_id ORDER BY started_at DESC LIMIT 10"),
                    {"user_id": user_id}
                ).fetchall()
                
                return {
                    "templates": [dict(template._mapping) for template in templates_result],
                    "recent_workouts": [dict(workout._mapping) for workout in workouts_result],
                    "total_templates": len(templates_result),
                    "total_workouts": len(workouts_result)
                }
                
        except Exception as e:
            logger.error(f"Failed to get user workout data for {user_id}: {e}")
            return {"templates": [], "recent_workouts": [], "total_templates": 0, "total_workouts": 0}
    
    async def get_user_activity_data(self, user_id: str) -> Dict[str, Any]:
        """Get user activity data from user database."""
        try:
            engine = self._get_user_engine()
            with engine.connect() as conn:
                # Get activity goals
                goals_result = conn.execute(
                    text("SELECT * FROM user_activity_goals WHERE user_id = :user_id"),
                    {"user_id": user_id}
                ).fetchall()
                
                # Get recent activity logs
                logs_result = conn.execute(
                    text("SELECT * FROM user_activity_logs WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 10"),
                    {"user_id": user_id}
                ).fetchall()
                
                return {
                    "goals": [dict(goal._mapping) for goal in goals_result],
                    "recent_logs": [dict(log._mapping) for log in logs_result],
                    "total_goals": len(goals_result),
                    "total_logs": len(logs_result)
                }
                
        except Exception as e:
            logger.error(f"Failed to get user activity data for {user_id}: {e}")
            return {"goals": [], "recent_logs": [], "total_goals": 0, "total_logs": 0}
    
    async def get_user_comprehensive_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user data from all tables."""
        try:
            profile = await self.get_user_profile(user_id)
            nutrition_data = await self.get_user_nutrition_data(user_id)
            workout_data = await self.get_user_workout_data(user_id)
            activity_data = await self.get_user_activity_data(user_id)
            
            return {
                "profile": profile,
                "nutrition": nutrition_data,
                "workout": workout_data,
                "activity": activity_data,
                "user_id": user_id,
                "data_source": "user_database"
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive user data for {user_id}: {e}")
            return {
                "profile": None,
                "nutrition": {"targets": [], "recent_logs": [], "meal_plans": [], "total_targets": 0, "total_logs": 0, "total_meal_plans": 0},
                "workout": {"templates": [], "recent_workouts": [], "total_templates": 0, "total_workouts": 0},
                "activity": {"goals": [], "recent_logs": [], "total_goals": 0, "total_logs": 0},
                "user_id": user_id,
                "data_source": "fallback",
                "error": str(e)
            }
    
    async def validate_user_exists(self, user_id: str) -> bool:
        """Validate that user exists in user database."""
        try:
            engine = self._get_user_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM users WHERE id = :user_id"),
                    {"user_id": user_id}
                ).scalar()
                
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to validate user existence for {user_id}: {e}")
            return False

# Global instance
user_data_access = UserDataAccess()
