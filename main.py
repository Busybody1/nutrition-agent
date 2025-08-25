#!/usr/bin/env python3
"""
Nutrition Agent - Groq AI Integration with Standardized Architecture

This agent provides intelligent nutrition functionality using Groq AI.
"""

import logging
import os
from datetime import datetime, timezone, timedelta, date, time
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from groq import Groq

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import from new utils structure
from utils.database import (
    get_main_db, get_user_db, get_nutrition_db, get_workout_db,
    test_database_connection, get_database_status, Base
)
from utils.config import (
    get_database_url, get_user_database_uri, get_nutrition_db_uri, get_workout_db_uri,
    get_redis_url, get_groq_api_key, get_environment, get_log_level,
    get_port, get_host, get_cors_origins, get_groq_model, get_groq_timeout
)

# Import models and schemas
from models import (
    MealPlan, MealPlanDay, MealPlanMeal, MealPlanItem,
    UserFoodLog, UserFoodLogItem, UserNutritionTarget, UserNutritionSummary
)
from schemas import (
    MealPlanCreate, MealPlanUpdate, UserFoodLogCreate, 
    UserNutritionTargetCreate, UserNutritionTargetUpdate
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLE NUTRITION AGENT
# =============================================================================

# Simple in-memory storage (for demo purposes)
nutrition_data = {}

# Initialize Groq client
groq_client = None

# Database connection status
database_status = {"main": False, "user": False, "nutrition": False, "workout": False}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class NutritionRequest(BaseModel):
    """Simple nutrition request model."""
    message: str
    user_id: str

class NutritionResponse(BaseModel):
    """Simple nutrition response model."""
    response: str
    user_id: str
    agent: str
    timestamp: str

# =============================================================================
# SIMPLE DATABASE INITIALIZATION
# =============================================================================

def initialize_databases():
    """Initialize and test all database connections."""
    global database_status
    
    try:
        logger.info("🔍 Testing database connections...")
        
        # Test each database connection
        for db_type in ["main", "user", "nutrition", "workout"]:
            try:
                is_connected, message = test_database_connection(db_type)
                database_status[db_type] = is_connected
                if is_connected:
                    logger.info(f"✅ {db_type.capitalize()} database: {message}")
                else:
                    logger.warning(f"⚠️ {db_type.capitalize()} database: {message}")
            except Exception as e:
                logger.error(f"❌ {db_type.capitalize()} database test failed: {e}")
                database_status[db_type] = False
        
        logger.info("✅ Database initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        database_status = {"main": False, "user": False, "nutrition": False, "workout": False}

@asynccontextmanager
async def get_db():
    """Get database session from main database."""
    try:
        db = get_main_db()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")

# =============================================================================
# AI INITIALIZATION
# =============================================================================

async def initialize_ai():
    """Initialize AI clients on startup."""
    global groq_client
    
    try:
        # Initialize Groq AI
        groq_api_key = get_groq_api_key()
        if groq_api_key:
            try:
                groq_client = Groq(api_key=groq_api_key)
                # Test connection with a simple request
                test_response = groq_client.chat.completions.create(
                    model=get_groq_model(),
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    temperature=0.1
                )
                logger.info("✅ Groq AI client initialized and tested successfully")
            except Exception as e:
                logger.error(f"❌ Groq AI initialization failed: {e}")
                groq_client = None
        else:
            logger.warning("⚠️ GROQ_API_KEY not set, AI nutrition features will not work")
            groq_client = None
            
    except Exception as e:
        logger.error(f"AI initialization error: {e}")
        groq_client = None

# =============================================================================
# USER VALIDATION FUNCTIONS
# =============================================================================

async def validate_user_exists(user_id: str) -> bool:
    """Validate that the user exists in the database."""
    try:
        # Environment-aware user validation
        environment = get_environment()
        
        # Allow test users in development mode
        if environment == "development":
            if user_id == "default_user" or user_id.startswith("test_"):
                logger.info(f"Development mode: Allowing test user: {user_id}")
                return True
        
        # Whitelist for specific verified users (both dev and production)
        verified_users = ["999bf274-5b87-46e5-8087-8fb908f51c03"]
        if user_id in verified_users:
            logger.info(f"Allowing verified user: {user_id}")
            return True
            
        # Check against the user database using USER_DATABASE_URI
        try:
            db = get_user_db()
            try:
                # Check if user exists in users table using correct schema
                result = db.execute(
                    text("SELECT user_id, email, account_status FROM users WHERE user_id = :user_id AND account_status = 'active'"),
                    {"user_id": user_id}
                ).fetchone()
                
                if not result:
                    logger.warning(f"User validation failed: User {user_id} does not exist or is inactive")
                    return False
                    
                logger.info(f"User validation successful: User {user_id} ({result.email}) is valid")
                return True
                
            finally:
                db.close()
                
        except Exception as db_error:
            logger.error(f"Database error during user validation: {db_error}")
            # If database fails, allow the user to proceed for now
            logger.warning(f"Database validation failed for user {user_id}, allowing access")
            return True
        
    except Exception as e:
        logger.error(f"Error validating user {user_id}: {e}")
        return False

async def require_valid_user(user_id: str):
    """Require a valid user_id or raise HTTPException."""
    if not await validate_user_exists(user_id):
        raise HTTPException(
            status_code=404, 
            detail=f"User with ID '{user_id}' does not exist in the database. Please provide a valid user ID."
        )

# =============================================================================
# NUTRITION FUNCTIONS WITH GROQ AI
# =============================================================================

async def log_meal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Log a meal that was already eaten with AI nutrition analysis."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (optional) and other parameters
        description = parameters.get("description", "")
        
        # Required and optional parameters for meal logging
        food_items = parameters.get("food_items", description)  # What they actually ate
        meal_type = parameters.get("meal_type", "snack")  # breakfast, lunch, dinner, snack
        portion_size = parameters.get("portion_size", "medium")  # small, medium, large, or specific amount
        eating_time = parameters.get("eating_time", "now")  # when they ate it
        location = parameters.get("location", "home")  # home, restaurant, work, etc.
        mood_before = parameters.get("mood_before", "neutral")  # mood before eating
        mood_after = parameters.get("mood_after", "satisfied")  # mood after eating
        hunger_level = parameters.get("hunger_level", "moderate")  # low, moderate, high
        satisfaction_level = parameters.get("satisfaction_level", "satisfied")  # unsatisfied, satisfied, very_satisfied
        estimated_calories = parameters.get("estimated_calories", 0)  # user's calorie estimate
        notes = parameters.get("notes", "")  # additional notes
        
        # Create meal log entry
        meal_log = {
            "user_id": user_id,
            "description": description,
            "food_items": food_items,
            "meal_type": meal_type,
            "portion_size": portion_size,
            "eating_time": eating_time,
            "location": location,
            "mood_before": mood_before,
            "mood_after": mood_after,
            "hunger_level": hunger_level,
            "satisfaction_level": satisfaction_level,
            "estimated_calories": estimated_calories,
            "notes": notes,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store meal log in memory (in future, this would be saved to database)
        nutrition_data[f"meal_log_{user_id}_{datetime.now().timestamp()}"] = meal_log
        
        # Add AI nutrition insights if Groq is available
        if groq_client:
            try:
                insight_prompt = f"""Analyze this meal that was already eaten and provide nutrition insights:

MEAL DETAILS:
- Food Items: {food_items}
- Meal Type: {meal_type}
- Portion Size: {portion_size}
- Eating Time: {eating_time}
- Location: {location}
- Mood Before: {mood_before}
- Mood After: {mood_after}
- Hunger Level: {hunger_level}
- Satisfaction Level: {satisfaction_level}
- User's Calorie Estimate: {estimated_calories if estimated_calories > 0 else 'not provided'}
- Additional Notes: {notes if notes else 'none'}
- User Description: {description if description else 'none'}

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "meal_analysis": {{
    "meal_name": "Grilled Chicken Salad",
    "serving_info": {{
      "serving_size": "1 large bowl",
      "quantity": "1 serving",
      "portion_description": "Medium portion appropriate for {meal_type}"
    }},
    "estimated_nutrition": {{
      "calories": 450,
      "macros": {{
        "protein_g": 35,
        "carbs_g": 25,
        "fat_g": 20
      }},
      "nutrients_summary": [
        {{
          "nutrient": "Protein",
          "amount": 35,
          "unit": "g",
          "daily_value_percent": 70,
          "importance": "Essential for muscle building and repair"
        }},
        {{
          "nutrient": "Fiber",
          "amount": 8,
          "unit": "g",
          "daily_value_percent": 32,
          "importance": "Promotes digestive health and satiety"
        }},
        {{
          "nutrient": "Vitamin A",
          "amount": 1200,
          "unit": "mcg",
          "daily_value_percent": 133,
          "importance": "Supports vision and immune function"
        }},
        {{
          "nutrient": "Vitamin C",
          "amount": 45,
          "unit": "mg",
          "daily_value_percent": 50,
          "importance": "Antioxidant and immune support"
        }},
        {{
          "nutrient": "Iron",
          "amount": 3.5,
          "unit": "mg",
          "daily_value_percent": 19,
          "importance": "Oxygen transport and energy production"
        }},
        {{
          "nutrient": "Calcium",
          "amount": 120,
          "unit": "mg",
          "daily_value_percent": 12,
          "importance": "Bone health and muscle function"
        }},
        {{
          "nutrient": "Potassium",
          "amount": 600,
          "unit": "mg",
          "daily_value_percent": 13,
          "importance": "Electrolyte balance and heart health"
        }}
      ],
      "key_nutrients": [
        "High in protein",
        "Good source of fiber",
        "Rich in vitamins A and C"
      ]
    }},
    "health_assessment": {{
      "benefits": [
        "Excellent protein source for muscle building",
        "High fiber content for digestive health",
        "Low glycemic index for stable blood sugar"
      ],
      "concerns": [
        "May be high in sodium if using store-bought dressing"
      ]
    }},
    "satisfaction_analysis": {{
      "hunger_satisfaction": "High - protein and fiber combination",
      "nutritional_completeness": "Good - covers multiple food groups",
      "portion_appropriateness": "Appropriate for {meal_type}"
    }},
    "timing_insights": "Good timing for {eating_time}",
    "mood_connection": "Protein-rich meals often improve mood and energy",
    "recommendations": [
        "Consider adding healthy fats like avocado",
        "Include more colorful vegetables for variety"
    ],
    "balance_suggestions": "Pair with complex carbs for sustained energy"
  }}
}}

Rules:
1. Return ONLY the JSON object, no other text
2. Provide realistic calorie and macro estimates based on the food items
3. Include specific, actionable recommendations
4. Make it supportive and educational
5. Base analysis on the actual meal details provided
6. Always include serving_info with serving_size and quantity
7. Expand nutrients_summary to include important micronutrients beyond just macros
8. Include daily value percentages where applicable"""

                insight_response = groq_client.chat.completions.create(
                    model=get_groq_model(),
                    messages=[{"role": "user", "content": insight_prompt}],
                    max_tokens=600,
                    temperature=0.7,
                    timeout=get_groq_timeout()
                )
                
                ai_insights = insight_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")
                ai_insights = "AI nutrition analysis temporarily unavailable. Please try again later."
        else:
            ai_insights = "AI nutrition insights are currently unavailable."
        
        # Parse AI response to extract structured meal analysis
        try:
            # Remove markdown code blocks if present
            ai_response_clean = ai_insights
            if "```json" in ai_insights:
                ai_response_clean = ai_insights.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_insights:
                ai_response_clean = ai_insights.split("```")[1].split("```")[0].strip()
            
            import json
            structured_analysis = json.loads(ai_response_clean)
            
            return {
                "status": "success",
                "user_id": user_id,
                "meal_log": {
                    "id": f"log_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "food_items": food_items,
                    "meal_type": meal_type,
                    "portion_size": portion_size,
                    "eating_time": eating_time,
                    "location": location,
                    "mood_before": mood_before,
                    "mood_after": mood_after,
                    "hunger_level": hunger_level,
                    "satisfaction_level": satisfaction_level,
                    "estimated_calories": estimated_calories,
                    "notes": notes,
                    "structured_analysis": structured_analysis,
                    "logged_at": meal_log["logged_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Meal logged successfully with AI nutrition analysis"
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI meal analysis as JSON: {e}")
            # Fallback to original response
            return {
                "status": "success",
                "user_id": user_id,
                "meal_log": {
                    "id": f"log_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "food_items": food_items,
                    "meal_type": meal_type,
                    "portion_size": portion_size,
                    "eating_time": eating_time,
                    "location": location,
                    "mood_before": mood_before,
                    "mood_after": mood_after,
                    "hunger_level": hunger_level,
                    "satisfaction_level": satisfaction_level,
                    "estimated_calories": estimated_calories,
                    "notes": notes,
                    "ai_nutrition_analysis": ai_insights,
                    "logged_at": meal_log["logged_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Meal logged successfully with AI nutrition analysis (AI response format issue)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging meal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log meal: {str(e)}")

async def get_nutrition_summary(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Get nutrition summary with AI analysis."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (required)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of nutrition summary you need.")
        
        # Optional parameters
        days = parameters.get("days", 7)
        goals = parameters.get("goals", "general health")
        
        # Generate AI-powered summary if Groq is available
        if groq_client:
            try:
                summary_prompt = f"""Based on this user request, provide a comprehensive nutrition summary:
- User request: {description}
- Time period: {days} days
- User goals: {goals}

Provide a structured response with:
1. Summary of typical nutrition patterns
2. Nutrient balance analysis with expanded micronutrients
3. Goal achievement assessment
4. Personalized recommendations
5. Meal planning suggestions
6. Next steps for improvement

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "nutrition_summary": {{
    "period_days": {days},
    "user_goals": "{goals}",
    "summary_analysis": "Comprehensive analysis of nutrition patterns and recommendations",
    "serving_guidelines": {{
      "general_serving_sizes": "Standard portion recommendations",
      "meal_frequency": "Optimal meal timing and frequency",
      "portion_control_tips": "Practical portion management strategies"
    }},
    "nutrient_analysis": {{
      "macros_overview": "Protein, carbs, and fat balance analysis",
      "micronutrients_focus": "Key vitamins and minerals for {goals}",
      "nutrients_summary": [
        {{
          "nutrient": "Protein",
          "daily_target": "1.2-1.6g per kg body weight",
          "importance": "Essential for muscle building and repair",
          "food_sources": "Lean meats, fish, eggs, legumes"
        }},
        {{
          "nutrient": "Fiber",
          "daily_target": "25-35g",
          "importance": "Digestive health and satiety",
          "food_sources": "Whole grains, fruits, vegetables, legumes"
        }},
        {{
          "nutrient": "Vitamin D",
          "daily_target": "15-20mcg",
          "importance": "Bone health and immune function",
          "food_sources": "Fatty fish, egg yolks, fortified dairy"
        }},
        {{
          "nutrient": "Iron",
          "daily_target": "8-18mg",
          "importance": "Oxygen transport and energy production",
          "food_sources": "Red meat, spinach, legumes, fortified cereals"
        }},
        {{
          "nutrient": "Calcium",
          "daily_target": "1000-1300mg",
          "importance": "Bone health and muscle function",
          "food_sources": "Dairy products, leafy greens, fortified foods"
        }},
        {{
          "nutrient": "Omega-3",
          "daily_target": "1.1-1.6g",
          "importance": "Heart health and brain function",
          "food_sources": "Fatty fish, flaxseeds, walnuts, chia seeds"
        }},
        {{
          "nutrient": "B Vitamins",
          "daily_target": "Various",
          "importance": "Energy metabolism and brain function",
          "food_sources": "Whole grains, meat, eggs, dairy, leafy greens"
        }}
      ]
    }},
    "goal_assessment": "Current progress and areas for improvement",
    "personalized_recommendations": [
      "Specific action items for {goals}",
      "Meal timing and portion strategies",
      "Nutrient-dense food suggestions"
    ],
    "meal_planning_suggestions": "Practical meal planning approaches",
    "next_steps": "Immediate actions to take for improvement",
    "generated_at": "{datetime.now(timezone.utc).isoformat()}"
  }}
}}

Make it personalized and actionable based on their description and goals."""

                summary_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=800,
                    temperature=0.7
                )
                
                ai_summary = summary_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI summary: {e}")
                ai_summary = "AI summary generation failed, but here's your basic summary."
        else:
            ai_summary = "AI features are currently unavailable."
        
        return {
            "status": "success",
            "user_id": user_id,
            "summary": {
                "period_days": days,
                "goals": goals,
                "ai_analysis": ai_summary,
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "message": "Nutrition summary generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating nutrition summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

async def create_meal_plan(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a personalized meal plan with AI recommendations for multiple days and meals."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (optional) and other parameters
        description = parameters.get("description", "")
        
        # Required and optional parameters for meal planning
        plan_type = parameters.get("plan_type", "weekly")  # single_meal, daily, weekly
        days_per_week = parameters.get("days_per_week", 5)  # 1-7 days, default 5
        meals_per_day = parameters.get("meals_per_day", 3)  # 1-5 meals, default 3
        dietary_restrictions = parameters.get("dietary_restrictions", [])  # vegetarian, vegan, gluten-free, etc.
        calorie_target = parameters.get("calorie_target", 0)  # 0 means no specific target
        cuisine_preference = parameters.get("cuisine_preference", "any")  # italian, asian, mediterranean, etc.
        cooking_time = parameters.get("cooking_time", "medium")  # quick (<30min), medium (30-60min), long (>60min)
        skill_level = parameters.get("skill_level", "intermediate")  # beginner, intermediate, advanced
        budget = parameters.get("budget", "medium")  # low, medium, high
        
        # Validate parameters
        if days_per_week < 1 or days_per_week > 7:
            days_per_week = 5  # Default to 5 if invalid
        if meals_per_day < 1 or meals_per_day > 5:
            meals_per_day = 3  # Default to 3 if invalid
        
        # Store meal plan data
        meal_plan_data = {
            "user_id": user_id,
            "description": description,
            "plan_type": plan_type,
            "days_per_week": days_per_week,
            "meals_per_day": meals_per_day,
            "dietary_restrictions": dietary_restrictions if isinstance(dietary_restrictions, list) else [dietary_restrictions],
            "calorie_target": calorie_target,
            "cuisine_preference": cuisine_preference,
            "cooking_time": cooking_time,
            "skill_level": skill_level,
            "budget": budget,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in memory (in future, this would go to database)
        nutrition_data[f"meal_plan_{user_id}_{datetime.now().timestamp()}"] = meal_plan_data
        
        # Generate AI-powered meal plan if Groq is available
        if groq_client:
            try:
                # Build detailed prompt based on parameters
                restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                calorie_text = f"{calorie_target} calories per day" if calorie_target > 0 else "no specific calorie target"
                
                # Determine meal types based on meals_per_day
                meal_types = []
                if meals_per_day >= 1:
                    meal_types.append("breakfast")
                if meals_per_day >= 2:
                    meal_types.append("lunch")
                if meals_per_day >= 3:
                    meal_types.append("dinner")
                if meals_per_day >= 4:
                    meal_types.append("snack_1")
                if meals_per_day >= 5:
                    meal_types.append("snack_2")
                
                meal_types_text = ", ".join(meal_types)
                
                meal_prompt = f"""Create a detailed {plan_type} meal plan with the following requirements:

REQUIREMENTS:
- Plan Type: {plan_type}
- Days Per Week: {days_per_week} days
- Meals Per Day: {meals_per_day} meals ({meal_types_text})
- User Request: {description if description else "No specific request"}
- Dietary Restrictions: {restrictions_text}
- Calorie Target: {calorie_text}
- Cuisine Preference: {cuisine_preference}
- Cooking Time: {cooking_time}
- Skill Level: {skill_level}
- Budget: {budget}

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "meal_plan": {{
    "plan_info": {{
      "plan_type": "{plan_type}",
      "days_per_week": {days_per_week},
      "meals_per_day": {meals_per_day},
      "total_meals": {days_per_week * meals_per_day}
    }},
    "days": [
      {{
        "day": 1,
        "day_name": "Monday",
        "meals": {{
          "breakfast": {{
            "name": "Meal Name",
            "serving_info": {{
              "serving_size": "1 bowl",
              "quantity": "1 serving",
              "portion_description": "Standard breakfast portion"
            }},
            "calories": 350,
            "macros": {{
              "protein": 12,
              "carbs": 45,
              "fat": 14
            }},
            "nutrients_summary": [
              {{
                "nutrient": "Protein",
                "amount": 12,
                "unit": "g",
                "daily_value_percent": 24,
                "importance": "Essential for muscle building and repair"
              }},
              {{
                "nutrient": "Fiber",
                "amount": 6,
                "unit": "g",
                "daily_value_percent": 24,
                "importance": "Promotes digestive health and satiety"
              }},
              {{
                "nutrient": "Iron",
                "amount": 2.5,
                "unit": "mg",
                "daily_value_percent": 14,
                "importance": "Oxygen transport and energy production"
              }},
              {{
                "nutrient": "B Vitamins",
                "amount": "Various",
                "unit": "mg",
                "daily_value_percent": "15-25",
                "importance": "Energy metabolism and brain function"
              }}
            ],
            "ingredients": [
              "1/2 cup rolled oats",
              "1 cup almond milk"
            ],
            "prep_time": "5 minutes",
            "cooking_time": "10 minutes"
          }},
          "lunch": {{
            "name": "Meal Name",
            "serving_info": {{
              "serving_size": "1 plate",
              "quantity": "1 serving",
              "portion_description": "Standard lunch portion"
            }},
            "calories": 500,
            "macros": {{
              "protein": 40,
              "carbs": 20,
              "fat": 25
            }},
            "nutrients_summary": [
              {{
                "nutrient": "Protein",
                "amount": 40,
                "unit": "g",
                "daily_value_percent": 80,
                "importance": "Essential for muscle building and repair"
              }},
              {{
                "nutrient": "Vitamin C",
                "amount": 35,
                "unit": "mg",
                "daily_value_percent": 39,
                "importance": "Antioxidant and immune support"
              }},
              {{
                "nutrient": "Folate",
                "amount": 120,
                "unit": "mcg",
                "daily_value_percent": 30,
                "importance": "Cell division and DNA synthesis"
              }},
              {{
                "nutrient": "Potassium",
                "amount": 450,
                "unit": "mg",
                "daily_value_percent": 10,
                "importance": "Electrolyte balance and heart health"
              }}
            ],
            "ingredients": [
              "150g grilled chicken breast",
              "2 cups mixed greens"
            ],
            "prep_time": "15 minutes",
            "cooking_time": "20 minutes"
          }},
          "dinner": {{
            "name": "Meal Name",
            "serving_info": {{
              "serving_size": "1 plate",
              "quantity": "1 serving",
              "portion_description": "Standard dinner portion"
            }},
            "calories": 600,
            "macros": {{
              "protein": 45,
              "carbs": 50,
              "fat": 22
            }},
            "nutrients_summary": [
              {{
                "nutrient": "Protein",
                "amount": 45,
                "unit": "g",
                "daily_value_percent": 90,
                "importance": "Essential for muscle building and repair"
              }},
              {{
                "nutrient": "Omega-3",
                "amount": 1.2,
                "unit": "g",
                "daily_value_percent": "N/A",
                "importance": "Heart health and brain function"
              }},
              {{
                "nutrient": "Vitamin D",
                "amount": 8,
                "unit": "mcg",
                "daily_value_percent": 40,
                "importance": "Bone health and immune function"
              }},
              {{
                "nutrient": "Selenium",
                "amount": 45,
                "unit": "mcg",
                "daily_value_percent": 82,
                "importance": "Antioxidant and thyroid function"
              }}
            ],
            "ingredients": [
              "150g baked salmon",
              "1/2 cup quinoa (cooked)"
            ],
            "prep_time": "20 minutes",
            "cooking_time": "25 minutes"
          }}
        }},
        "total_calories": 1450,
        "daily_nutrition_summary": {{
          "total_protein": 97,
          "total_carbs": 115,
          "total_fat": 61,
          "total_fiber": 25,
          "key_nutrients": ["High protein", "Good fiber", "Balanced macros"]
        }}
      }}
    ],
    "weekly_summary": {{
      "total_calories": {days_per_week * 1450},
      "average_daily_calories": 1450,
      "nutrition_goals": "Balanced macronutrients with focus on protein",
      "shopping_list": [
        "Proteins: chicken breast, salmon, eggs, tuna",
        "Vegetables: spinach, mixed greens, bell peppers",
        "Grains: quinoa, oats, whole grain bread",
        "Dairy: almond milk, Greek yogurt"
      ]
    }}
  }}
}}

Rules:
1. Return ONLY the JSON object, no other text
2. Include exactly {days_per_week} days based on days_per_week parameter
3. Each day must have exactly {meals_per_day} meals based on meals_per_day parameter
4. All meals must include name, serving_info (serving_size, quantity, portion_description), calories, macros (protein, carbs, fat), nutrients_summary with expanded micronutrients, ingredients, prep_time, and cooking_time
5. Calculate total_calories for each day and weekly totals
6. Make it practical and delicious for the user's requirements
7. Always include serving_info with serving_size, quantity, and portion_description
8. Expand nutrients_summary to include important micronutrients beyond just macros
9. Include daily value percentages where applicable
10. Add prep_time and cooking_time for each meal
11. Include daily_nutrition_summary and weekly_summary
12. Provide a shopping list in the weekly_summary"""

                meal_response = groq_client.chat.completions.create(
                    model=get_groq_model(),
                    messages=[{"role": "user", "content": meal_prompt}],
                    max_tokens=800,
                    temperature=0.7,
                    timeout=get_groq_timeout()
                )
                
                ai_meal_plan = meal_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI meal plan: {e}")
                ai_meal_plan = "AI meal planning temporarily unavailable. Please try again later."
        else:
            ai_meal_plan = "AI meal planning features are currently unavailable."
        
        # Parse AI response to extract structured meal plan
        try:
            # Remove markdown code blocks if present
            ai_response_clean = ai_meal_plan
            if "```json" in ai_meal_plan:
                ai_response_clean = ai_meal_plan.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_meal_plan:
                ai_response_clean = ai_meal_plan.split("```")[1].split("```")[0].strip()
            
            import json
            structured_plan = json.loads(ai_response_clean)
            
            return {
                "status": "success",
                "user_id": user_id,
                "meal_plan": {
                    "id": f"plan_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "plan_type": plan_type,
                    "days_per_week": days_per_week,
                    "meals_per_day": meals_per_day,
                    "dietary_restrictions": dietary_restrictions,
                    "calorie_target": calorie_target,
                    "cuisine_preference": cuisine_preference,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "budget": budget,
                    "structured_plan": structured_plan,
                    "created_at": meal_plan_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Personalized {plan_type} meal plan created successfully"
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI meal plan as JSON: {e}")
            # Fallback to original response
            return {
                "status": "success",
                "user_id": user_id,
                "meal_plan": {
                    "id": f"plan_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "plan_type": plan_type,
                    "days_per_week": days_per_week,
                    "meals_per_day": meals_per_day,
                    "dietary_restrictions": dietary_restrictions,
                    "calorie_target": calorie_target,
                    "cuisine_preference": cuisine_preference,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "budget": budget,
                    "ai_generated_plan": ai_meal_plan,
                    "created_at": meal_plan_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Personalized {plan_type} meal plan created successfully (AI response format issue)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")

async def create_meal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a single meal with AI recommendations."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (optional) and other parameters
        description = parameters.get("description", "")
        
        # Required and optional parameters for single meal
        meal_type = parameters.get("meal_type", "dinner")  # breakfast, lunch, dinner, snack
        dietary_restrictions = parameters.get("dietary_restrictions", [])  # vegetarian, vegan, gluten-free, etc.
        calorie_target = parameters.get("calorie_target", 0)  # 0 means no specific target
        cuisine_preference = parameters.get("cuisine_preference", "any")  # italian, asian, mediterranean, etc.
        cooking_time = parameters.get("cooking_time", "medium")  # quick (<30min), medium (30-60min), long (>60min)
        skill_level = parameters.get("skill_level", "intermediate")  # beginner, intermediate, advanced
        budget = parameters.get("budget", "medium")  # low, medium, high
        
        # Store meal data
        meal_data = {
            "user_id": user_id,
            "description": description,
            "meal_type": meal_type,
            "dietary_restrictions": dietary_restrictions if isinstance(dietary_restrictions, list) else [dietary_restrictions],
            "calorie_target": calorie_target,
            "cuisine_preference": cuisine_preference,
            "cooking_time": cooking_time,
            "skill_level": skill_level,
            "budget": budget,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in memory (in future, this would go to database)
        nutrition_data[f"meal_{user_id}_{datetime.now().timestamp()}"] = meal_data
        
        # Generate AI-powered meal if Groq is available
        if groq_client:
            try:
                # Build detailed prompt based on parameters
                restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                calorie_text = f"{calorie_target} calories" if calorie_target > 0 else "no specific calorie target"
                
                meal_prompt = f"""Create a detailed single {meal_type} meal with the following requirements:

REQUIREMENTS:
- Meal Type: {meal_type}
- User Request: {description if description else "No specific request"}
- Dietary Restrictions: {restrictions_text}
- Calorie Target: {calorie_text}
- Cuisine Preference: {cuisine_preference}
- Cooking Time: {cooking_time}
- Skill Level: {skill_level}
- Budget: {budget}

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "meal": {{
    "meal_info": {{
      "meal_type": "{meal_type}",
      "cuisine": "{cuisine_preference}",
      "difficulty": "{skill_level}",
      "budget": "{budget}"
    }},
    "meal_details": {{
      "name": "Meal Name",
      "serving_info": {{
        "serving_size": "1 plate",
        "quantity": "1 serving",
        "portion_description": "Standard {meal_type} portion"
      }},
      "calories": 450,
      "macros": {{
        "protein": 35,
        "carbs": 25,
        "fat": 20
      }},
      "nutrients_summary": [
        {{
          "nutrient": "Protein",
          "amount": 35,
          "unit": "g",
          "daily_value_percent": 70,
          "importance": "Essential for muscle building and repair"
        }},
        {{
          "nutrient": "Fiber",
          "amount": 8,
          "unit": "g",
          "daily_value_percent": 32,
          "importance": "Promotes digestive health and satiety"
        }},
        {{
          "nutrient": "Vitamin C",
          "amount": 45,
          "unit": "mg",
          "daily_value_percent": 50,
          "importance": "Antioxidant and immune support"
        }},
        {{
          "nutrient": "Iron",
          "amount": 3.5,
          "unit": "mg",
          "daily_value_percent": 19,
          "importance": "Oxygen transport and energy production"
        }}
      ],
      "ingredients": [
        "150g protein source",
        "2 cups vegetables",
        "1/2 cup grains"
      ],
      "prep_time": "15 minutes",
      "cooking_time": "20 minutes",
      "total_time": "35 minutes"
    }},
    "cooking_instructions": [
      "Step 1: Prepare ingredients",
      "Step 2: Cook protein",
      "Step 3: Add vegetables",
      "Step 4: Season and serve"
    ],
    "tips": [
      "Use fresh ingredients for best flavor",
      "Don't overcook the protein"
    ],
    "variations": [
      "Substitute with different protein",
      "Add more vegetables for fiber"
    ],
    "nutrition_notes": "This meal provides a good balance of protein, carbs, and healthy fats"
  }}
}}

Rules:
1. Return ONLY the JSON object, no other text
2. Create a single, complete meal suitable for {meal_type}
3. Include comprehensive serving_info, nutrition details, and cooking instructions
4. Make it practical and delicious for the user's requirements
5. Always include serving_info with serving_size, quantity, and portion_description
6. Expand nutrients_summary to include important micronutrients beyond just macros
7. Include daily value percentages where applicable
8. Add prep_time, cooking_time, and total_time
9. Provide clear cooking instructions and helpful tips"""
                
                meal_response = groq_client.chat.completions.create(
                    model=get_groq_model(),
                    messages=[{"role": "user", "content": meal_prompt}],
                    max_tokens=800,
                    temperature=0.7,
                    timeout=get_groq_timeout()
                )
                
                ai_meal = meal_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI meal: {e}")
                ai_meal = "AI meal creation temporarily unavailable. Please try again later."
        else:
            ai_meal = "AI meal creation features are currently unavailable."
        
        # Parse AI response to extract structured meal
        try:
            # Remove markdown code blocks if present
            ai_response_clean = ai_meal
            if "```json" in ai_meal:
                ai_response_clean = ai_meal.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_meal:
                ai_response_clean = ai_meal.split("```")[1].split("```")[0].strip()
            
            import json
            structured_meal = json.loads(ai_response_clean)
            
            return {
                "status": "success",
                "user_id": user_id,
                "meal": {
                    "id": f"meal_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "meal_type": meal_type,
                    "dietary_restrictions": dietary_restrictions,
                    "calorie_target": calorie_target,
                    "cuisine_preference": cuisine_preference,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "budget": budget,
                    "structured_meal": structured_meal,
                    "created_at": meal_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Personalized {meal_type} meal created successfully"
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI meal as JSON: {e}")
            # Fallback to original response
            return {
                "status": "success",
                "user_id": user_id,
                "meal": {
                    "id": f"meal_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "meal_type": meal_type,
                    "dietary_restrictions": dietary_restrictions,
                    "calorie_target": calorie_target,
                    "cuisine_preference": cuisine_preference,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "budget": budget,
                    "ai_generated_meal": ai_meal,
                    "created_at": meal_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Personalized {meal_type} meal created successfully (AI response format issue)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal: {str(e)}")

async def create_recipe(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a detailed recipe with AI-powered suggestions."""
    try:
        await require_valid_user(user_id)
        
        description = parameters.get("description", "")
        recipe_name = parameters.get("recipe_name", "")
        cuisine_type = parameters.get("cuisine_type", "general")
        dietary_restrictions = parameters.get("dietary_restrictions", [])
        cooking_time = parameters.get("cooking_time", "30 minutes")
        skill_level = parameters.get("skill_level", "intermediate")
        servings = parameters.get("servings", 4)
        calorie_target = parameters.get("calorie_target", 0)
        
        recipe_data = {
            "user_id": user_id,
            "description": description,
            "recipe_name": recipe_name,
            "cuisine_type": cuisine_type,
            "dietary_restrictions": dietary_restrictions if isinstance(dietary_restrictions, list) else [dietary_restrictions],
            "cooking_time": cooking_time,
            "skill_level": skill_level,
            "servings": servings,
            "calorie_target": calorie_target,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        nutrition_data[f"recipe_{user_id}_{datetime.now().timestamp()}"] = recipe_data
        
        if groq_client:
            try:
                restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                calorie_text = f"{calorie_target} calories per serving" if calorie_target > 0 else "no specific calorie target"
                
                recipe_prompt = f"""Create a detailed recipe with the following requirements:

REQUIREMENTS:
- Recipe Name: {recipe_name if recipe_name else "Create a creative name"}
- User Request: {description if description else "No specific request"}
- Cuisine Type: {cuisine_type}
- Dietary Restrictions: {restrictions_text}
- Cooking Time: {cooking_time}
- Skill Level: {skill_level}
- Servings: {servings}
- Calorie Target: {calorie_text}

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "recipe": {{
    "name": "Recipe Name",
    "cuisine": "Cuisine Type",
    "prep_time": "15 minutes",
    "cook_time": "30 minutes",
    "total_time": "45 minutes",
    "servings": 4,
    "serving_info": {{
      "serving_size": "1 plate",
      "quantity": "1 serving",
      "portion_description": "Standard dinner portion"
    }},
    "difficulty": "intermediate",
    "nutrition_per_serving": {{
      "calories": 350,
      "macros": {{
        "protein_g": 25,
        "carbs_g": 30,
        "fat_g": 15
      }},
      "nutrients_summary": [
        {{
          "nutrient": "Protein",
          "amount": 25,
          "unit": "g",
          "daily_value_percent": 50,
          "importance": "Essential for muscle building and repair"
        }},
        {{
          "nutrient": "Fiber",
          "amount": 8,
          "unit": "g",
          "daily_value_percent": 32,
          "importance": "Promotes digestive health and satiety"
        }},
        {{
          "nutrient": "Iron",
          "amount": 3.2,
          "unit": "mg",
          "daily_value_percent": 18,
          "importance": "Oxygen transport and energy production"
        }},
        {{
          "nutrient": "Vitamin C",
          "amount": 28,
          "unit": "mg",
          "daily_value_percent": 31,
          "importance": "Antioxidant and immune support"
        }},
        {{
          "nutrient": "Folate",
          "amount": 85,
          "unit": "mcg",
          "daily_value_percent": 21,
          "importance": "Cell division and DNA synthesis"
        }},
        {{
          "nutrient": "Potassium",
          "amount": 420,
          "unit": "mg",
          "daily_value_percent": 9,
          "importance": "Electrolyte balance and heart health"
        }},
        {{
          "nutrient": "Calcium",
          "amount": 95,
          "unit": "mg",
          "daily_value_percent": 10,
          "importance": "Bone health and muscle function"
        }}
      ]
    }},
    "ingredients": [
      {{
        "item": "2 cups all-purpose flour",
        "category": "dry ingredients",
        "quantity": "2 cups",
        "notes": "Can substitute with whole wheat flour for more fiber"
      }},
      {{
        "item": "1 cup milk",
        "category": "wet ingredients",
        "quantity": "1 cup",
        "notes": "Can use almond milk for dairy-free option"
      }}
    ],
    "instructions": [
      "Step 1: Mix dry ingredients",
      "Step 2: Add wet ingredients",
      "Step 3: Bake at 350°F for 30 minutes"
    ],
    "tips": [
      "Use room temperature ingredients",
      "Don't overmix the batter"
    ],
    "variations": [
      "Add chocolate chips for sweetness",
      "Use whole wheat flour for more fiber"
    ],
    "storage": "Store in airtight container for up to 3 days",
    "dietary_notes": "Can be made gluten-free by using gluten-free flour"
  }}
}}

Rules:
1. Return ONLY the JSON object, no other text
2. Make the recipe practical and delicious
3. Include realistic cooking times and difficulty levels
4. Provide accurate nutritional estimates
5. Include helpful tips and variations
6. Consider the dietary restrictions specified
7. Always include serving_info with serving_size, quantity, and portion_description
8. Expand nutrients_summary to include important micronutrients beyond just macros
9. Include daily value percentages where applicable
10. Add quantity and notes to ingredients for better clarity"""
                
                recipe_response = groq_client.chat.completions.create(
                    model=get_groq_model(),
                    messages=[{"role": "user", "content": recipe_prompt}],
                    max_tokens=800,
                    temperature=0.7,
                    timeout=get_groq_timeout()
                )
                
                ai_recipe = recipe_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI recipe: {e}")
                ai_recipe = "AI recipe creation temporarily unavailable. Please try again later."
        else:
            ai_recipe = "AI recipe creation features are currently unavailable."
        
        # Parse AI response to extract structured recipe
        try:
            # Remove markdown code blocks if present
            ai_response_clean = ai_recipe
            if "```json" in ai_recipe:
                ai_response_clean = ai_recipe.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_recipe:
                ai_response_clean = ai_recipe.split("```")[1].split("```")[0].strip()
            
            import json
            structured_recipe = json.loads(ai_response_clean)
            
            return {
                "status": "success",
                "user_id": user_id,
                "recipe": {
                    "id": f"recipe_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "recipe_name": recipe_name,
                    "cuisine_type": cuisine_type,
                    "dietary_restrictions": dietary_restrictions,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "servings": servings,
                    "calorie_target": calorie_target,
                    "structured_recipe": structured_recipe,
                    "created_at": recipe_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Recipe created successfully with AI-powered suggestions"
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI recipe as JSON: {e}")
            # Fallback to original response
            return {
                "status": "success",
                "user_id": user_id,
                "recipe": {
                    "id": f"recipe_{user_id}_{int(datetime.now().timestamp())}",
                    "description": description,
                    "recipe_name": recipe_name,
                    "cuisine_type": cuisine_type,
                    "dietary_restrictions": dietary_restrictions,
                    "cooking_time": cooking_time,
                    "skill_level": skill_level,
                    "servings": servings,
                    "calorie_target": calorie_target,
                    "ai_generated_recipe": ai_recipe,
                    "created_at": recipe_data["created_at"]
                },
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Recipe created successfully with AI-powered suggestions (AI response format issue)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating recipe: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create recipe: {str(e)}")

async def general_nutrition_response(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """General nutrition response with AI."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description/message (required)
        message = parameters.get("description", "") or parameters.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Description or message parameter is required.")
        
        # Generate AI response if Groq is available
        if groq_client:
            try:
                ai_prompt = f"""You are a nutrition and health expert. The user asks: {message}

Provide a helpful, encouraging response that:
1. Addresses their specific nutrition question or concern
2. Offers practical advice and tips
3. Motivates them to make healthy choices
4. Suggests next steps or alternatives
5. Maintains a positive, supportive tone

IMPORTANT: You must respond with ONLY a valid JSON object in this exact structure:

{{
  "nutrition_response": {{
    "user_question": "{message}",
    "expert_analysis": "Comprehensive analysis of the user's nutrition question",
    "practical_advice": "Actionable tips and recommendations",
    "serving_guidelines": {{
      "portion_recommendations": "Specific serving size suggestions",
      "meal_timing": "Optimal timing for meals and snacks",
      "frequency_guidance": "How often to eat for best results"
    }},
    "nutrient_focus": {{
      "key_nutrients": "Most important nutrients for this concern",
      "nutrients_summary": [
        {{
          "nutrient": "Protein",
          "recommendation": "Specific protein guidance",
          "food_sources": "Best food sources for this need",
          "importance": "Why this nutrient matters for their concern"
        }},
        {{
          "nutrient": "Fiber",
          "recommendation": "Specific fiber guidance",
          "food_sources": "Best food sources for this need",
          "importance": "Why this nutrient matters for their concern"
        }},
        {{
          "nutrient": "Vitamins",
          "recommendation": "Specific vitamin guidance",
          "food_sources": "Best food sources for this need",
          "importance": "Why this nutrient matters for their concern"
        }},
        {{
          "nutrient": "Minerals",
          "recommendation": "Specific mineral guidance",
          "food_sources": "Best food sources for this need",
          "importance": "Why this nutrient matters for their concern"
        }}
      ]
    }},
    "motivation": "Encouraging words to inspire healthy choices",
    "next_steps": [
      "Immediate actions they can take",
      "Short-term goals to work toward",
      "Long-term strategies for success"
    ],
    "additional_resources": "Suggestions for further learning or support",
    "generated_at": "{datetime.now(timezone.utc).isoformat()}"
  }}
}}

Keep it concise but comprehensive, and always include serving guidelines and expanded nutrient information."""

                ai_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": ai_prompt}],
                    max_tokens=600,
                    temperature=0.7
                )
                
                response_text = ai_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI response: {e}")
                response_text = "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need."
        else:
            response_text = "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need."
        
        return {
            "status": "success",
            "response": response_text,
            "user_id": user_id,
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating nutrition response: {e}")
        # Fallback to basic response if AI fails
        return {
            "status": "success",
            "response": "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need.",
            "user_id": user_id,
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Nutrition Agent - Groq AI Integration",
    description="Intelligent nutrition agent with Groq AI for personalized meal planning and nutrition advice",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# LIFESPAN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting Nutrition Agent...")
    
    # Log environment info
    logger.info(f"🔍 Environment: {get_environment()}")
    logger.info(f"🔍 Log Level: {get_log_level()}")
    logger.info(f"🔍 Port: {get_port()}")
    
    # Initialize databases
    initialize_databases()
    
    # Initialize AI
    await initialize_ai()
    
    logger.info("✅ Nutrition Agent startup complete")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve the test interface HTML file."""
    try:
        with open("static/test_interface.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Test interface not found</h1>", status_code=404)

@app.get("/health")
async def health_check():
    """Comprehensive health check with all service statuses."""
    try:
        # Get database status using utils
        db_status = get_database_status()
        
        # Check AI status
        ai_status = "connected" if groq_client else "disconnected"
        
        # Build services status
        services_status = {
            "groq_ai": ai_status,
            "main_database": "connected" if db_status.get("main", {}).get("connected", False) else "disconnected",
            "user_database": "connected" if db_status.get("user", {}).get("connected", False) else "disconnected",
            "nutrition_database": "connected" if db_status.get("nutrition", {}).get("connected", False) else "disconnected",
            "workout_database": "connected" if db_status.get("workout", {}).get("connected", False) else "disconnected"
        }
        
        logger.info(f"Health check services status: {services_status}")
        
        return {
            "status": "healthy",
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": get_environment(),
            "services": services_status,
            "message": "Nutrition agent is running smoothly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": get_environment(),
            "error": str(e)
        }

@app.get("/test-db")
async def test_database():
    """Test database connection."""
    try:
        async with get_db() as db:
            result = db.execute(text("SELECT 1")).fetchone()
            return {
                "status": "success",
                "database": "connected",
                "test_query": result[0] if result else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return {
            "status": "error",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.post("/execute-tool")
async def execute_tool(
    tool_name: str = Body(...),
    parameters: Dict[str, Any] = Body(...)
):
    """Execute a nutrition tool."""
    try:
        user_id = parameters.get("user_id", "default_user")
        
        # Execute tool based on name
        if tool_name == "log_meal":
            return await log_meal(parameters, user_id)
        elif tool_name == "get_nutrition_summary":
            return await get_nutrition_summary(parameters, user_id)
        elif tool_name == "create_meal_plan":
            return await create_meal_plan(parameters, user_id)
        elif tool_name == "create_recipe":
            return await create_recipe(parameters, user_id)
        elif tool_name == "general_nutrition":
            # General nutrition response with AI
            return await general_nutrition_response(parameters, user_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# =============================================================================
# CRUD ENDPOINTS FOR NUTRITION DATA
# =============================================================================

# Meal Plan CRUD Operations
@app.post("/api/meal-plans", response_model=Dict[str, Any])
async def create_meal_plan_endpoint(meal_plan: MealPlanCreate, user_id: str):
    """Create a new meal plan."""
    try:
        await require_valid_user(user_id)
        
        # Get database session
        db = get_user_db()
        
        # Create meal plan
        db_meal_plan = MealPlan(
            user_id=UUID(user_id),
            name=meal_plan.name,
            description=meal_plan.description,
            start_date=meal_plan.start_date,
            end_date=meal_plan.end_date,
            is_active=meal_plan.is_active,
            is_template=meal_plan.is_template
        )
        
        db.add(db_meal_plan)
        db.commit()
        db.refresh(db_meal_plan)
        
        # Create days if provided
        if meal_plan.days:
            for day_data in meal_plan.days:
                db_day = MealPlanDay(
                    meal_plan_id=db_meal_plan.meal_plan_id,
                    day_number=day_data.day_number,
                    date=day_data.date,
                    notes=day_data.notes
                )
                db.add(db_day)
                db.commit()
                db.refresh(db_day)
                
                # Create meals for this day
                if day_data.meals:
                    for meal_data in day_data.meals:
                        db_meal = MealPlanMeal(
                            day_id=db_day.day_id,
                            meal_type_id=meal_data.meal_type_id,
                            time_of_day=meal_data.time_of_day,
                            notes=meal_data.notes
                        )
                        db.add(db_meal)
                        db.commit()
                        db.refresh(db_meal)
                        
                        # Create items for this meal
                        if meal_data.items:
                            for item_data in meal_data.items:
                                db_item = MealPlanItem(
                                    meal_id=db_meal.meal_id,
                                    food_id=item_data.food_id,
                                    recipe_id=item_data.recipe_id,
                                    quantity=item_data.quantity,
                                    servings=item_data.servings,
                                    notes=item_data.notes
                                )
                                db.add(db_item)
                        
                        db.commit()
        
        db.close()
        
        return {
            "status": "success",
            "message": "Meal plan created successfully",
            "meal_plan_id": str(db_meal_plan.meal_plan_id),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")


@app.get("/api/meal-plans/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_meal_plans(user_id: str, active_only: bool = True):
    """Get meal plans for a user."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        query = db.query(MealPlan).filter(MealPlan.user_id == UUID(user_id))
        if active_only:
            query = query.filter(MealPlan.is_active == True)
        
        meal_plans = query.all()
        
        result = []
        for plan in meal_plans:
            plan_data = {
                "meal_plan_id": str(plan.meal_plan_id),
                "name": plan.name,
                "description": plan.description,
                "start_date": plan.start_date.isoformat(),
                "end_date": plan.end_date.isoformat(),
                "is_active": plan.is_active,
                "is_template": plan.is_template,
                "created_at": plan.created_at.isoformat(),
                "updated_at": plan.updated_at.isoformat()
            }
            result.append(plan_data)
        
        db.close()
        
        return {
            "status": "success",
            "meal_plans": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error getting meal plans: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get meal plans: {str(e)}")


@app.put("/api/meal-plans/{meal_plan_id}", response_model=Dict[str, Any])
async def update_meal_plan(meal_plan_id: str, meal_plan_update: MealPlanUpdate, user_id: str):
    """Update a meal plan."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing meal plan
        db_meal_plan = db.query(MealPlan).filter(
            MealPlan.meal_plan_id == UUID(meal_plan_id),
            MealPlan.user_id == UUID(user_id)
        ).first()
        
        if not db_meal_plan:
            db.close()
            raise HTTPException(status_code=404, detail="Meal plan not found")
        
        # Update fields
        update_data = meal_plan_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_meal_plan, field, value)
        
        db_meal_plan.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Meal plan updated successfully",
            "meal_plan_id": meal_plan_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update meal plan: {str(e)}")


@app.delete("/api/meal-plans/{meal_plan_id}", response_model=Dict[str, Any])
async def delete_meal_plan(meal_plan_id: str, user_id: str):
    """Delete a meal plan."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing meal plan
        db_meal_plan = db.query(MealPlan).filter(
            MealPlan.meal_plan_id == UUID(meal_plan_id),
            MealPlan.user_id == UUID(user_id)
        ).first()
        
        if not db_meal_plan:
            db.close()
            raise HTTPException(status_code=404, detail="Meal plan not found")
        
        # Delete meal plan (cascade will handle related records)
        db.delete(db_meal_plan)
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Meal plan deleted successfully",
            "meal_plan_id": meal_plan_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete meal plan: {str(e)}")


# Food Log CRUD Operations
@app.post("/api/food-logs", response_model=Dict[str, Any])
async def create_food_log(food_log: UserFoodLogCreate, user_id: str):
    """Create a new food log entry."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Create food log
        db_food_log = UserFoodLog(
            user_id=UUID(user_id),
            date=food_log.date,
            meal_type_id=food_log.meal_type_id,
            time_of_day=food_log.time_of_day
        )
        
        db.add(db_food_log)
        db.commit()
        db.refresh(db_food_log)
        
        # Create food log items
        if food_log.items:
            for item_data in food_log.items:
                db_item = UserFoodLogItem(
                    log_id=db_food_log.log_id,
                    food_id=item_data.food_id,
                    recipe_id=item_data.recipe_id,
                    quantity=item_data.quantity,
                    serving=item_data.serving,
                    calories=item_data.calories,
                    protein_g=item_data.protein_g,
                    carbs_g=item_data.carbs_g,
                    fat_g=item_data.fat_g,
                    photo_url=item_data.photo_url,
                    notes=item_data.notes
                )
                db.add(db_item)
            
            db.commit()
        
        db.close()
        
        return {
            "status": "success",
            "message": "Food log created successfully",
            "log_id": str(db_food_log.log_id),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error creating food log: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create food log: {str(e)}")


@app.get("/api/food-logs/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_food_logs(user_id: str, date: Optional[str] = None, meal_type: Optional[str] = None):
    """Get food logs for a user."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        query = db.query(UserFoodLog).filter(UserFoodLog.user_id == UUID(user_id))
        
        if date:
            query = query.filter(UserFoodLog.date == date)
        if meal_type:
            query = query.filter(UserFoodLog.meal_type_id == UUID(meal_type))
        
        food_logs = query.order_by(UserFoodLog.date.desc(), UserFoodLog.time_of_day).all()
        
        result = []
        for log in food_logs:
            log_data = {
                "log_id": str(log.log_id),
                "date": log.date.isoformat(),
                "meal_type_id": str(log.meal_type_id),
                "time_of_day": log.time_of_day.isoformat() if log.time_of_day else None,
                "created_at": log.created_at.isoformat(),
                "updated_at": log.updated_at.isoformat()
            }
            result.append(log_data)
        
        db.close()
        
        return {
            "status": "success",
            "food_logs": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error getting food logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get food logs: {str(e)}")


@app.put("/api/food-logs/{log_id}", response_model=Dict[str, Any])
async def update_food_log(log_id: str, food_log_update: UserFoodLogCreate, user_id: str):
    """Update a food log entry."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing food log
        db_food_log = db.query(UserFoodLog).filter(
            UserFoodLog.log_id == UUID(log_id),
            UserFoodLog.user_id == UUID(user_id)
        ).first()
        
        if not db_food_log:
            db.close()
            raise HTTPException(status_code=404, detail="Food log not found")
        
        # Update fields
        db_food_log.date = food_log_update.date
        db_food_log.meal_type_id = food_log_update.meal_type_id
        db_food_log.time_of_day = food_log_update.time_of_day
        db_food_log.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Food log updated successfully",
            "log_id": log_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating food log: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update food log: {str(e)}")


@app.delete("/api/food-logs/{log_id}", response_model=Dict[str, Any])
async def delete_food_log(log_id: str, user_id: str):
    """Delete a food log entry."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing food log
        db_food_log = db.query(UserFoodLog).filter(
            UserFoodLog.log_id == UUID(log_id),
            UserFoodLog.user_id == UUID(user_id)
        ).first()
        
        if not db_food_log:
            db.close()
            raise HTTPException(status_code=404, detail="Food log not found")
        
        # Delete food log (cascade will handle related records)
        db.delete(db_food_log)
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Food log deleted successfully",
            "log_id": log_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting food log: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete food log: {str(e)}")


# Nutrition Target CRUD Operations
@app.post("/api/nutrition-targets", response_model=Dict[str, Any])
async def create_nutrition_target(target: UserNutritionTargetCreate, user_id: str):
    """Create a new nutrition target."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Check if target already exists for this date
        existing_target = db.query(UserNutritionTarget).filter(
            UserNutritionTarget.user_id == UUID(user_id),
            UserNutritionTarget.date == target.date
        ).first()
        
        if existing_target:
            db.close()
            raise HTTPException(status_code=400, detail="Nutrition target already exists for this date")
        
        # Create nutrition target
        db_target = UserNutritionTarget(
            user_id=UUID(user_id),
            date=target.date,
            calories=target.calories,
            protein_g=target.protein_g,
            carbs_g=target.carbs_g,
            fat_g=target.fat_g,
            fiber_g=target.fiber_g,
            sugar_g=target.sugar_g,
            sodium_mg=target.sodium_mg,
            water_ml=target.water_ml
        )
        
        db.add(db_target)
        db.commit()
        db.refresh(db_target)
        db.close()
        
        return {
            "status": "success",
            "message": "Nutrition target created successfully",
            "target_id": str(db_target.target_id),
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating nutrition target: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create nutrition target: {str(e)}")


@app.get("/api/nutrition-targets/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_nutrition_targets(user_id: str, date: Optional[str] = None):
    """Get nutrition targets for a user."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        query = db.query(UserNutritionTarget).filter(UserNutritionTarget.user_id == UUID(user_id))
        
        if date:
            query = query.filter(UserNutritionTarget.date == date)
        
        targets = query.order_by(UserNutritionTarget.date.desc()).all()
        
        result = []
        for target in targets:
            target_data = {
                "target_id": str(target.target_id),
                "date": target.date.isoformat(),
                "calories": target.calories,
                "protein_g": float(target.protein_g) if target.protein_g else None,
                "carbs_g": float(target.carbs_g) if target.carbs_g else None,
                "fat_g": float(target.fat_g) if target.fat_g else None,
                "fiber_g": float(target.fiber_g) if target.fiber_g else None,
                "sugar_g": float(target.sugar_g) if target.sugar_g else None,
                "sodium_mg": target.sodium_mg,
                "water_ml": target.water_ml,
                "created_at": target.created_at.isoformat(),
                "updated_at": target.updated_at.isoformat()
            }
            result.append(target_data)
        
        db.close()
        
        return {
            "status": "success",
            "nutrition_targets": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error getting nutrition targets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrition targets: {str(e)}")


@app.put("/api/nutrition-targets/{target_id}", response_model=Dict[str, Any])
async def update_nutrition_target(target_id: str, target_update: UserNutritionTargetUpdate, user_id: str):
    """Update a nutrition target."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing target
        db_target = db.query(UserNutritionTarget).filter(
            UserNutritionTarget.target_id == UUID(target_id),
            UserNutritionTarget.user_id == UUID(user_id)
        ).first()
        
        if not db_target:
            db.close()
            raise HTTPException(status_code=404, detail="Nutrition target not found")
        
        # Update fields
        update_data = target_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_target, field, value)
        
        db_target.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Nutrition target updated successfully",
            "target_id": target_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating nutrition target: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update nutrition target: {str(e)}")


@app.delete("/api/nutrition-targets/{target_id}", response_model=Dict[str, Any])
async def delete_nutrition_target(target_id: str, user_id: str):
    """Delete a nutrition target."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get existing target
        db_target = db.query(UserNutritionTarget).filter(
            UserNutritionTarget.target_id == UUID(target_id),
            UserNutritionTarget.user_id == UUID(user_id)
        ).first()
        
        if not db_target:
            db.close()
            raise HTTPException(status_code=404, detail="Nutrition target not found")
        
        # Delete target
        db.delete(db_target)
        db.commit()
        db.close()
        
        return {
            "status": "success",
            "message": "Nutrition target deleted successfully",
            "target_id": target_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting nutrition target: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete nutrition target: {str(e)}")


# Nutrition Summary and Stats
@app.get("/api/nutrition-summary/{user_id}", response_model=Dict[str, Any])
async def get_nutrition_summary_endpoint(user_id: str, date: Optional[str] = None):
    """Get nutrition summary for a user."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Get target date
        target_date = date if date else datetime.now().date()
        
        # Get nutrition target for this date
        target = db.query(UserNutritionTarget).filter(
            UserNutritionTarget.user_id == UUID(user_id),
            UserNutritionTarget.date == target_date
        ).first()
        
        # Get food logs for this date
        food_logs = db.query(UserFoodLog).filter(
            UserFoodLog.user_id == UUID(user_id),
            UserFoodLog.date == target_date
        ).all()
        
        # Calculate totals from food logs
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for log in food_logs:
            log_items = db.query(UserFoodLogItem).filter(UserFoodLogItem.log_id == log.log_id).all()
            for item in log_items:
                if item.calories:
                    total_calories += item.calories
                if item.protein_g:
                    total_protein += float(item.protein_g)
                if item.carbs_g:
                    total_carbs += float(item.carbs_g)
                if item.fat_g:
                    total_fat += float(item.fat_g)
        
        # Calculate remaining calories
        target_calories = target.calories if target else 2000  # Default
        calories_remaining = target_calories - total_calories
        
        summary = {
            "date": target_date.isoformat(),
            "target_calories": target_calories,
            "total_calories": total_calories,
            "calories_remaining": calories_remaining,
            "total_protein_g": round(total_protein, 2),
            "total_carbs_g": round(total_carbs, 2),
            "total_fat_g": round(total_fat, 2),
            "target_met_percentage": round((total_calories / target_calories) * 100, 1) if target_calories > 0 else 0
        }
        
        db.close()
        
        return {
            "status": "success",
            "nutrition_summary": summary,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting nutrition summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrition summary: {str(e)}")


@app.get("/api/nutrition-stats/{user_id}/weekly", response_model=Dict[str, Any])
async def get_weekly_nutrition_stats(user_id: str, week_start: Optional[str] = None):
    """Get weekly nutrition statistics for a user."""
    try:
        await require_valid_user(user_id)
        
        db = get_user_db()
        
        # Calculate week start date
        if week_start:
            start_date = datetime.strptime(week_start, "%Y-%m-%d").date()
        else:
            today = datetime.now().date()
            start_date = today - timedelta(days=today.weekday())
        
        end_date = start_date + timedelta(days=6)
        
        # Get nutrition data for the week
        daily_stats = []
        weekly_totals = {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0
        }
        
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            
            # Get daily summary
            daily_summary = await get_nutrition_summary_endpoint(user_id, current_date.isoformat())
            daily_stats.append(daily_summary["nutrition_summary"])
            
            # Add to weekly totals
            weekly_totals["calories"] += daily_summary["nutrition_summary"]["total_calories"]
            weekly_totals["protein_g"] += daily_summary["nutrition_summary"]["total_protein_g"]
            weekly_totals["carbs_g"] += daily_summary["nutrition_summary"]["total_carbs_g"]
            weekly_totals["fat_g"] += daily_summary["nutrition_summary"]["total_fat_g"]
        
        # Calculate weekly averages
        weekly_averages = {
            "calories": round(weekly_totals["calories"] / 7, 1),
            "protein_g": round(weekly_totals["protein_g"] / 7, 2),
            "carbs_g": round(weekly_totals["carbs_g"] / 7, 2),
            "fat_g": round(weekly_totals["fat_g"] / 7, 2)
        }
        
        db.close()
        
        return {
            "status": "success",
            "weekly_stats": {
                "week_start": start_date.isoformat(),
                "week_end": end_date.isoformat(),
                "daily_stats": daily_stats,
                "weekly_averages": weekly_averages,
                "weekly_totals": weekly_totals
            },
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting weekly nutrition stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get weekly nutrition stats: {str(e)}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=get_host(), 
        port=get_port(),
        log_level=get_log_level().lower()
    )
