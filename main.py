#!/usr/bin/env python3
"""
Nutrition Agent - OpenAI AI Integration with Standardized Architecture

This agent provides intelligent nutrition functionality using OpenAI AI.
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from celery.result import AsyncResult
import requests

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
    get_redis_url, get_openai_api_key, get_environment, get_log_level,
    get_port, get_host, get_cors_origins,
    get_openai_model, get_openai_timeout, get_openai_max_tokens
)

# Import nutrition-specific batching utilities
from utils.batching import NutritionBatchManager, NutritionCache, NutritionOptimizer
from utils.batching.nutrition_batch_manager import NutritionPriority, NutritionBatchStrategy

# Import legacy batching utilities for backward compatibility
from utils.ai_batching import AIBatchManager, AICall, AIResponse
from utils.ai_cache import AIResponseCache

# Note: Advanced utilities removed - using agent-specific batching instead
# Each agent now has its own optimized batching utilities

# Import models and schemas
from models import (
    User, MealPlan, MealPlanDay, MealPlanMeal, MealPlanItem,
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
# JSON PARSING UTILITIES
# =============================================================================

def simple_json_parse(ai_response: str) -> Dict[str, Any]:
    """
    Enhanced JSON parsing function that handles markdown code blocks and extra text.
    Tries multiple strategies to extract valid JSON from AI responses.
    """
    try:
        import json
        import re
        
        # Strategy 1: Try direct parsing
        try:
            parsed = json.loads(ai_response)
            return parsed
        except:
            pass
        
        # Strategy 2: Remove markdown code blocks
        # Remove ```json ... ``` or ``` ... ```
        cleaned = ai_response.strip()
        if cleaned.startswith("```"):
            # Find the first { and last }
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]
        
        # Try parsing after removing markdown
        try:
            parsed = json.loads(cleaned)
            return parsed
        except:
            pass
        
        # Strategy 3: Extract JSON using regex
        # Look for JSON object pattern
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, ai_response, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # If it parses successfully and looks like our expected structure
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except:
                continue
        
        # Strategy 4: Try to find and extract the largest JSON-like structure
        start_idx = ai_response.find("{")
        end_idx = ai_response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            potential_json = ai_response[start_idx:end_idx + 1]
            try:
                parsed = json.loads(potential_json)
                return parsed
            except:
                pass
        
        # If all strategies fail, return error with raw text
        logger.warning(f"Failed to parse AI response as JSON after all strategies")
        # Return raw AI response for supervisor to process
        return {
            "response": ai_response,
            "format_issue": True,
            "raw_ai_response": True
        }
        
    except Exception as e:
        logger.error(f"Critical error in JSON parsing: {e}")
        # Return raw AI response for supervisor to process
        return {
            "response": ai_response,
            "format_issue": True,
            "raw_ai_response": True
        }

# =============================================================================
# SIMPLE NUTRITION AGENT
# =============================================================================

# Simple in-memory storage (for demo purposes)
nutrition_data = {}

# Initialize AI clients
openai_client = None

# Initialize nutrition-specific batching and caching
nutrition_batch_manager = NutritionBatchManager(
    max_batch_size=6,  # Reduced for faster processing
    max_wait_time=1.0,  # Increased wait time for better batching
    strategy=NutritionBatchStrategy.BALANCED,
    enable_dietary_priority=True,
    enable_nutrition_optimization=True
)
nutrition_cache = NutritionCache(max_size=800, enable_nutrition_optimization=True)
nutrition_optimizer = NutritionOptimizer(enable_ml_optimization=True)

# Initialize legacy batching and caching (backward compatibility)
batch_manager = AIBatchManager(max_batch_size=5, max_wait_time=0.5)
ai_cache = AIResponseCache()

# Note: Advanced utilities removed - using agent-specific batching instead
# Each agent now has its own optimized batching utilities

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
    """Initialize AI clients on startup using direct HTTP requests (OpenAI only)."""
    global openai_client
    
    try:
        # Initialize OpenAI API key
        openai_api_key = get_openai_api_key()
        
        if not openai_api_key:
            logger.error("❌ OPENAI_API_KEY not set, AI nutrition features will not work")
            openai_client = None
            return
        
        try:
            # Test connection with a simple request using HTTP API
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            
            test_data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
            
            test_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=test_data,
                timeout=15
            )
            
            if test_response.status_code == 200:
                logger.info("✅ OpenAI connection test successful")
                openai_client = openai_api_key  # Store API key, not client object
            else:
                logger.error(f"❌ OpenAI test failed with status {test_response.status_code}")
                openai_client = None
                
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            openai_client = None
            
    except Exception as e:
        logger.error(f"AI initialization error: {e}")
        openai_client = None

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
        verified_users = ["e589fb61-6cd0-4ccd-8a9c-ee8a8dd85e8a"]
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
# AI HELPER FUNCTIONS
# =============================================================================

async def get_ai_response(prompt: str, max_tokens: int = 16000, temperature: float = 0.7, 
                         function_name: str = "", user_id: str = "", use_batching: bool = True) -> tuple[str, str]:
    """Get AI response using GPT-4o with nutrition-specific batching support."""
    
    # Check nutrition-specific cache first
    if use_batching:
        cached_response = nutrition_cache.get(prompt, max_tokens, temperature, function_name, "nutrition_analysis")
        if cached_response:
            logger.info(f"Nutrition cache hit for {function_name}")
            return cached_response.get("content", ""), cached_response.get("model", "cached")
    
    # Define the AI client function for batching
    async def ai_client_func(prompt: str, max_tokens: int, temperature: float) -> tuple[str, str]:
        # Use OpenAI HTTP API directly
        if openai_client:
            try:
                headers = {
                    "Authorization": f"Bearer {openai_client}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 16000
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"], "gpt-4o"
                else:
                    logger.error(f"OpenAI request failed with status {response.status_code}")
            except Exception as e:
                logger.error(f"OpenAI request failed: {e}")
        
        return "AI features are currently unavailable. Please try again later.", "unavailable"
    
    # Use nutrition-specific batching if enabled
    if use_batching:
        try:
            response, model = await nutrition_batch_manager.get_nutrition_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                function_name=function_name,
                user_id=user_id,
                priority=NutritionPriority.HIGH,  # High priority for nutrition functions
                use_cache=True,
                ai_client_func=ai_client_func,
                nutrition_type="meal_plan",
                dietary_restrictions=False
            )
            
            # Cache the response in nutrition-specific cache
            nutrition_cache.set(prompt, {"content": response, "model": model}, max_tokens, temperature, function_name, "meal_plan")
            
            return response, model
        except Exception as e:
            logger.error(f"Nutrition batching failed, falling back to legacy batching: {e}")
            # Fallback to legacy batching
            try:
                response, model = await batch_manager.get_ai_response(
                    prompt=prompt,
                    function_name=function_name,
                    user_id=user_id,
                    priority=1,
                    use_cache=True,
                    ai_client_func=ai_client_func
                )
                
                # Cache the response
                ai_cache.set(prompt, {"content": response, "model": model}, max_tokens, temperature, function_name)
                
                return response, model
            except Exception as e2:
                logger.error(f"Legacy batching also failed, falling back to direct OpenAI call: {e2}")
                
                # Final fallback: direct OpenAI HTTP API call
                try:
                    headers = {
                        "Authorization": f"Bearer {openai_client}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result["choices"][0]["message"]["content"]
                        model = "gpt-4o"
                    else:
                        raise Exception(f"OpenAI API error: {response.status_code}")
                    
                    # Cache the response
                    if use_batching:
                        nutrition_cache.set(prompt, {"content": ai_response, "model": model}, max_tokens, temperature, function_name, "meal_plan")
                    
                    return ai_response, model
                except Exception as direct_error:
                    logger.error(f"Direct OpenAI call also failed: {direct_error}")
                    raise Exception(f"All AI methods failed: nutrition_batch={e}, legacy_batch={e2}, direct_openai={direct_error}")
    
    # Fallback to direct call
    try:
        response, model = await ai_client_func(prompt, max_tokens, temperature)
        
        # Cache the response
        if use_batching:
            nutrition_cache.set(prompt, {"content": response, "model": model}, max_tokens, temperature, function_name, "meal_plan")
        
        return response, model
    except Exception as direct_error:
        logger.error(f"Direct ai_client_func call failed: {direct_error}")
        raise Exception(f"All AI methods failed: {direct_error}")

# Note: Advanced AI response functions removed - using agent-specific batching instead

# Note: Groq response function removed - using OpenAI only

# =============================================================================
# NUTRITION FUNCTIONS WITH AI
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
        
        # Add AI nutrition insights if OpenAI is available
        if openai_client:
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

IMPORTANT: The user's description takes priority over individual parameters. If the description specifies different details about the meal (e.g., "large portion" or "high protein meal"), use those values instead of the parameters above. Only use the parameters when they align with the description or when the description is unclear.

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
          "unit": "g"
        }},
        {{
          "nutrient": "Fiber",
          "amount": 8,
          "unit": "g"
        }},
        {{
          "nutrient": "Vitamin A",
          "amount": 1200,
          "unit": "mcg"
        }},
        {{
          "nutrient": "Vitamin C",
          "amount": 45,
          "unit": "mg"
        }},
        {{
          "nutrient": "Iron",
          "amount": 3.5,
          "unit": "mg"
        }},
        {{
          "nutrient": "Calcium",
          "amount": 120,
          "unit": "mg"
        }},
        {{
          "nutrient": "Potassium",
          "amount": 600,
          "unit": "mg"
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

IMPORTANT: Return ONLY a valid JSON object. Do not include markdown, explanations, or extra text. JSON must start with '{' and end with '}' with no other content.

Rules:
1. Provide realistic calorie and macro estimates based on the food items
2. Include specific, actionable recommendations
3. Make it supportive and educational
4. Base analysis on the actual meal details provided
5. Always include serving_info with serving_size and quantity
6. Expand nutrients_summary to include important micronutrients beyond just macros
7. Focus on essential nutrient information"""

                ai_insights, _ = await get_ai_response(
                    insight_prompt, 
                    function_name="log_meal",
                    user_id=user_id
                )
                
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")
                ai_insights = "AI nutrition analysis temporarily unavailable. Please try again later."
        else:
            ai_insights = "AI nutrition insights are currently unavailable."
        
        # Simple and reliable JSON parsing
        try:
            import json
            
            # Try to parse the AI response directly
            parsed = json.loads(ai_insights)
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse AI meal analysis as JSON: {e}")
            logger.warning(f"AI response content: {ai_insights[:500]}...")
            # Fallback to original response
            return {
                "response": ai_insights,
                "format_issue": True,
                "raw_ai_response": True
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
        
        # Generate AI-powered summary if OpenAI is available
        if openai_client:
            try:
                summary_prompt = f"""Based on this user request, provide a comprehensive nutrition summary:
- User request: {description}
- Time period: {days} days
- User goals: {goals}

IMPORTANT: The user's description takes priority over individual parameters. If the description specifies different requirements (e.g., "focus on protein" or "quick summary"), use those values instead of the parameters above. Only use the parameters when they align with the description or when the description is unclear.

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
          "food_sources": "Lean meats, fish, eggs, legumes"
        }},
        {{
          "nutrient": "Fiber",
          "daily_target": "25-35g",
          "food_sources": "Whole grains, fruits, vegetables, legumes"
        }},
        {{
          "nutrient": "Vitamin D",
          "daily_target": "15-20mcg",
          "food_sources": "Fatty fish, egg yolks, fortified dairy"
        }},
        {{
          "nutrient": "Iron",
          "daily_target": "8-18mg",
          "food_sources": "Red meat, spinach, legumes, fortified cereals"
        }},
        {{
          "nutrient": "Calcium",
          "daily_target": "1000-1300mg",
          "food_sources": "Dairy products, leafy greens, fortified foods"
        }},
        {{
          "nutrient": "Omega-3",
          "daily_target": "1.1-1.6g",
          "food_sources": "Fatty fish, flaxseeds, walnuts, chia seeds"
        }},
        {{
          "nutrient": "B Vitamins",
          "daily_target": "Various",
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

IMPORTANT: Return ONLY a valid JSON object. Do not include markdown, explanations, or extra text. JSON must start with '{' and end with '}' with no other content.

Make it personalized and actionable based on their description and goals."""

                ai_summary, _ = await get_ai_response(
                    summary_prompt, 
                    function_name="get_nutrition_summary",
                    user_id=user_id
                )
                
            except Exception as e:
                logger.warning(f"Failed to generate AI summary: {e}")
                ai_summary = "AI summary generation failed, but here's your basic summary."
        else:
            ai_summary = "AI features are currently unavailable."
        
        # Simple and reliable JSON parsing
        try:
            import json
            
            # Try to parse the AI response directly
            parsed = json.loads(ai_summary)
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse AI nutrition summary as JSON: {e}")
            logger.warning(f"AI response content: {ai_summary[:500]}...")
            # Fallback to original response
        return {
                "response": ai_summary,
                "format_issue": True,
                "raw_ai_response": True
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
        days_per_week = int(parameters.get("days_per_week", 5))  # 1-7 days, default 5
        meals_per_day = int(parameters.get("meals_per_day", 3))  # 1-5 meals, default 3
        dietary_restrictions = parameters.get("dietary_restrictions", [])  # vegetarian, vegan, gluten-free, etc.
        calorie_target = int(parameters.get("calorie_target", 0))  # 0 means no specific target
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
        
        # Generate AI-powered meal plan if OpenAI is available
        if openai_client:
            try:
                # Build detailed prompt based on parameters
                if isinstance(dietary_restrictions, list):
                    restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                else:
                    restrictions_text = dietary_restrictions if dietary_restrictions else "none"
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
                
                meal_prompt = f"""Create a {days_per_week}-day meal plan with {meals_per_day} meals per day ({meal_types_text}).

{description if description else ''}
Dietary: {restrictions_text}. Calories: {calorie_text}. Cuisine: {cuisine_preference}.

🚨 CRITICAL: STRICT NUMBER ADHERENCE REQUIRED 🚨
- If user specifies "{calorie_target} calories", you MUST stick to EXACTLY {calorie_target} calories per day
- If user says "200 calories", provide EXACTLY 200 calories, not 180 or 220
- If user says "1800 calories", provide EXACTLY 1800 calories, not 1750 or 1850
- If user specifies protein amounts (e.g., "150g protein"), stick to EXACTLY that amount
- If user specifies meal counts (e.g., "3 meals"), provide EXACTLY that number
- NEVER approximate or round user-specified numbers - use them EXACTLY as given
- Only use estimates when user doesn't specify exact numbers

Respond with JSON only:
{{
  "meal_plan": {{
    "days": [
      {{
        "day": 1,
        "meals": {{
          "breakfast": {{"name": "Meal", "calories": 400, "macros": {{"protein": 20, "carbs": 50, "fat": 12}}, "ingredients": ["item1", "item2"]}},
          "lunch": {{"name": "Meal", "calories": 500, "macros": {{"protein": 35, "carbs": 45, "fat": 18}}, "ingredients": ["item1", "item2"]}},
          "dinner": {{"name": "Meal", "calories": 600, "macros": {{"protein": 40, "carbs": 55, "fat": 20}}, "ingredients": ["item1", "item2"]}}
        }}
      }}
    ],
    "grocery_list": {{
      "proteins": ["chicken breast", "salmon", "eggs", "greek yogurt"],
      "vegetables": ["broccoli", "spinach", "bell peppers", "carrots"],
      "grains": ["brown rice", "quinoa", "oatmeal"],
      "dairy": ["milk", "cheese", "butter"],
      "pantry_items": ["olive oil", "spices", "herbs"],
      "total_estimated_cost": "$45-65"
    }}
  }}
}}

Include {days_per_week} days with {meals_per_day} meals each.

🚨 FINAL REMINDER: Use EXACT numbers specified by the user. If they say "{calorie_target} calories", provide EXACTLY {calorie_target} calories per day.

IMPORTANT: Always include a comprehensive grocery_list with categorized items and estimated cost. If the user specifically asks for a grocery list, make it detailed and practical for shopping."""

                # Use direct OpenAI HTTP API call to avoid Heroku 30s timeout with batching
                logger.info("Using direct OpenAI HTTP API for create_meal_plan to avoid timeout")
                try:
                    headers = {
                        "Authorization": f"Bearer {openai_client}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": meal_prompt}],
                        "temperature": 0.7,
                        "max_tokens": 12000
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=20  # Reduced timeout to prevent cascading timeouts
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_meal_plan = result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"OpenAI API returned status {response.status_code}: {response.text}")
                        raise Exception(f"OpenAI API error: {response.status_code}")
                except Exception as openai_error:
                    logger.error(f"Direct OpenAI HTTP API call failed: {openai_error}")
                    raise
                
            except Exception as e:
                logger.warning(f"Failed to generate AI meal plan: {e}")
                ai_meal_plan = "AI meal planning temporarily unavailable. Please try again later."
        else:
            ai_meal_plan = "AI meal planning features are currently unavailable."
        
        # Enhanced JSON parsing with markdown stripping
        try:
            # Use enhanced JSON parser that handles markdown blocks
            parsed = simple_json_parse(ai_meal_plan)
            logger.info(f"JSON parsing successful. Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
            
            # Check if parsing returned an error
            if isinstance(parsed, dict) and "error" in parsed:
                logger.warning(f"JSON parser returned error: {parsed.get('error')}")
                return parsed
            
            # Step 2: Verify ingredients and calculate accurate macros using nutrition database
            # COMMENTED OUT TO PREVENT HEROKU TIMEOUT
            # try:
            #     # Import database service with error handling
            #     try:
            #         from utils.nutrition_database_service import nutrition_db_service
            #         db_service_available = True
            #     except ImportError as import_error:
            #         logger.warning(f"Nutrition database service not available: {import_error}")
            #         db_service_available = False
            #     
            #     # Extract ingredients from the meal plan
            #     all_ingredients = []
            #     if isinstance(parsed, dict) and "meal_plan" in parsed:
            #         meal_plan = parsed["meal_plan"]
            #         if "days" in meal_plan:
            #             for day in meal_plan["days"]:
            #                 if "meals" in day:
            #                     for meal_type, meal in day["meals"].items():
            #                         if isinstance(meal, dict) and "ingredients" in meal:
            #                             all_ingredients.extend(meal["ingredients"])
            #     
            #     # Verify ingredients in database and calculate accurate macros
            #     if all_ingredients and db_service_available:
            #         # Clean ingredient names (remove quantities, keep just the food name)
            #         cleaned_ingredients = []
            #         for ingredient in all_ingredients:
            #             # Extract food name from ingredient string (e.g., "150g grilled chicken breast" -> "grilled chicken breast")
            #             import re
            #             # Remove common quantity patterns
            #             cleaned = re.sub(r'^\d+(?:\.\d+)?\s*(?:g|kg|oz|lb|cup|tbsp|tsp|ml|l|tablespoon|teaspoon|gram|ounce|pound|milliliter|liter)\s*', '', ingredient, flags=re.IGNORECASE)
            #             cleaned = cleaned.strip()
            #             if cleaned:
            #                 cleaned_ingredients.append(cleaned)
            #         
            #         # Search for ingredients in nutrition database
            #         try:
            #             verified_ingredients = nutrition_db_service.search_ingredients_for_meal(cleaned_ingredients)
            #             
            #             # Calculate accurate macros for the meal plan
            #             if verified_ingredients:
            #                 # Update meal plan with verified nutrition data
            #                 parsed["meal_plan"]["nutrition_verification"] = {
            #                     "ingredients_verified": len(verified_ingredients),
            #                     "total_ingredients": len(cleaned_ingredients),
            #                     "verification_status": "partial" if len(verified_ingredients) < len(cleaned_ingredients) else "complete",
            #                     "database_used": True
            #                 }
            #                 
            #                 # Add verified ingredients data
            #                 parsed["meal_plan"]["verified_ingredients"] = verified_ingredients
            #                 
            #                 logger.info(f"Verified {len(verified_ingredients)} out of {len(cleaned_ingredients)} ingredients in nutrition database")
            #             else:
            #                 parsed["meal_plan"]["nutrition_verification"] = {
            #                     "ingredients_verified": 0,
            #                     "total_ingredients": len(cleaned_ingredients),
            #                     "verification_status": "none_found",
            #                     "database_used": True,
            #                     "note": "No ingredients found in nutrition database, using AI-generated nutrition data"
            #                 }
            #                 logger.info("No ingredients found in nutrition database, using AI-generated data")
            #         except Exception as db_call_error:
            #             logger.warning(f"Database service call failed: {db_call_error}")
            #             parsed["meal_plan"]["nutrition_verification"] = {
            #                 "ingredients_verified": 0,
            #                 "total_ingredients": len(cleaned_ingredients),
            #                 "verification_status": "database_error",
            #                 "database_used": False,
            #                 "error": str(db_call_error)
            #             }
            #     elif all_ingredients and not db_service_available:
            #         parsed["meal_plan"]["nutrition_verification"] = {
            #             "ingredients_verified": 0,
            #             "total_ingredients": len(all_ingredients),
            #             "verification_status": "service_unavailable",
            #             "database_used": False,
            #             "note": "Nutrition database service not available, using AI-generated nutrition data"
            #         }
            #     else:
            #         parsed["meal_plan"]["nutrition_verification"] = {
            #             "ingredients_verified": 0,
            #             "total_ingredients": 0,
            #             "verification_status": "no_ingredients",
            #             "database_used": False,
            #             "note": "No ingredients found in meal plan"
            #         }
            #         
            # except Exception as db_error:
            #     logger.warning(f"Failed to verify ingredients in nutrition database: {db_error}")
            #     # Continue with AI-generated data if database verification fails
            #     if isinstance(parsed, dict) and "meal_plan" in parsed:
            #         parsed["meal_plan"]["nutrition_verification"] = {
            #             "ingredients_verified": 0,
            #             "total_ingredients": 0,
            #             "verification_status": "database_error",
            #             "database_used": False,
            #             "error": str(db_error)
            #         }
            
            logger.info("Database verification disabled to prevent timeouts - using AI-generated nutrition data only")
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse AI meal plan as JSON: {e}")
            logger.warning(f"AI response content: {ai_meal_plan[:500]}...")
            # Return raw AI response for supervisor to process
            return {
                "response": ai_meal_plan,
                "format_issue": True,
                "raw_ai_response": True
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        # Return a proper JSON error response instead of raising HTTPException
        return {
            "error": "Failed to create meal plan",
            "message": str(e),
            "status": "error"
        }

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
        
        # Generate AI-powered meal if OpenAI is available
        if openai_client:
            try:
                # Build detailed prompt based on parameters
                if isinstance(dietary_restrictions, list):
                    restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                else:
                    restrictions_text = dietary_restrictions if dietary_restrictions else "none"
                calorie_text = f"{calorie_target} calories" if calorie_target > 0 else "no specific calorie target"
                
                meal_prompt = f"""Create a {meal_type} meal. {description if description else ''}
Dietary: {restrictions_text}. Calories: {calorie_text}. Cuisine: {cuisine_preference}.

🚨 CRITICAL: STRICT NUMBER ADHERENCE REQUIRED 🚨
- If user specifies "{calorie_target} calories", you MUST stick to EXACTLY {calorie_target} calories
- If user says "200 calories", provide EXACTLY 200 calories, not 180 or 220
- If user says "500 calories", provide EXACTLY 500 calories, not 480 or 520
- If user specifies protein amounts (e.g., "30g protein"), stick to EXACTLY that amount
- NEVER approximate or round user-specified numbers - use them EXACTLY as given
- Only use estimates when user doesn't specify exact numbers

Respond with JSON only:
{{
  "meal": {{
      "name": "Meal Name",
    "calories": 500,
    "macros": {{"protein": 30, "carbs": 45, "fat": 18}},
    "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
    "instructions": ["step1", "step2", "step3"]
  }}
}}"""
                
                # Use direct OpenAI HTTP API call to avoid Heroku 30s timeout with batching
                logger.info("Using direct OpenAI HTTP API for create_meal to avoid timeout")
                try:
                    headers = {
                        "Authorization": f"Bearer {openai_client}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": meal_prompt}],
                        "temperature": 0.7,
                        "max_tokens": 12000
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=20  # Reduced timeout to prevent cascading timeouts
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_meal = result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"OpenAI API returned status {response.status_code}: {response.text}")
                        raise Exception(f"OpenAI API error: {response.status_code}")
                except Exception as openai_error:
                    logger.error(f"Direct OpenAI HTTP API call failed: {openai_error}")
                    raise
                
            except Exception as e:
                logger.warning(f"Failed to generate AI meal: {e}")
                ai_meal = "AI meal creation temporarily unavailable. Please try again later."
        else:
            ai_meal = "AI meal creation features are currently unavailable."
        
        # Enhanced JSON parsing with markdown stripping
        parsed = simple_json_parse(ai_meal)
        return parsed
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meal: {e}")
        # Return a proper JSON error response instead of raising HTTPException
        return {
            "error": "Failed to create meal",
            "message": str(e),
            "status": "error"
        }

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
        
        if openai_client:
            try:
                if isinstance(dietary_restrictions, list):
                    restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
                else:
                    restrictions_text = dietary_restrictions if dietary_restrictions else "none"
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

IMPORTANT: The user's description takes priority over individual parameters. If the description specifies different requirements (e.g., "quick recipe" or "high protein"), use those values instead of the parameters above. Only use the parameters when they align with the description or when the description is unclear.

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
          "unit": "g"
        }},
        {{
          "nutrient": "Fiber",
          "amount": 8,
          "unit": "g"
        }},
        {{
          "nutrient": "Iron",
          "amount": 3.2,
          "unit": "mg"
        }},
        {{
          "nutrient": "Vitamin C",
          "amount": 28,
          "unit": "mg"
        }},
        {{
          "nutrient": "Folate",
          "amount": 85,
          "unit": "mcg"
        }},
        {{
          "nutrient": "Potassium",
          "amount": 420,
          "unit": "mg"
        }},
        {{
          "nutrient": "Calcium",
          "amount": 95,
          "unit": "mg"
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

IMPORTANT: Return ONLY a valid JSON object. Do not include markdown, explanations, or extra text. JSON must start with '{' and end with '}' with no other content.

Rules:
1. Make the recipe practical and delicious
2. Include realistic cooking times and difficulty levels
3. Provide accurate nutritional estimates
4. Include helpful tips and variations
5. Consider the dietary restrictions specified
6. Always include serving_info with serving_size, quantity, and portion_description
7. Expand nutrients_summary to include important micronutrients beyond just macros
8. Focus on essential nutrient information
9. Add quantity and notes to ingredients for better clarity"""
                
                ai_recipe, _ = await get_ai_response(
                    recipe_prompt, 
                    max_tokens=get_openai_max_tokens(), 
                    function_name="create_recipe",
                    user_id=user_id
                )
                
            except Exception as e:
                logger.warning(f"Failed to generate AI recipe: {e}")
                ai_recipe = "AI recipe creation temporarily unavailable. Please try again later."
        else:
            ai_recipe = "AI recipe creation features are currently unavailable."
        
        # Simple and reliable JSON parsing
        try:
            import json
            
            # Try to parse the AI response directly
            parsed = json.loads(ai_recipe)
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse AI recipe as JSON: {e}")
            logger.warning(f"AI response content: {ai_recipe[:500]}...")
            # Fallback to original response
            return {
                "response": ai_recipe,
                "format_issue": True,
                "raw_ai_response": True
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating recipe: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create recipe: {str(e)}")

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Nutrition Agent - OpenAI AI Integration",
    description="Intelligent nutrition agent with OpenAI AI for personalized meal planning and nutrition advice",
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

@app.post("/test-gpt4o")
async def test_gpt4o_direct(prompt: str = Body(..., embed=True)):
    """Test GPT-4o directly without batching using HTTP API - for troubleshooting."""
    start_time = time.time()
    
    try:
        logger.info(f"Testing GPT-4o with prompt: {prompt[:100]}...")
        
        # Check if OpenAI client is available
        if not openai_client:
            return {
                "status": "error",
                "error": "OpenAI API key not initialized",
                "duration_seconds": 0,
                "test_type": "direct_http_call",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Direct OpenAI HTTP API call - no SDK, no batching, no processing overhead
        # Using GPT-4o for test endpoint (faster and more reliable)
        headers = {
            "Authorization": f"Bearer {openai_client}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 16000
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20  # Reduced timeout to prevent cascading timeouts
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.info(f"GPT-4o direct test completed in {duration:.2f}s")
            
            return {
                "status": "success",
                "response": content,
                "response_length": len(content),
                "duration_seconds": round(duration, 2),
                "model": "gpt-4o",
                "test_type": "direct_http_call",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            logger.error(f"OpenAI API error {response.status_code}: {response.text}")
            return {
                "status": "error",
                "error": f"OpenAI API returned status {response.status_code}",
                "error_details": response.text[:500],
                "duration_seconds": round(duration, 2),
                "test_type": "direct_http_call",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"GPT-4o direct test failed: {e}", exc_info=True)
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "duration_seconds": round(duration, 2),
            "test_type": "direct_http_call",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/health")
async def health_check():
    """Comprehensive health check with all service statuses."""
    try:
        # Get database status using utils
        db_status = get_database_status()
        
        # Check AI status
        openai_status = "connected" if openai_client else "disconnected"
        
        # Get nutrition-specific batching statistics
        nutrition_batching_stats = nutrition_batch_manager.get_nutrition_stats()
        nutrition_cache_stats = nutrition_cache.get_stats()
        nutrition_optimizer_stats = nutrition_optimizer.get_stats()
        
        # Get legacy batching statistics (for backward compatibility)
        batching_stats = batch_manager.get_stats()
        cache_stats = ai_cache.get_stats()
        
        # Note: Advanced statistics removed - using agent-specific batching instead
        
        # Build services status
        services_status = {
            "openai_ai": openai_status,
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
            "nutrition_batching": nutrition_batching_stats,
            "nutrition_caching": nutrition_cache_stats,
            "nutrition_optimization": nutrition_optimizer_stats,
            "legacy_batching": batching_stats,
            "legacy_caching": cache_stats,
            "message": "Nutrition agent is running smoothly with nutrition-specific batching enabled"
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
        elif tool_name == "create_meal":
            return await create_meal(parameters, user_id)
        elif tool_name == "create_recipe":
            return await create_recipe(parameters, user_id)
        # Note: general_nutrition tool removed - handled by supervisor agent
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
# NUTRITION DATABASE ENDPOINTS
# =============================================================================

@app.get("/nutrition-database/status")
async def nutrition_database_status():
    """Get nutrition database connection status and available foods count."""
    try:
        from utils.nutrition_database_service import nutrition_db_service
        
        status = nutrition_db_service.get_database_status()
        
        return {
            "status": "success",
            "nutrition_database": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking nutrition database status: {e}")
        return {
            "status": "error",
            "nutrition_database": {
                "status": "error",
                "message": f"Status check failed: {str(e)}",
                "available": False
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/nutrition-database/foods")
async def get_available_foods(
    query: Optional[str] = Query(None, min_length=2, description="Search query for food name"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of foods to return")
):
    """Get available foods from nutrition database with optional search."""
    try:
        from utils.nutrition_database_service import nutrition_db_service
        
        if query:
            foods = nutrition_db_service.search_foods_by_name(query, limit=limit)
        else:
            # Get a sample of foods if no query provided
            foods = nutrition_db_service.search_foods_by_name("chicken", limit=limit)
        
        return {
            "status": "success",
            "foods": foods,
            "total_found": len(foods),
            "query": query,
            "limit": limit,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available foods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get foods: {str(e)}")

@app.get("/nutrition-database/search")
async def search_foods(
    q: str = Query(..., min_length=2, description="Food name to search for"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """Search foods by name in nutrition database."""
    try:
        from utils.nutrition_database_service import nutrition_db_service
        
        foods = nutrition_db_service.search_foods_by_name(q, limit=limit)
        
        return {
            "status": "success",
            "query": q,
            "foods": foods,
            "total_found": len(foods),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching foods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search foods: {str(e)}")

@app.get("/nutrition-database/ingredients/verify")
async def verify_ingredients(
    ingredients: str = Query(..., description="Comma-separated list of ingredients to verify")
):
    """Verify ingredients against nutrition database and get nutrition data."""
    try:
        from utils.nutrition_database_service import nutrition_db_service
        
        # Parse comma-separated ingredients
        ingredient_list = [ingredient.strip() for ingredient in ingredients.split(",") if ingredient.strip()]
        
        if not ingredient_list:
            raise HTTPException(status_code=400, detail="No valid ingredients provided")
        
        # Search for ingredients in database
        verified_ingredients = nutrition_db_service.search_ingredients_for_meal(ingredient_list)
        
        # Calculate total macros for all verified ingredients
        total_macros = nutrition_db_service.calculate_meal_macros(verified_ingredients)
        
        return {
            "status": "success",
            "ingredients_requested": ingredient_list,
            "ingredients_verified": verified_ingredients,
            "total_macros": total_macros,
            "verification_summary": {
                "total_requested": len(ingredient_list),
                "total_found": len(verified_ingredients),
                "verification_rate": round((len(verified_ingredients) / len(ingredient_list)) * 100, 1) if ingredient_list else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying ingredients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify ingredients: {str(e)}")

# Note: Phase 3 endpoints removed - using agent-specific batching instead


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

# =============================================================================
# ASYNC ENDPOINTS FOR BACKGROUND PROCESSING
# =============================================================================

@app.post("/generate-meal-plan-async")
async def generate_meal_plan_async(request: dict):
    """
    Generate meal plan asynchronously.
    Returns immediately with job ID for polling.
    """
    try:
        user_id = request.get("user_id")
        preferences = request.get("preferences", {})
        dietary_restrictions = request.get("dietary_restrictions", [])
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        logger.info(f"Creating async meal plan generation task for user: {user_id}")
        
        # Import here to avoid circular imports
        from tasks import generate_meal_plan_task
        
        # Create Celery task
        task = generate_meal_plan_task.delay(
            user_id=user_id,
            preferences=preferences,
            dietary_restrictions=dietary_restrictions
        )
        
        # Return immediate response with job ID
        return {
            "status": "processing",
            "job_id": task.id,
            "message": "Meal plan generation is being processed. This may take 20-40 seconds.",
            "estimated_time": 40,
            "poll_url": f"/meal-plan/job/{task.id}",
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Async meal plan generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create meal plan generation task: {str(e)}"
        )

@app.post("/analyze-nutrition-async")
async def analyze_nutrition_async(request: dict):
    """
    Analyze nutrition data asynchronously.
    Returns immediately with job ID for polling.
    """
    try:
        user_id = request.get("user_id")
        nutrition_data = request.get("nutrition_data", {})
        date_range = request.get("date_range", "week")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        if not nutrition_data:
            raise HTTPException(status_code=400, detail="nutrition_data is required")
        
        logger.info(f"Creating async nutrition analysis task for user: {user_id}")
        
        # Import here to avoid circular imports
        from tasks import analyze_nutrition_data_task
        
        # Create Celery task
        task = analyze_nutrition_data_task.delay(
            user_id=user_id,
            nutrition_data=nutrition_data,
            date_range=date_range
        )
        
        # Return immediate response with job ID
        return {
            "status": "processing",
            "job_id": task.id,
            "message": "Nutrition analysis is being processed. This may take 15-30 seconds.",
            "estimated_time": 30,
            "poll_url": f"/nutrition-analysis/job/{task.id}",
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Async nutrition analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create nutrition analysis task: {str(e)}"
        )

@app.post("/generate-nutrition-recommendations-async")
async def generate_nutrition_recommendations_async(request: dict):
    """
    Generate nutrition recommendations asynchronously.
    Returns immediately with job ID for polling.
    """
    try:
        user_id = request.get("user_id")
        goals = request.get("goals", {})
        current_diet = request.get("current_diet", {})
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        logger.info(f"Creating async nutrition recommendations task for user: {user_id}")
        
        # Import here to avoid circular imports
        from tasks import generate_nutrition_recommendations_task
        
        # Create Celery task
        task = generate_nutrition_recommendations_task.delay(
            user_id=user_id,
            goals=goals,
            current_diet=current_diet
        )
        
        # Return immediate response with job ID
        return {
            "status": "processing",
            "job_id": task.id,
            "message": "Nutrition recommendations are being generated. This may take 15-25 seconds.",
            "estimated_time": 25,
            "poll_url": f"/nutrition-recommendations/job/{task.id}",
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Async nutrition recommendations failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create nutrition recommendations task: {str(e)}"
        )

# =============================================================================
# JOB STATUS POLLING ENDPOINTS
# =============================================================================

@app.get("/meal-plan/job/{job_id}")
async def get_meal_plan_job_status(job_id: str):
    """Get status of meal plan generation job."""
    try:
        task_result = AsyncResult(job_id)
        
        if task_result.state == 'PENDING':
            return {
                "status": "processing",
                "job_id": job_id,
                "message": "Meal plan generation is still in progress..."
            }
        elif task_result.state == 'SUCCESS':
            return {
                "status": "completed",
                "job_id": job_id,
                "result": task_result.result,
                "message": "Meal plan generated successfully!"
            }
        elif task_result.state == 'FAILURE':
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(task_result.result),
                "message": "Meal plan generation failed."
            }
        else:
            return {
                "status": task_result.state,
                "job_id": job_id,
                "message": f"Job is in {task_result.state} state"
            }
    
    except Exception as e:
        logger.error(f"Error checking meal plan job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check job status: {str(e)}"
        )

@app.get("/nutrition-analysis/job/{job_id}")
async def get_nutrition_analysis_job_status(job_id: str):
    """Get status of nutrition analysis job."""
    try:
        task_result = AsyncResult(job_id)
        
        if task_result.state == 'PENDING':
            return {
                "status": "processing",
                "job_id": job_id,
                "message": "Nutrition analysis is still in progress..."
            }
        elif task_result.state == 'SUCCESS':
            return {
                "status": "completed",
                "job_id": job_id,
                "result": task_result.result,
                "message": "Nutrition analysis completed successfully!"
            }
        elif task_result.state == 'FAILURE':
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(task_result.result),
                "message": "Nutrition analysis failed."
            }
        else:
            return {
                "status": task_result.state,
                "job_id": job_id,
                "message": f"Job is in {task_result.state} state"
            }
    
    except Exception as e:
        logger.error(f"Error checking nutrition analysis job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check job status: {str(e)}"
        )

@app.get("/nutrition-recommendations/job/{job_id}")
async def get_nutrition_recommendations_job_status(job_id: str):
    """Get status of nutrition recommendations job."""
    try:
        task_result = AsyncResult(job_id)
        
        if task_result.state == 'PENDING':
            return {
                "status": "processing",
                "job_id": job_id,
                "message": "Nutrition recommendations are still being generated..."
            }
        elif task_result.state == 'SUCCESS':
            return {
                "status": "completed",
                "job_id": job_id,
                "result": task_result.result,
                "message": "Nutrition recommendations generated successfully!"
            }
        elif task_result.state == 'FAILURE':
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(task_result.result),
                "message": "Nutrition recommendations generation failed."
            }
        else:
            return {
                "status": task_result.state,
                "job_id": job_id,
                "message": f"Job is in {task_result.state} state"
            }
    
    except Exception as e:
        logger.error(f"Error checking nutrition recommendations job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check job status: {str(e)}"
        )

