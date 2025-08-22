#!/usr/bin/env python3
"""
Simple Nutrition Agent - Groq AI Integration

This agent provides intelligent nutrition functionality using Groq AI.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from groq import Groq

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
# SIMPLE DATABASE CONNECTION
# =============================================================================

@asynccontextmanager
async def get_db():
    """Get database session - simple and reliable."""
    try:
        from shared.database import get_db as _get_db
        db = _get_db()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")

# =============================================================================
# USER VALIDATION FUNCTIONS
# =============================================================================

async def validate_user_exists(user_id: str) -> bool:
    """Validate that the user exists in the database."""
    try:
        async with get_db() as db:
            # Check if user exists in users table
            result = db.execute(
                text("SELECT id FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            ).fetchone()
            
            if not result:
                logger.warning(f"User validation failed: User {user_id} does not exist")
                return False
                
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

async def create_meal_plan(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create an intelligent meal plan using Groq AI."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        if not groq_client:
            raise HTTPException(status_code=503, detail="Groq AI client not initialized")
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of meal plan you need.")
        
        # Optional parameters with defaults
        calories = parameters.get("calories", 2000)
        meals_per_day = parameters.get("meals_per_day", 3)
        dietary_restrictions = parameters.get("dietary_restrictions", [])
        goals = parameters.get("goals", "balanced")
        
        prompt = f"""Based on this user request, create a detailed meal plan:

USER REQUEST: {description}

Additional context:
- Daily calorie target: {calories} calories
- Meals per day: {meals_per_day}
- Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
- Goals: {goals}

Provide:
1. A structured meal plan with specific foods and portions that matches their description
2. Calorie breakdown per meal
3. Nutrition highlights (protein, carbs, fat)
4. Shopping list based on their preferences
5. Preparation tips
6. Health benefits
7. Any specific recommendations based on their description

Format as a clear, actionable meal plan that directly addresses what they asked for."""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.6
        )
        
        meal_plan_text = response.choices[0].message.content
        
        meal_plan = {
            "user_id": user_id,
            "user_request": description,
            "calories": calories,
            "meals_per_day": meals_per_day,
            "dietary_restrictions": dietary_restrictions,
            "goals": goals,
            "meal_plan": meal_plan_text,
            "ai_model": "llama3-70b-8192",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        
        return {
            "status": "success",
            "meal_plan": meal_plan,
            "message": "Intelligent meal plan created successfully using AI based on your description"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")

async def log_meal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Log a meal with AI nutrition insights."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what you ate for this meal.")
        
        # Optional parameters with defaults
        meal_type = parameters.get("meal_type", "general")
        foods = parameters.get("foods", [])
        estimated_calories = parameters.get("estimated_calories", 500)
        notes = parameters.get("notes", "")
        
        meal_log = {
            "user_id": user_id,
            "description": description,
            "meal_type": meal_type,
            "foods": foods,
            "estimated_calories": estimated_calories,
            "notes": notes,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add AI nutrition insights if Groq is available
        if groq_client:
            try:
                insight_prompt = f"""Analyze this meal and provide nutrition insights:

USER DESCRIPTION: {description}

Additional context:
- Meal type: {meal_type}
- Foods consumed: {', '.join(foods) if foods else 'Based on description'}
- Estimated calories: {estimated_calories}
- Additional notes: {notes}

Provide:
1. Nutrition highlights (protein, carbs, fat, vitamins, minerals) based on their description
2. Health benefits of the foods mentioned
3. Suggestions for balance if needed
4. Tips for future meals
5. Any concerns or recommendations
6. Specific insights based on their detailed description

Keep it brief but informative and personalized to what they described."""

                insight_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": insight_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                meal_log["ai_insights"] = insight_response.choices[0].message.content
                meal_log["ai_model"] = "llama3-70b-8192"
                
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")
        
        return {
            "status": "success",
            "meal_log": meal_log,
            "message": "Meal logged successfully with AI insights"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging meal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log meal: {str(e)}")

async def get_nutrition_summary(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Get intelligent nutrition summary using Groq AI."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        if not groq_client:
            raise HTTPException(status_code=503, detail="Groq AI client not initialized")
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of nutrition summary you need.")
        
        # Optional parameters with defaults
        days = parameters.get("days", 7)
        nutrition_data = parameters.get("nutrition_data", [])
        goals = parameters.get("goals", "balanced nutrition")
        
        prompt = f"""Based on this user request, provide a comprehensive nutrition summary and analysis:

USER REQUEST: {description}

Additional context:
- Period: {days} days
- Nutrition data available: {nutrition_data}
- User goals: {goals}

Provide:
1. Summary of nutrition patterns based on their description
2. Progress toward goals mentioned in their request
3. Areas for improvement specific to what they asked for
4. Recommendations for next period
5. Health benefits achieved
6. Tips for maintaining good nutrition
7. Any specific advice based on their description

Format as a clear, actionable summary that directly addresses what they asked for."""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.6
        )
        
        summary_text = response.choices[0].message.content
        
        # Mock nutrition summary (in real app, this would come from database)
        summary = {
            "user_id": user_id,
            "user_request": description,
            "period_days": days,
            "total_meals": 21,
            "total_calories": 14000,
            "average_daily_calories": 2000,
            "ai_analysis": summary_text,
            "ai_model": "llama3-70b-8192",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting nutrition summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrition summary: {str(e)}")

async def set_nutrition_goal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Set an intelligent nutrition goal using Groq AI."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        if not groq_client:
            raise HTTPException(status_code=503, detail="Groq AI client not initialized")
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of nutrition goal you want to set.")
        
        # Optional parameters with defaults
        goal_type = parameters.get("goal_type", "general")
        target_value = parameters.get("target_value", "flexible")
        timeframe = parameters.get("timeframe", "flexible")
        current_diet = parameters.get("current_diet", "not specified")
        motivation = parameters.get("motivation", "not specified")
        
        prompt = f"""Based on this user request, create a personalized nutrition goal plan:

USER REQUEST: {description}

Additional context:
- Goal type: {goal_type}
- Target value: {target_value}
- Timeframe: {timeframe}
- Current diet: {current_diet}
- Motivation: {motivation}

Provide:
1. Realistic goal breakdown based on their description
2. Weekly progression plan
3. Tips for achieving the goal
4. Alternative food options if needed
5. Motivation strategies
6. Success metrics and milestones
7. Any specific recommendations based on their description

Format as a clear, actionable goal plan that directly addresses what they asked for."""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.6
        )
        
        goal_plan = response.choices[0].message.content
        
        goal = {
            "user_id": user_id,
            "user_request": description,
            "goal_type": goal_type,
            "target_value": target_value,
            "timeframe": timeframe,
            "current_diet": current_diet,
            "motivation": motivation,
            "goal_plan": goal_plan,
            "ai_model": "llama3-70b-8192",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        
        return {
            "status": "success",
            "goal": goal,
            "message": "Intelligent nutrition goal set successfully using AI based on your description"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting nutrition goal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set nutrition goal: {str(e)}")

async def general_nutrition_response(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Generate intelligent general nutrition response using Groq AI."""
    try:
        if not groq_client:
            raise HTTPException(status_code=503, detail="Groq AI client not initialized")
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what you need help with regarding nutrition.")
        
        # Optional parameters
        message = parameters.get("message", description)
        
        prompt = f"""You are a knowledgeable nutrition expert. The user asks: "{message}"

User's detailed description: {description}

Provide a helpful, accurate, and encouraging response about nutrition and healthy eating that directly addresses their specific request. Include:
1. Clear, actionable advice based on their description
2. Health benefits relevant to their situation
3. Practical tips they can implement today
4. Encouragement and motivation
5. Suggestions for further learning
6. Any specific recommendations based on their description

Keep the response conversational, helpful, and directly relevant to what they asked for."""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "status": "success",
            "response": ai_response,
            "user_id": user_id,
            "agent": "nutrition",
            "ai_model": "llama3-70b-8192",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating nutrition response: {e}")
        # Fallback to basic response if AI fails
        return {
            "status": "success",
            "response": "I'm here to help with nutrition! I can create meal plans, log meals, and provide nutrition advice. Please provide a description of what you need.",
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
    allow_origins=["*"],
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
    """Initialize Groq AI client on startup."""
    global groq_client
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            groq_client = Groq(api_key=groq_api_key)
            logger.info("Groq AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq AI client: {e}")
            groq_client = None
    else:
        logger.warning("GROQ_API_KEY not set, AI nutrition features will not work")
        groq_client = None

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
    """Simple health check with Groq AI status."""
    return {
        "status": "healthy",
        "agent": "nutrition",
        "groq_ai_status": "connected" if groq_client else "disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Nutrition agent is running smoothly"
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
        if tool_name == "create_meal_plan":
            return await create_meal_plan(parameters, user_id)
        elif tool_name == "log_meal":
            return await log_meal(parameters, user_id)
        elif tool_name == "get_nutrition_summary":
            return await get_nutrition_summary(parameters, user_id)
        elif tool_name == "set_nutrition_goal":
            return await set_nutrition_goal(parameters, user_id)
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
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
