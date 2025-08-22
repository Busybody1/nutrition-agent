#!/usr/bin/env python3
"""
Simple Nutrition Agent - OpenAI GPT Integration

This agent provides intelligent nutrition functionality using OpenAI GPT.
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
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLE NUTRITION AGENT
# =============================================================================

# Simple in-memory storage (for demo purposes)
nutrition_data = {}

# Initialize OpenAI client
openai_client = None

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
# NUTRITION FUNCTIONS WITH OPENAI
# =============================================================================

async def create_meal_plan(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create an intelligent meal plan using OpenAI GPT."""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not initialized")
        
        calories = parameters.get("calories", 2000)
        meals_per_day = parameters.get("meals_per_day", 3)
        dietary_restrictions = parameters.get("dietary_restrictions", [])
        goals = parameters.get("goals", "balanced")
        
        prompt = f"""Create a detailed meal plan for a person with:
- Daily calorie target: {calories} calories
- Meals per day: {meals_per_day}
- Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
- Goals: {goals}

Provide:
1. A structured meal plan with specific foods and portions
2. Calorie breakdown per meal
3. Nutrition highlights (protein, carbs, fat)
4. Shopping list
5. Preparation tips
6. Health benefits

Format as a clear, actionable meal plan."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        meal_plan_text = response.choices[0].message.content
        
        meal_plan = {
            "user_id": user_id,
            "calories": calories,
            "meals_per_day": meals_per_day,
            "dietary_restrictions": dietary_restrictions,
            "goals": goals,
            "meal_plan": meal_plan_text,
            "ai_model": "gpt-4o",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "status": "success",
            "meal_plan": meal_plan,
            "message": "Intelligent meal plan created successfully using AI"
        }
        
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")

async def calculate_calories(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Calculate intelligent calorie needs using OpenAI GPT."""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not initialized")
        
        age = parameters.get("age", 30)
        weight = parameters.get("weight", 70)
        height = parameters.get("height", 170)
        activity_level = parameters.get("activity_level", "moderate")
        gender = parameters.get("gender", "not specified")
        goals = parameters.get("goals", "maintenance")
        
        prompt = f"""Calculate personalized calorie needs for:
- Age: {age} years
- Weight: {weight} kg
- Height: {height} cm
- Activity level: {activity_level}
- Gender: {gender}
- Goals: {goals}

Provide:
1. BMR (Basal Metabolic Rate) calculation
2. TDEE (Total Daily Energy Expenditure) for different activity levels
3. Recommended calorie intake for the specified goal
4. Macronutrient breakdown (protein, carbs, fat percentages)
5. Meal timing suggestions
6. Additional nutrition advice

Format as clear, actionable recommendations with calculations."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.5
        )
        
        calorie_analysis = response.choices[0].message.content
        
        return {
            "status": "success",
            "calories": {
                "age": age,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "gender": gender,
                "goals": goals,
                "analysis": calorie_analysis,
                "ai_model": "gpt-4o"
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating calories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate calories: {str(e)}")

async def get_nutrition_recommendations(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Get intelligent nutrition recommendations using OpenAI GPT."""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not initialized")
        
        focus_area = parameters.get("focus_area", "balanced")
        current_diet = parameters.get("current_diet", "standard")
        health_conditions = parameters.get("health_conditions", [])
        fitness_level = parameters.get("fitness_level", "beginner")
        
        prompt = f"""Provide comprehensive nutrition recommendations for:
- Focus area: {focus_area}
- Current diet: {current_diet}
- Health conditions: {', '.join(health_conditions) if health_conditions else 'None'}
- Fitness level: {fitness_level}

Provide:
1. Specific food recommendations
2. Foods to avoid or limit
3. Meal timing and frequency
4. Supplement suggestions (if applicable)
5. Hydration guidelines
6. Long-term nutrition strategy
7. Tips for sustainable changes

Format as practical, actionable advice."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.6
        )
        
        recommendations_text = response.choices[0].message.content
        
        recommendations = {
            "user_id": user_id,
            "focus_area": focus_area,
            "current_diet": current_diet,
            "health_conditions": health_conditions,
            "fitness_level": fitness_level,
            "recommendations": recommendations_text,
            "ai_model": "gpt-4o"
        }
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting nutrition recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrition recommendations: {str(e)}")

async def general_nutrition_response(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Generate intelligent general nutrition response using OpenAI GPT."""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not initialized")
        
        message = parameters.get("message", "Tell me about nutrition")
        
        prompt = f"""You are a knowledgeable nutrition expert. The user asks: "{message}"

Provide a helpful, accurate, and encouraging response about nutrition. Include:
1. Clear, actionable advice
2. Scientific backing where appropriate
3. Practical tips they can implement today
4. Encouragement and motivation
5. Suggestions for further learning

Keep the response conversational and helpful."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "status": "success",
            "response": ai_response,
            "user_id": user_id,
            "agent": "nutrition",
            "ai_model": "gpt-4o",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating nutrition response: {e}")
        # Fallback to basic response if AI fails
        return {
            "status": "success",
            "response": "I'm here to help with nutrition! I can create meal plans, calculate calories, and provide recommendations.",
            "user_id": user_id,
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Nutrition Agent - OpenAI GPT Integration",
    description="Intelligent nutrition agent with OpenAI GPT for personalized nutrition advice",
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
    """Initialize OpenAI client on startup."""
    global openai_client
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            openai_client = None
    else:
        logger.warning("OPENAI_API_KEY not set, AI nutrition features will not work")
        openai_client = None

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
    """Simple health check with OpenAI status."""
    return {
        "status": "healthy",
        "agent": "nutrition",
        "openai_status": "connected" if openai_client else "disconnected",
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
        if tool_name == "meal_plan":
            return await create_meal_plan(parameters, user_id)
        elif tool_name == "calculate_calories":
            return await calculate_calories(parameters, user_id)
        elif tool_name == "nutrition_recommendations":
            return await get_nutrition_recommendations(parameters, user_id)
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
